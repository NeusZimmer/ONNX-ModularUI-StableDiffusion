# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTokenizer

#from ...configuration_utils import FrozenDict
from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import PIL_INTERPOLATION, deprecate, logging
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


from pathlib import Path
from tempfile import TemporaryDirectory
from transformers import CLIPFeatureExtractor, CLIPTokenizer, AutoConfig
from optimum.onnxruntime import ORTStableDiffusionPipeline
import onnxruntime as ort
from optimum.onnxruntime.modeling_ort import ORTModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess with 8->64
def preprocess(image):

    warnings.warn(
        (
            "The preprocess method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor.preprocess instead"
        ),
        FutureWarning,
    )
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 64 for x in (w, h))  # resize to integer multiple of 64

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class OnnxOptimumStableDiffusionHiResPipeline(ORTStableDiffusionPipeline):
    r"""
        TODO
    """
    vae_encoder: OnnxRuntimeModel
    vae_decoder: OnnxRuntimeModel
    text_encoder: OnnxRuntimeModel
    tokenizer: CLIPTokenizer
    unet: OnnxRuntimeModel
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]
    safety_checker: OnnxRuntimeModel
    feature_extractor: CLIPImageProcessor

    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        model_path,
        provider,
        vae_decoder_session: ort.InferenceSession,
        text_encoder_session: ort.InferenceSession,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        vae_encoder_session: Optional[ort.InferenceSession] = None,
        text_encoder_2_session: Optional[ort.InferenceSession] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        requires_safety_checker: bool = True,
        sess_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[dict] = None,
        low_res_model_path: Optional[str] = None,
        low_res_provider:Optional[str] = None,
        low_res_provider_options:Optional[str] = None,
        ):

        model_name=(model_path.split('\\'))[-1]
        print(f"loading Hi-Res Model:{model_name}")
        unet_session = ORTModel.load_model(model_path+"/unet/model.onnx", provider,sess_options, provider_options=provider_options)

        tokenizer,config=load_config_and_tokenizer(model_path)  
        super().__init__(
                    vae_decoder_session,text_encoder_session,unet_session,
                    config, tokenizer, scheduler,
                    vae_encoder_session=vae_encoder_session,
                    )        
        self.unet_session=self.unet


        if (low_res_model_path!=None) and not(low_res_model_path==model_path):
            model_name=(low_res_model_path.split('\\'))[-1]
            print(f"loading Low-Res Model:{model_name}")
            low_unet_session = ORTModel.load_model(low_res_model_path+"/unet/model.onnx", low_res_provider,sess_options, provider_options=low_res_provider_options)
            tokenizer2,config2=load_config_and_tokenizer(low_res_model_path)
            super().__init__(
                        vae_decoder_session,text_encoder_session,low_unet_session,
                        config2, tokenizer2, scheduler,
                        vae_encoder_session=vae_encoder_session,
                        )
            self.low_unet_session=self.unet            

        else:
            print("Low_res = Hi-Res Model")
            self.low_unet_session = self.unet_session

                  
            


        if False: #borrar cuando funcione
            print("inicializacion hires")
            print(vae_decoder_session )
            print(vae_encoder_session )
            print(text_encoder_session)
            print(sess_options)
            print(tokenizer)
            print(scheduler)





        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)
        """
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
        """
        

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion.OnnxStableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: Optional[int],
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[str],
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`np.ndarray`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="np").input_ids

            if not np.array_equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]

        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            negative_prompt_embeds = self.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def check_inputs_img2img(
        self,
        prompt: Union[str, List[str]],
        callback_steps: int,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
    ):
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def check_inputs_txt2img(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int],
        width: Optional[int],
        callback_steps: int,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def __2call__(**args):
        return
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,        
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        hires_height: Optional[int] = 576,
        hires_width: Optional[int] = 576,
        num_inference_steps: Optional[int] = 24,
        num_hires_steps: Optional[int] = 24,        
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        strength=0.8,
        strength_var=0,
        hires_steps=2,
        provide_imgs_for_all_hires_steps=True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1
    ): 
        # check inputs. Raise error if not correct
        self.check_inputs_txt2img(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)
    
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2) of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf
        #`guidance_scale = 1`corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = np.random
        #process_latent : if low-res is equal to high-res, avoid VAE postprocress for txt2img call and preprocess in img2img call
        process_latent=True if ((hires_height==height) and (hires_width==width)) else False

        latent=latents
        #if latent!=None:
        if type(latent) is np.ndarray:
            skip_to_img2img=True
            process_latent=True
        else:
            skip_to_img2img=False
        

        if not process_latent:
            print("Warm up")
            self.unet=self.unet_session
            self.txt2img_call(
                hires_height,
                hires_width,
                1,
                guidance_scale,
                num_images_per_prompt,
                eta,
                generator,
                prompt_embeds,
                output_type,
                return_dict,
                batch_size,
                do_classifier_free_guidance,
                callback,
                callback_steps,
            )
        else:
            print(f"Skip warm-up")

        if not skip_to_img2img:
            self.unet=self.low_unet_session
            print("Low-Res Generation")
            #image_result,latent=self.txt2img_call(
            latent=self.txt2img_call(
                height,
                width,
                num_inference_steps,
                guidance_scale,
                num_images_per_prompt,
                eta,
                generator,
                prompt_embeds,
                output_type,
                return_dict,
                batch_size,
                do_classifier_free_guidance,
                callback,
                callback_steps,
            )
            generator = np.random.RandomState(1)
        else:
            print(f"Procesing latent")
        #return [self.output_to_numpy(latent)],[self.output_to_numpy(latent)]
        
        low_res_image=None
        latents_list=[]
        latents_list.append(latent)

        if not process_latent:
            Prueba=True
            if Prueba:
                #Mas rapido, pero resultado inicial queda demasiado borroso, imagen no usable, pero resultados similares en imagen final???
                import torch.nn.functional as F
                import torch.onnx
                import torch
                
                latent1=torch.from_numpy(latent)
                #latent1= F.upsample(latent1,(int(hires_height/8), int(hires_width/8)),mode='bilinear')
                latent1= F.interpolate(latent1,size=(int(hires_height/8), int(hires_width/8)), mode='bilinear')
                latent = latent1.numpy()
                process_latent=True
                image_resize=None
            else:
                #Esto es lo original                
                image_result=self.output_to_numpy(latent)
                low_res_image=image_result
                image_resize=self.resize_and_crop(image_result, hires_height, hires_width)

        else:
            image_resize=None
        self.unet=self.unet_session
        i=0
        while i<hires_steps:
            print(f"{i+1} Pass of Hi-Res Generation")   
            if i>0: process_latent=True
            i+=1

            latent=self.img2img_call(                
                prompt,
                prompt_embeds,
                image_resize,        
                num_hires_steps,
                guidance_scale,
                num_images_per_prompt,
                eta,
                generator,
                output_type,
                return_dict,
                batch_size,
                strength,
                do_classifier_free_guidance,
                latent,
                process_latent,
                callback,
                callback_steps,
            )
            strength = strength +(strength_var*i)
            #if strength < 0.0: strength=0.0
            latents_list.append(latent)


        
        images=[]
        if low_res_image is not None:
            images.append(low_res_image)
        else:
            images.append(self.output_to_numpy(latents_list[0]))
        
        
        if provide_imgs_for_all_hires_steps:
            for latent_img in latents_list[1:]:
                #print("Decoding image")
                result=self.output_to_numpy(latent_img)
                images.append(result)
        else:             
            #for latent_img in [latents_list[0],latents_list[-1]]:
                #print("Decoding low and last image")
                #result=self.output_to_numpy(latent_img)
            #print("Decoding last image")
            result=self.output_to_numpy(latents_list[-1])
            images.append(result)
        return images[0],images[1:]

    def change_model(self):
        from Engine.General_parameters import Engine_Configuration

        if " " in Engine_Configuration().MAINPipe_provider:
            provider =eval(Engine_Configuration().MAINPipe_provider)
        else:
            provider =Engine_Configuration().MAINPipe_provider

        modelpath="C:\\AMD_ML\\models\\lvl4-uber-cardos-fp16"
        self.unet = None
        unet_path = modelpath +"/unet"
        self.unet = OnnxRuntimeModel.from_pretrained(unet_path,provider=provider)


    
    def img2img_call(
        self,
        prompt: Union[str, List[str]],
        prompt_embeds:np.ndarray,
        image: Union[np.ndarray, PIL.Image.Image] = None,
        num_hires_steps: Optional[int] = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[np.random.RandomState] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        batch_size=1,
        strength: float = 0.8,
        do_classifier_free_guidance=True,
        latent=None,
        process_latent=False,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        
        num_inference_steps=num_hires_steps
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        latents_dtype = prompt_embeds.dtype

        if not process_latent:
            image = preprocess(image).cpu().numpy()
            
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.


            image = image.astype(latents_dtype)
            # encode the init image into latents and scale the latents
            init_latents = self.vae_encoder(sample=image)[0]
            init_latents = 0.18215 * init_latents
            #print("Memory size of numpy array in bytes:",init_latents.nbytes)
            path="./latents"

            np.save(f"{path}/Low_Res_TempSave.npy", init_latents)            
        else:
            init_latents= latent

        #image = self.image_processor.preprocess(image)


        if isinstance(prompt, str):
            prompt = [prompt]
        if len(prompt) > init_latents.shape[0] and len(prompt) % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {len(prompt)} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = len(prompt) // init_latents.shape[0]
            init_latents = np.concatenate([init_latents] * additional_image_per_prompt * num_images_per_prompt, axis=0)
        elif len(prompt) > init_latents.shape[0] and len(prompt) % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {len(prompt)} text prompts."
            )
        else:
            init_latents = np.concatenate([init_latents] * num_images_per_prompt, axis=0)


        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps.numpy()[-init_timestep]
        timesteps = np.array([timesteps] * batch_size * num_images_per_prompt)

        # add noise to latents using the timesteps
        noise = generator.randn(*init_latents.shape).astype(latents_dtype)
        init_latents = self.scheduler.add_noise(
            torch.from_numpy(init_latents), torch.from_numpy(noise), torch.from_numpy(timesteps)
        )
        init_latents = init_latents.numpy()

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].numpy()

        # Adapted from diffusers to extend it for other runtimes than ORT
        timestep_dtype = self.unet.input_dtype.get("timestep", np.float32)
        """En el antiguo pipeline
        timestep_dtype = next(
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]
        
        """
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)[
                0
            ]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        return latents

    def img2img_call_old(
        self,
        prompt,
        prompt_embeds,
        image: Union[np.ndarray, PIL.Image.Image] = None,
        num_hires_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        batch_size=1,
        strength=0.8,
        do_classifier_free_guidance=True,
        latent=None,
        process_latent=False,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):

        num_inference_steps=num_hires_steps
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        latents_dtype = prompt_embeds.dtype

        if not process_latent:
            image = preprocess(image).cpu().numpy()
            
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.


            image = image.astype(latents_dtype)
            # encode the init image into latents and scale the latents
            init_latents = self.vae_encoder(sample=image)[0]
            init_latents = 0.18215 * init_latents
            #print("Memory size of numpy array in bytes:",init_latents.nbytes)
            path="./latents"

            np.save(f"{path}/Low_Res_TempSave.npy", init_latents)            
        else:
            init_latents= latent

        if isinstance(prompt, str):
            prompt = [prompt]
        if len(prompt) > init_latents.shape[0] and len(prompt) % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {len(prompt)} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = len(prompt) // init_latents.shape[0]
            init_latents = np.concatenate([init_latents] * additional_image_per_prompt * num_images_per_prompt, axis=0)
        elif len(prompt) > init_latents.shape[0] and len(prompt) % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {len(prompt)} text prompts."
            )
        else:
            init_latents = np.concatenate([init_latents] * num_images_per_prompt, axis=0)

        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        #offset = int(strength*10)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)



        timesteps = self.scheduler.timesteps.numpy()[-init_timestep]
        timesteps = np.array([timesteps] * batch_size * num_images_per_prompt)


        # add noise to latents using the timesteps
        noise = generator.randn(*init_latents.shape).astype(latents_dtype)
        init_latents = self.scheduler.add_noise(
            torch.from_numpy(init_latents), torch.from_numpy(noise), torch.from_numpy(timesteps)
        )

        init_latents = init_latents.numpy()

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].numpy()

        """print(f"Fuerza:{strength}")
        print(f"offset:{offset}")
        print(f"t_start:{t_start}")   
        print(f"timesteps:{timesteps}")"""
        timestep_dtype = next(
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)[
                0
            ]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        #latents2 =  latents
        """latents = 1 / 0.18215 * latents
        # image = self.vae_decoder(latent_sample=latents)[0]
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate(
            [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
        )

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        #if output_type == "pil":
        image = self.numpy_to_pil(image)

        return image[0],latents2"""
        return latents

    def output_to_numpy(self,latents):
        latents = 1 / 0.18215 * latents
        # image = self.vae_decoder(latent_sample=latents)[0]
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate(
            [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
        )

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        #if output_type == "pil":
        image = self.numpy_to_pil(image)

        return image[0]  

    def txt2img_call(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[np.random.RandomState] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        batch_size=1,
        do_classifier_free_guidance=True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
        guidance_rescale: float = 0.0,
    ):

        height = height or self.unet.config.get("sample_size", 64) * self.vae_scale_factor
        width = width or self.unet.config.get("sample_size", 64) * self.vae_scale_factor

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config.get("in_channels", 4),
            height,
            width,
            prompt_embeds.dtype,
            generator,
            None,
        )
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # Adapted from diffusers to extend it for other runtimes than ORT
        timestep_dtype = self.unet.input_dtype.get("timestep", np.float32)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            #latent_model_input =  latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)
            noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                if guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        return latents

    def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """        
        std_text = np.std(noise_pred_text, axis=tuple(range(1, noise_pred_text.ndim)), keepdims=True)
        std_cfg = np.std(noise_cfg, axis=tuple(range(1, noise_cfg.ndim)), keepdims=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg


    def txt2img_call_orig(
        self,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        batch_size=1,
        do_classifier_free_guidance=True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        """


        # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        latents = generator.randn(*latents_shape).astype(latents_dtype)


        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * np.float64(self.scheduler.init_noise_sigma)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = next(
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)
            noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        return latents

    def resize_and_crop(self,input_image: PIL.Image.Image, height: int, width: int):
        from PIL import Image
        input_width, input_height = input_image.size

        # nearest neighbor for upscaling
        if (input_width * input_height) < (width * height):
            resample_type = Image.NEAREST
        # lanczos for downscaling
        else:
            resample_type = Image.LANCZOS

        if height / width > input_height / input_width:
            adjust_width = int(input_width * height / input_height)
            input_image = input_image.resize((adjust_width, height),
                                            resample=resample_type)
            left = (adjust_width - width) // 2
            right = left + width
            input_image = input_image.crop((left, 0, right, height))
        else:
            adjust_height = int(input_height * width / input_width)
            input_image = input_image.resize((width, adjust_height),
                                            resample=resample_type)
            top = (adjust_height - height) // 2
            bottom = top + height
            input_image = input_image.crop((0, top, width, bottom))
        return input_image
    
def load_config_and_tokenizer(model_path):
        import json
        CONFIG_NAME="model_index.json"
        config_file=model_path+"/"+CONFIG_NAME
        config=None
        try:
            import json
            with open(config_file, "r", encoding="utf-8") as reader:
                text = reader.read()
                config=json.loads(text)
        except:
            raise OSError(f"model_index.json not found in {model_path} local folder")
        
        from transformers import CLIPTokenizer
        return CLIPTokenizer.from_pretrained(model_path+"/"+'tokenizer'),config