import gc,json


from Engine.General_parameters import (
    Engine_Configuration,
    ControlNet_config
    )

from Engine.engine_common_funcs import load_tokenizer_and_config,seed_generator
from Engine import Vae_and_Text_Encoders
from Engine import SchedulersConfig
#from Engine.pipelines_engines import SchedulersConfig


from Engine.pipeline_onnx_stable_diffusion_controlnet import OnnxStableDiffusionControlNetPipeline


from diffusers.utils.torch_utils import randn_tensor
from diffusers import (
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionInpaintPipeline,
    OnnxStableDiffusionInpaintPipelineLegacy,
    OnnxStableDiffusionImg2ImgPipeline,
)

class Borg6:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state


class ControlNet_pipe(Borg6):
#import onnxruntime as ort
    controlnet_Model_ort= None
    #controlnet_unet_ort= None
    ControlNET_Name=None
    ControlNet_pipe = None
    unet_model =None
    seeds = []

    def __init__(self):
        Borg6.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def load_ControlNet_model(self,model_path,ControlNET_drop):
        #self.__load_ControlNet_model(model_path,ControlNET_drop)
        self.__load_ControlNet_model(ControlNET_drop)

    def __load_ControlNet_model(self,ControlNET_drop):
        import onnxruntime as ort

        provider=Engine_Configuration().ControlNet_provider['provider']
        provider_options=Engine_Configuration().ControlNet_provider['provider_options']
        print(f"Loading ControlNet module at {provider} with options:{provider_options}")
        available_models=dict(ControlNet_config().available_controlnet_models())
 

        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.log_severity_level=3   
        self.controlnet_Model_ort= None

        self.ControlNET_Name=ControlNET_drop
        ControlNet_path = available_models[ControlNET_drop]
        provider=(str(provider),dict(provider_options))
        #self.controlnet_Model_ort = OnnxRuntimeModel.from_pretrained(ControlNet_path, sess_options=opts, provider=provider)
        self.controlnet_Model_ort = OnnxRuntimeModel.from_pretrained(ControlNet_path, provider=provider, sess_options=sess_options)
        
        return self.controlnet_Model_ort

    def __load_uNet_model(self,model_path):
        #Aqui cargar con ort el modelo unicamente en el provider principal.
        print("Loading Unet module")

        provider=Engine_Configuration().MAINPipe_provider['provider']
        provider_options=Engine_Configuration().MAINPipe_provider['provider_options']

        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.log_severity_level=3
        provider=(str(provider),dict(provider_options))
        unet_model = OnnxRuntimeModel.from_pretrained(model_path + "/unet", provider=provider, sess_options=sess_options)
        return unet_model

    def initialize(self,model_path,sched_name,ControlNET_drop):
        if Vae_and_Text_Encoders().text_encoder == None:
            Vae_and_Text_Encoders().load_textencoder(model_path,old_version=True)
        if Vae_and_Text_Encoders().vae_decoder == None:
            Vae_and_Text_Encoders().load_vaedecoder(model_path,old_version=True)
        if Vae_and_Text_Encoders().vae_encoder == None:
            Vae_and_Text_Encoders().load_vaeencoder(model_path,old_version=True)

        if self.unet_model == None:
            self.unet_model=self.__load_uNet_model(model_path)

        #print("Loading Tokenizer")
        self.tokenizer,self.config=load_tokenizer_and_config(model_path)


        if self.ControlNet_pipe == None:
            print(f"Using modified model for ControlNET:{model_path}")            
            self.controlnet_Model_ort= self.__load_ControlNet_model(ControlNET_drop)

            self.ControlNet_pipe = OnnxStableDiffusionControlNetPipeline(  #cambiar desde pretrained a __init__ 
                vae_encoder=Vae_and_Text_Encoders().vae_encoder,
                vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                text_encoder=Vae_and_Text_Encoders().text_encoder,
                tokenizer= self.tokenizer,
                unet= self.unet_model,
                controlnet=self.controlnet_Model_ort,
                scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker= False
            )            
        else:
            self.ControlNet_pipe.scheduler= SchedulersConfig().scheduler(sched_name,model_path)
            if self.ControlNET_Name!=ControlNET_drop:
                self.ControlNet_pipe.controlnet= None
                gc.collect()
                self.__load_ControlNet_model(ControlNET_drop)
                self.ControlNet_pipe.controlnet= self.controlnet_Model_ort

        return self.ControlNet_pipe

    def run_inference(self,prompt,neg_prompt,input_image,width,height,eta,steps,guid,seed,pose_image=None,controlnet_conditioning_scale=1.0):
        import numpy as np
        rng = np.random.RandomState(int(seed))
        image = self.ControlNet_pipe(
            prompt,
            input_image,
            negative_prompt=neg_prompt,
            width = width,
            height = height,
            num_inference_steps = steps,
            guidance_scale=guid,
            eta=eta,
            num_images_per_prompt=1,
            generator=rng,
            controlnet_conditioning_scale=controlnet_conditioning_scale
        ).images[0]
        #Añadir el diccionario
        dictio={'prompt':prompt,'neg_prompt':neg_prompt,'steps':steps,'guid':guid,'eta':eta,'strength':controlnet_conditioning_scale,'seed':seed}        
        return image,dictio

    def create_seeds(self,seed=None,iter=1,same_seeds=False):
        self.seeds=seed_generator(seed,iter)
        if same_seeds:
            for seed in self.seeds:
                seed = self.seeds[0]

    def unload_from_memory(self):
        self.ControlNet_pipe= None
        self.controlnet_Model_ort= None
        self.unet_model=None
        gc.collect()

