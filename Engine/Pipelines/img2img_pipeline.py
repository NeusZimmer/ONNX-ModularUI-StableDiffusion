#from sched import scheduler
from Engine.General_parameters import Engine_Configuration
from Engine import Vae_and_Text_Encoders

from Engine.pipelines_engines import SchedulersConfig
from Engine.pipelines_engines import seed_generator

#optimum pipes
from optimum.onnxruntime import ORTStableDiffusionImg2ImgPipeline

import gc
import numpy as np




class Borg5:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state


class img2img_pipe(Borg5):
    img2img_pipe = None
    model = None
    seeds = []
    def __init__(self):
        Borg5.__init__(self)

    def __str__(self): 
        import json
        return json.dumps(self.__dict__)

    def initialize(self,model_path,sched_name):
        #from Engine.General_parameters import Engine_Configuration as en_config
        if Vae_and_Text_Encoders().text_encoder == None:
            Vae_and_Text_Encoders().load_textencoder(model_path)
        if Vae_and_Text_Encoders().vae_decoder == None:
            Vae_and_Text_Encoders().load_vaedecoder(model_path)
        if Vae_and_Text_Encoders().vae_encoder == None:
            Vae_and_Text_Encoders().load_vaeencoder(model_path)

        provider=Engine_Configuration().MAINPipe_provider['provider']
        provider_options=Engine_Configuration().MAINPipe_provider['provider_options']


        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level=3

        unet_session=load_unet_model(model_path, provider=provider, sess_options=sess_options, provider_options=provider_options)
        tokenizer,config=load_tokenizer(model_path)

        if self.img2img_pipe == None:
            print(f"Loading Img2Img pipe in {provider} with options: {provider_options}")

            self.img2img_pipe =ORTStableDiffusionImg2ImgPipeline(
                unet_session=unet_session,
                tokenizer=tokenizer,
                config=config,
                scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                text_encoder_session=Vae_and_Text_Encoders().text_encoder,
                vae_decoder_session=Vae_and_Text_Encoders().vae_decoder,
                vae_encoder_session=Vae_and_Text_Encoders().vae_encoder,
            )
        else:
             self.img2img_pipe.scheduler=SchedulersConfig().scheduler(sched_name,model_path)
        return self.img2img_pipe

    def create_seeds(self,seed=None,iter=1,same_seeds=False):
        self.seeds=seed_generator(seed,iter)
        if same_seeds:
            for seed in self.seeds:
                seed = self.seeds[0]

    def unload_from_memory(self):
        self.img2img_pipe= None
        self.model = None
        gc.collect()


    def run_inference(self,prompt,neg_prompt,init_image,strength,steps,guid,eta,batch,seed):
        import numpy as np
        rng = np.random.RandomState(seed)
        prompt.strip("\n")
        neg_prompt.strip("\n")


        batch_images = self.img2img_pipe(
            prompt,
            negative_prompt=neg_prompt,
            image=init_image,
            strength= strength,
            num_inference_steps=steps,
            guidance_scale=guid,
            eta=eta,
            num_images_per_prompt=batch,
            generator=rng,
        ).images
        dictio={'Img2ImgPrompt':prompt,'neg_prompt':neg_prompt,'steps':steps,'guid':guid,'eta':eta,'strength':strength,'seed':seed}
        return batch_images,dictio


def load_unet_model(model_path,provider, sess_options, provider_options,model_name: str = None):
    from optimum.onnxruntime.modeling_ort import ORTModel
    if model_name==None:
        model_name="/unet/model.onnx"
    return ORTModel.load_model(model_path+model_name, provider,sess_options, provider_options=provider_options)


def load_tokenizer(model_path):
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