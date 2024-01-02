from Engine.General_parameters import Engine_Configuration


import gc,json
import numpy as np

from Engine.General_parameters import Engine_Configuration
from Engine import SchedulersConfig
#from Engine.pipelines_engines import SchedulersConfig
from Engine.engine_common_funcs import seed_generator
from Engine import Vae_and_Text_Encoders


#optimum pipes
from optimum.onnxruntime import ORTStableDiffusionInpaintPipeline

class Borg_Inpaint:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state


class inpaint_pipe(Borg_Inpaint):
    inpaint_pipe = None
    seeds = []
    def __init__(self):
        Borg_Inpaint.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def initialize(self,model_path,sched_name):
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

        if self.inpaint_pipe == None:
            from optimum.onnxruntime.modeling_ort import ORTModel
            from Engine.engine_common_funcs import load_tokenizer_and_config

            print(f"Loading Inpaint Unet session in [{provider}] with options:{provider_options}")            
            unet_session = ORTModel.load_model(model_path+"/unet/model.onnx", provider,sess_options, provider_options=provider_options)
            tokenizer,config=load_tokenizer_and_config(model_path)  

            self.inpaint_pipe = ORTStableDiffusionInpaintPipeline(
                unet_session=unet_session,
                vae_decoder_session= Vae_and_Text_Encoders().vae_decoder,
                text_encoder_session= Vae_and_Text_Encoders().text_encoder,
                vae_encoder_session=Vae_and_Text_Encoders().vae_encoder,
                tokenizer=tokenizer,
                config=config,
                scheduler=SchedulersConfig().scheduler(sched_name,model_path)
            )
  
        else:
             self.inpaint_pipe.scheduler=SchedulersConfig().scheduler(sched_name,model_path)
        return self.inpaint_pipe

    def create_seeds(self,seed=None,iter=1,same_seeds=False):
        self.seeds=seed_generator(seed,iter)
        if same_seeds:
            for seed in self.seeds:
                seed = self.seeds[0]

    def unload_from_memory(self):
        self.inpaint_pipe= None
        self.seeds= None
        gc.collect()


    def run_inference(self,prompt,neg_prompt,init_image,init_mask,height,width,steps,guid,eta,batch,seed):
        import numpy as np
        rng = np.random.RandomState(seed)
        prompt.strip("\n")
        neg_prompt.strip("\n")


        batch_images = self.inpaint_pipe(
            prompt,
            negative_prompt=neg_prompt,
            image=init_image,
            mask_image=init_mask,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guid,
            eta=eta,
            num_images_per_prompt=batch,
            generator=rng,
        ).images

        dictio={'prompt':prompt,'neg_prompt':neg_prompt,'height':height,'width':width,'steps':steps,'guid':guid,'eta':eta,'batch':batch,'seed':seed}
        return batch_images,dictio

