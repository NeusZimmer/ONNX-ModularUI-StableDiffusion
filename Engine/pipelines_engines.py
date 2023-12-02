from sched import scheduler
from Engine.General_parameters import Engine_Configuration
from Engine.pipeline_onnx_stable_diffusion_instruct_pix2pix import OnnxStableDiffusionInstructPix2PixPipeline

import gc
import numpy as np

from diffusers.utils.torch_utils import randn_tensor
from diffusers import (
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
    OnnxStableDiffusionInpaintPipeline,
    OnnxStableDiffusionInpaintPipelineLegacy,
    OnnxStableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UniPCMultistepScheduler,
# Non working schedulers
    VQDiffusionScheduler,
    UnCLIPScheduler,
    KarrasVeScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    DDIMInverseScheduler,
    ScoreSdeVeScheduler
)

class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class Borg1:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state
class Borg2:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class Borg4:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class Borg6:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class SchedulersConfig(Borg):
    available_schedulers= None
    selected_scheduler= None
    _model_path = None
    _scheduler_name = None
    _low_res_scheduler = None

    def __init__(self):
        Borg.__init__(self)
        if self.available_schedulers == None:
            self._load_list()

    def __str__(self): return json.dumps(self.__dict__)

    def _load_list(self):
        self.available_schedulers= ["DPMS_ms", "DPMS_ss", "DPMS++_Heun","DPMS_Heun", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC","KDPM2-A"]
        #self.available_schedulers= ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC","VQD","UnCLIP","Karras","KDPM2-A","IPNDMS","DDIM-Inverse","SDE-1"]
        #self.available_schedulers= ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC"]

    def schedulers_controlnet_list(self):
        return ["DPMS_ms", "DPMS_ss", "DPMS++_Heun","DPMS_Heun", "DDIM", "LMS", "PNDM"]

    def reset_scheduler(self):
        return self.scheduler(self._scheduler_name,self._model_path)
    
    def low_res_scheduler(self,model_path=None):
        if model_path==None:
            model_path=self._model_path
        self._low_res_scheduler = DPMSolverSinglestepScheduler.from_pretrained(self._model_path, subfolder="scheduler",provider=['DmlExecutionProvider'])
        return self._low_res_scheduler    

    def scheduler(self,scheduler_name,model_path):
        scheduler = None
        self.selected_scheduler = None
        self._model_path = model_path
        self._scheduler_name = scheduler_name
        #provider = Engine_Configuration().Scheduler_provider
        provider = Engine_Configuration().Scheduler_provider['provider']
        provider_options=Engine_Configuration().Scheduler_provider['provider_options']
        """match scheduler_name:
            case "PNDM":
                scheduler = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "LMS":
                scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "DDIM" :
                scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "Euler" :
                scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "EulerA" :
                scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "DPMS_ms" :
                scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "DPMS_ss" :
                scheduler = DPMSolverSinglestepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)   
            case "DEIS" :
                scheduler = DEISMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "HEUN" :
                scheduler = HeunDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "KDPM2":
                scheduler = KDPM2DiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
            case "UniPC":
                scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
#Test schedulers, maybe not working
            case "VQD":
                scheduler = VQDiffusionScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
            case "UnCLIP":
                scheduler = UnCLIPScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
            case "Karras":
                scheduler = KarrasVeScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
            case "KDPM2-A":
                scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
            case "IPNDMS":
                scheduler = IPNDMScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
            case "DDIM-Inverse":
                scheduler = DDIMInverseScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
            case "DPMS_Heun":
                scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider,algorithm_type="dpmsolver", solver_type="heun") 
            case "DPMS++_Heun":
                scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider,algorithm_type="dpmsolver++", solver_type="heun") 
            case "SDE-1":
                scheduler = ScoreSdeVeScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider,algorithm_type="dpmsolver++", solver_type="heun") """
        #match scheduler_name:
        if scheduler_name=="PNDM":
            scheduler = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
        elif scheduler_name== "LMS":
            scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
        elif scheduler_name== "DDIM" :
            scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
        elif scheduler_name== "Euler" :
            scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
        elif scheduler_name== "EulerA" :
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
        elif scheduler_name== "DPMS_ms" :
            scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
        elif scheduler_name== "DPMS_ss" :
            scheduler = DPMSolverSinglestepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)   
        elif scheduler_name== "DEIS" :
            scheduler = DEISMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
        elif scheduler_name== "HEUN" :
            scheduler = HeunDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
        elif scheduler_name== "KDPM2":
            scheduler = KDPM2DiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)
        elif scheduler_name== "UniPC":
            scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
#Test schedulers, maybe not working
        elif scheduler_name== "VQD":
            scheduler = VQDiffusionScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
        elif scheduler_name== "UnCLIP":
            scheduler = UnCLIPScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
        elif scheduler_name== "Karras":
            scheduler = KarrasVeScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
        elif scheduler_name== "KDPM2-A":
            scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
        elif scheduler_name== "IPNDMS":
            scheduler = IPNDMScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
        elif scheduler_name== "DDIM-Inverse":
            scheduler = DDIMInverseScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)  
        elif scheduler_name== "DPMS_Heun":
            scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider,algorithm_type="dpmsolver", solver_type="heun") 
        elif scheduler_name== "DPMS++_Heun":
            scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider,algorithm_type="dpmsolver++", solver_type="heun") 
        elif scheduler_name== "SDE-1":
            scheduler = ScoreSdeVeScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider,algorithm_type="dpmsolver++", solver_type="heun")         
           
            
        self.selected_scheduler =scheduler
        return self.selected_scheduler


class inpaint_pipe(Borg2):
    inpaint_pipe = None
    model = None
    seeds = []
    def __init__(self):
        Borg2.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def initialize(self,model_path,sched_name,legacy):
        from Engine.General_parameters import Engine_Configuration as en_config
        if Vae_and_Text_Encoders().text_encoder == None:
            Vae_and_Text_Encoders().load_textencoder(model_path)
        if Vae_and_Text_Encoders().vae_decoder == None:
            Vae_and_Text_Encoders().load_vaedecoder(model_path)
        if Vae_and_Text_Encoders().vae_encoder == None:
            Vae_and_Text_Encoders().load_vaeencoder(model_path)


        if " " in Engine_Configuration().MAINPipe_provider:
            provider =eval(Engine_Configuration().MAINPipe_provider)
        else:
            provider =Engine_Configuration().MAINPipe_provider

        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level=3

        if self.inpaint_pipe == None:
            if legacy:
                print("Legacy")
                print(f"Loading Inpaint unet pipe in {provider}")
                self.inpaint_pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                    text_encoder=Vae_and_Text_Encoders().text_encoder,
                    vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                    vae_encoder=Vae_and_Text_Encoders().vae_encoder,
                    sess_options=sess_options                    
                )
            else:
                print("No Legacy")
                print(f"Loading Inpaint unet pipe in {provider}")
                self.inpaint_pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
                    model_path,
                    provider=provider,
                    scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                    text_encoder=Vae_and_Text_Encoders().text_encoder,
                    vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                    vae_encoder=Vae_and_Text_Encoders().vae_encoder,
                    sess_options=sess_options
                )
        else:
             self.inpaint_pipe.scheduler=SchedulersConfig().scheduler(sched_name,model_path)
        return self.inpaint_pipe

    def create_seeds(self,seed=None,iter=1,same_seeds=False):
        self.seeds=seed_generator(seed,iter)
        if same_seeds:
            for seed in seeds:
                seed = seeds[0]

    def unload_from_memory(self):
        self.inpaint_pipe= None
        self.model = None
        #self.running = False
        gc.collect()


    def run_inference(self,prompt,neg_prompt,init_image,init_mask,height,width,steps,guid,eta,batch,seed,legacy):
        import numpy as np
        rng = np.random.RandomState(seed)
        prompt.strip("\n")
        neg_prompt.strip("\n")

        if legacy is True:
            batch_images = self.inpaint_pipe(
                prompt,
                negative_prompt=neg_prompt,
                image=init_image,
                mask_image=init_mask,
                num_inference_steps=steps,
                guidance_scale=guid,
                eta=eta,
                num_images_per_prompt=batch,
                generator=rng,
            ).images
        else:
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

        dictio={'prompt':prompt,'neg_prompt':neg_prompt,'height':height,'width':width,'steps':steps,'guid':guid,'eta':eta,'batch':batch,'seed':seed,'legacy':legacy}
        return batch_images,dictio


class instruct_p2p_pipe(Borg4):
    instruct_p2p_pipe = None
    model = None
    seed = None

    def __init__(self):
        Borg4.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def initialize(self,model_path,sched_name):
        from Engine.General_parameters import Engine_Configuration as en_config
        if Vae_and_Text_Encoders().text_encoder == None:
            Vae_and_Text_Encoders().load_textencoder(model_path)
        if Vae_and_Text_Encoders().vae_decoder == None:
            Vae_and_Text_Encoders().load_vaedecoder(model_path)
        if Vae_and_Text_Encoders().vae_encoder == None:
            Vae_and_Text_Encoders().load_vaeencoder(model_path)

        if " " in Engine_Configuration().MAINPipe_provider:
            provider =eval(Engine_Configuration().MAINPipe_provider)
        else:
            provider =Engine_Configuration().MAINPipe_provider

        if self.instruct_p2p_pipe == None:
            print(f"Loading Instruct pix2pix pipe in {provider}")
            self.instruct_p2p_pipe = OnnxStableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_path,
                provider=provider,
                scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                text_encoder=Vae_and_Text_Encoders().text_encoder,
                vae_decoder=Vae_and_Text_Encoders().vae_decoder,
                vae_encoder=Vae_and_Text_Encoders().vae_encoder,
                safety_checker=None)
        else:
             self.instruct_p2p_pipe.scheduler=SchedulersConfig().scheduler(sched_name,model_path)

        return self.instruct_p2p_pipe


    def run_inference(self,prompt,input_image,steps,guid,eta):
        import numpy as np
        import torch
        prompt.strip("\n")
        generator = torch.Generator()
        generator = generator.manual_seed(self.seed)
        batch_images = self.instruct_p2p_pipe(
            prompt,
            image=input_image,
            num_inference_steps=steps,
            guidance_scale=guid,
            eta=eta,
            generator=generator,
            return_dict=False
        )
        dictio={'Pix2Pix prompt':prompt,'steps':steps,'guid':guid,'seed':self.seed}
        return batch_images[0],dictio

    def create_seed(self,seed=None):
        import numpy as np
        if seed == "" or seed == None:
            rng = np.random.default_rng()
            self.seed = int(rng.integers(np.iinfo(np.uint32).max))
        else:
            self.seed= int(seed)

    def unload_from_memory(self):
        self.instruct_p2p_pipe= None
        self.model = None
        gc.collect()

def seed_generator(seed,iteration_count):
    import numpy as np
    # generate seeds for iterations
    if seed == "" or seed == None:
        rng = np.random.default_rng()
        seed = rng.integers(np.iinfo(np.uint32).max)
    else:
        try:
            seed = int(seed) & np.iinfo(np.uint32).max
        except ValueError:
            seed = hash(seed) & np.iinfo(np.uint32).max

    # use given seed for the first iteration
    seeds = np.array([seed], dtype=np.uint32)

    if iteration_count > 1:
        seed_seq = np.random.SeedSequence(seed)
        seeds = np.concatenate((seeds, seed_seq.generate_state(iteration_count - 1)))

    return seeds