#from sched import scheduler
from Engine.General_parameters import Engine_Configuration


from diffusers import (
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
    DDPMWuerstchenScheduler,
    DDPMParallelScheduler,
    DDIMParallelScheduler,
    VQDiffusionScheduler,
    UnCLIPScheduler,
    KarrasVeScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    DDIMInverseScheduler,
    ScoreSdeVeScheduler,
    LCMScheduler
)

class Scheduler_Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class SchedulersConfig(Scheduler_Borg):
    available_schedulers= None
    selected_scheduler= None
    _model_path = None
    _scheduler_name = None
    _low_res_scheduler = None

    def __init__(self):
        Scheduler_Borg.__init__(self)
        #super.__init__(self)
        if self.available_schedulers == None:
            self._load_list()

    def __str__(self):return json.dumps(self.__dict__)

    def _load_list(self):
        self.available_schedulers= ["DDPM_Parallel","DDIM_Parallel","LCM_Scheduler","DPMS_ms", "DPMS_ss", "DPMS++_Heun","DPMS_Heun", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC","KDPM2-A"]
        #self.available_schedulers= ["DDPMWuerstchenScheduler","DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC","VQD","UnCLIP","Karras","KDPM2-A","IPNDMS","DDIM-Inverse","SDE-1"]
        #self.available_schedulers= ["DPMS_ms", "DPMS_ss", "EulerA", "Euler", "DDIM", "LMS", "PNDM", "DEIS", "HEUN", "KDPM2", "UniPC"]

    
    def schedulers_controlnet_list(self):
        return ["DPMS_ms", "DPMS_ss", "DPMS++_Heun","DPMS_Heun", "DDIM", "LMS", "PNDM"]

    def reset_scheduler(self):
        return self.scheduler(self._scheduler_name,self._model_path)
    
    def low_res_scheduler(self,model_path=None):
        if model_path==None:
            model_path=self._model_path
        self._low_res_scheduler = DPMSolverSinglestepScheduler.from_pretrained(self._model_path, subfolder="scheduler",provider=['DmlExecutionProvider'],kwargs={"use_karras_sigmas":True,"beta_schedule":"scaled_linear",})
        return self._low_res_scheduler    

    def reload(self):
        print(f"Reloading Scheduler...({self._scheduler_name})")
        name=self._scheduler_name
        path=self._model_path
        new_class=SchedulersConfig()
        new_class.scheduler(name,path)

        del self
        self=new_class
        return self.selected_scheduler
        #return self.scheduler(name,path)

    def scheduler(self,scheduler_name,model_path):
        scheduler = None
        self.selected_scheduler = None
        self._model_path = model_path
        self._scheduler_name = scheduler_name
        provider = Engine_Configuration().Scheduler_provider
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
            scheduler = HeunDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider,use_karras_sigmas=True,prediction_type="epsilon",beta_schedule="linear")
            #beta_schedule: str = "linear"  or "scaled_linear"
            #sample + linear + karras;True/False == ruido, no vale
            #sample + scaled_linear + karras;True/False == ruido, no vale
            #epsilon + linear + karras;True == Primera ok segunda en negro (hires), con karras en false, la segunda error, la primera mas borrosa.
            #epsilon + scaled_linear + karras;True == 
            #prediction_type: str= epsilon, sample, v_prediction
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
            scheduler = ScoreSdeVeScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)         
        elif scheduler_name== "DDPMWuerstchenScheduler":
            scheduler = DDPMWuerstchenScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)         
        elif scheduler_name== "DDPM_Parallel":
            scheduler = DDPMParallelScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)         
            #scheduler = DDPMParallelScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider,clip_sample=True ,clip_sample_range=0.8)         
        elif scheduler_name== "DDIM_Parallel":                                   
            scheduler = DDIMParallelScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)         
        elif scheduler_name== "LCM_Scheduler":                                   
            scheduler = LCMScheduler.from_pretrained(model_path, subfolder="scheduler",provider=provider)                

        self.selected_scheduler =scheduler
        #return self.selected_scheduler
        return scheduler

