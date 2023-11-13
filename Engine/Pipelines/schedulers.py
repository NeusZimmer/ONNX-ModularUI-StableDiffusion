from sched import scheduler
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