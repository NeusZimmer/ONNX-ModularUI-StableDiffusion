import gc,os
from Engine import pipelines_engines

from Engine.General_parameters import running_config
from Engine.General_parameters import UI_Configuration
#from Engine.Pipelines.txt2img_hires import txt2img_hires_pipe   
#from Engine.Pipelines.txt2img_pipeline import  txt2img_pipe
from Engine import (
    txt2img_pipe,
    Vae_and_Text_Encoders,
    txt2img_hires_pipe,
    ControlNet_pipe,
    img2img_pipe
    )


def clean_memory_click():
    print("Cleaning Memory")
    Vae_and_Text_Encoders().unload_from_memory()
    txt2img_pipe().unload_from_memory()
    txt2img_hires_pipe().unload_from_memory()
    ControlNet_pipe().unload_from_memory()
    img2img_pipe().unload_from_memory()
    #pipelines_engines.inpaint_pipe().unload_from_memory()
    #pipelines_engines.instruct_p2p_pipe().unload_from_memory()
    
    gc.collect()    

def cancel_iteration():
    running_config().Running_information.update({"cancelled":True})
    print("\nCancelling at the end of the current iteration")



def get_model_list(pipeline=None):
    model_list = []
    inpaint_model_list = []
    controlnet_model_list = []
    i2p_model_list = []
    txt2img_model_list = []

    try:
        with os.scandir(UI_Configuration().models_dir) as scan_it:
            for entry in scan_it:
                if entry.is_dir():
                    model_list.append(entry.name)

    except:
        model_list.append("Models directory does not exist, configure it")

    if len(model_list)>0:
        for model in model_list:
            if "inpaint" in model.lower(): inpaint_model_list.append(model)
            elif "controlnet" in model.lower(): controlnet_model_list.append(model)
            elif "ip2p" in model.lower(): i2p_model_list.append(model)
            else: txt2img_model_list.append(model)

    if pipeline=="txt2img": retorno=txt2img_model_list
    elif pipeline=="inpaint": retorno=inpaint_model_list
    elif pipeline=="controlnet": retorno= controlnet_model_list
    elif pipeline=="ip2p": retorno= i2p_model_list
    else: retorno= model_list  

    return retorno