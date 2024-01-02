#This file exist mainly for better area definition understanding.

def show_new_tabs(list_modules,tab="main_ui"):
    pass

def show_tools_area(list_modules):
    _all_areas_process(list_modules,"tools_area") #Area of tool modules

def show_prompt_preprocess_area(list_modules):
    _all_areas_process(list_modules,"prompt_process") #Area of modules for processing prompts

def show_image_postprocess_area(list_modules):
    _all_areas_process(list_modules,"image_postprocess")#Area of modules for postprocessing images

def show_footer_area(list_modules,tab="NoUI"):
    _all_areas_process(list_modules,"footer",tab)#Area of modules for postprocessing images

def _all_areas_process(list_modules,area_name,tab="NoUI"):
    for module in list_modules:
        show=False
        if tab=="main_ui" and module['is_global_ui'](): show=True
        if tab!="main_ui" and not module['is_global_ui']():
            if module['ui_position']==area_name: 
                show= True
        
        if show:
            module['show']() #modules[2]= show, #modules[3] =process


if __name__ == "main":
    print("This file is not intended to run as standalone")
