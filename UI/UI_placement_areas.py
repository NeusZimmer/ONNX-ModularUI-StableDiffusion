#This file exist mainly for better area definition understanding.



def show_prompt_preprocess_area(list_modules):
    _all_areas_process(list_modules,"prompt_process") #Area of modules for processing prompts

def show_image_postprocess_area(list_modules):
    _all_areas_process(list_modules,"image_postprocess")#Area of modules for postprocessing images

def show_footer_area(list_modules):
    _all_areas_process(list_modules,"footer")#Area of modules for postprocessing images

def _all_areas_process(list_modules,area_name):
    for module in list_modules:
        if module['ui_position']==area_name: 
            module['show']() #modules[2]= show, #modules[3] =process


if __name__ == "main":
    print("This file is not intended to run as standalone")
