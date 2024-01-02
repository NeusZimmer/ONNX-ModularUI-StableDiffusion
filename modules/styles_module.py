

#### MODULE NECESSARY FUNCTIONS (__init__ , show and __call__) ####
def __init__(*args):
    __name__='StylesModule'
    print(args[0])
    #here a check of access of initializacion  if needed
    #return (["txt2img","hires"],"prompt_process")
    # must return a dict: tabs: in which tabs is to be shown,ui_position: area within the UI tab and func_processing where is to be processed the data
    return {
        'tabs':["txt2img","hires"],
        'ui_position':"prompt_process",
        'func_processing':'prompt_process'}

def is_global_ui():
    return False

def is_global_function():
    return False

def show():
    show_styles_ui()

def __call__(datos):
    #what to do when the module is call __call__
    #print("Processing Styles Module2")
    if type(datos)==dict:
        #print("Entrando Styles como dict")
        prompt = datos['prompt_t0']
        prompt = process_prompt(prompt)
        datos.update({'prompt_t0':prompt})
    elif type(datos)==str:
        #print("Entrando Styles como str")
        datos=process_prompt(datos)
    else:
        print("Not recognized input for module Styles %s" % type(datos)) 
        datos=None

    return datos



##### MAIN MODULE CODE #####

import gradio as gr
global styles_dict,style

def show_styles_ui():
    global styles_dict
    global style
    style=None  
    styles_dict= get_styles()
    styles_keys= list(styles_dict.keys())
    if True:
        with gr.Accordion(label="Styles",open=False):
            gr.Markdown("Use your preferred Styles")
            with gr.Row():
                Style_Select = gr.Radio(styles_keys,value=styles_keys[0],label="Available Styles")
            with gr.Row():
                with gr.Accordion(label="Style - modificate running config",open=False):                    
                    styletext_pre = gr.Textbox(value="", lines=2, label="Style previous text")
                    styletext_post = gr.Textbox(value="", lines=2, label="Style posterior text")
            with gr.Row():
                apply_btn = gr.Button("Apply Modified Style")
                reload_btn = gr.Button("Reload Styles")

        all_inputs=[Style_Select,styletext_pre,styletext_post]

        apply_btn.click(fn=saveto_memory_style, inputs=all_inputs, outputs=None)
        reload_btn.click(fn=reload_styles, inputs=None, outputs=Style_Select)        
        Style_Select.change(fn=apply_styles, inputs=Style_Select, outputs=[styletext_pre,styletext_post])

def get_styles():
    import json
    with open('./Engine/config_files/Styles.json', 'r') as openfile:
        jsonStr = json.load(openfile)

    jsonStr.update({"None":True})
    return jsonStr

def reload_styles(*args):
    global styles_dict
    styles_dict=get_styles()
    styles_keys= list(styles_dict.keys())
    apply_styles("None")
    return  gr.Radio.update(choices=styles_keys,value="None")    

def apply_styles(*args):
    global styles_dict,style
    dict=styles_dict
    style_selected=args[0]
    style=False

    if style_selected != "None":
        style=dict[style_selected]
        params=dict[style_selected].split("|")
        return params[0],params[1]        
    else:
        return "",""

def saveto_memory_style(*args):
    global styles_dict,style
    styles_dict

    style_selected=args[0]
    style_pre=args[1]
    style_post=args[2]
       
    style=False

    if style_selected != "None":
        styles_dict.update({style_selected:f"{style_pre}|{style_post}"})
        style=styles_dict[style_selected]

def process_prompt(prompt):
    style_pre =""
    style_post=""
    global style

    style2=style
    if style2:
        styles=style2.split("|")
        style_pre =styles[0]
        style_post=styles[1]

    return style_pre+" " +prompt+" " +style_post

if __name__ == "__main__":
    print("This is a module not intended to run as standalone")
    pass
