import gradio as gr
#from Engine.General_parameters import ControlNet_config
from UI import styles_ui
global styles_dict

def show_styles_ui():
    global styles_dict
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
    """dict={
            "None":True,
            "StudioPhoto":"(RAW, 8k) |, studio lights,pseudo-impasto",
            "Style1":"(cartoon) |, Ink drawing line art",
            "Style2":"unity wallpaper, 8k, high quality, | masterpiece,(masterpiece, top quality, best quality)"
        }"""
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
    global styles_dict
    dict=styles_dict
    style=args[0]

    from Engine.General_parameters import running_config
    Running_information= running_config().Running_information    
    Running_information.update({"Style":False})

    if style != "None":
        Running_information.update({"Style":dict[style]})
        params=dict[style].split("|")
        return params[0],params[1]        
    else:
        return "",""

def saveto_memory_style(*args):
    import json
    global styles_dict
    styles_dict

    style=args[0]
    style_pre=args[1]
    style_post=args[2]
       

    from Engine.General_parameters import running_config
    Running_information= running_config().Running_information    
    Running_information.update({"Style":False})

    if style != "None":
        styles_dict.update({style:f"{style_pre}|{style_post}"})
        Running_information.update({"Style":styles_dict[style]})


