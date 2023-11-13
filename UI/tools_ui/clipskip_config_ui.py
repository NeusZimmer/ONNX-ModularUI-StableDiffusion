import gradio as gr
from UI import styles_ui
global styles_dict

def show_edit_styles_ui():
    global styles_dict
    styles_dict= get_styles()
    styles_keys= list(styles_dict.keys())
    if True:
        with gr.Accordion(label="Styles",open=False):
            gr.Markdown("Edit your preferred Styles")
            with gr.Row():
                with gr.Column(scale=1):
                    Style_Select = gr.Radio(styles_keys,value=styles_keys[0],label="Available Styles")
                with gr.Column(scale=8):
                    styletext_name = gr.Textbox(value="", lines=1, label="Style name")                    
                    styletext_pre = gr.Textbox(value="", lines=2, label="Style previous text")
                    styletext_post = gr.Textbox(value="", lines=2, label="Style posterior text")
            with gr.Row():
                save_btn = gr.Button("Save this Style")
                new_style_btn = gr.Button("Create New Blank Style")                
                del_style_btn = gr.Button("Delete Selected Style")      
                reload_btn = gr.Button("Reload Styles")

        all_inputs=[Style_Select,styletext_pre,styletext_post,styletext_name]
        del_style_btn.click(fn=delete_style, inputs=Style_Select, outputs=Style_Select)
        save_btn.click(fn=save_styles, inputs=all_inputs, outputs=Style_Select)
        Style_Select.change(fn=select_style, inputs=Style_Select, outputs=[styletext_name,styletext_pre,styletext_post,Style_Select])
        new_style_btn.click(fn=add_new_style, inputs=all_inputs, outputs=Style_Select)
        reload_btn.click(fn=reload_styles, inputs=None, outputs=Style_Select)   
        


def reload_styles(*args):
    global styles_dict
    styles_dict=get_styles()
    styles_keys= list(styles_dict.keys())
    return  gr.Radio.update(choices=styles_keys,value="None")    


def add_new_style(*args):
    global styles_dict
    styles_dict.update({'NewStyle':" | "})
    return Update_StyleSelect()

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
    #Aqui añadir por si es la primera ejecucion y no existe el fichero")

    jsonStr.update({"None":True})
    return jsonStr


def select_style(*args):
    global styles_dict
    dict=styles_dict
    style=args[0]

    if style != "None":
        params=dict[style].split("|")
        return style,params[0],params[1], Update_StyleSelect()  
    else:
        return "None","","",gr.Radio.update(visible=True)

def delete_style(*args):
    import json
    global styles_dict
    styles_dict

    style=args[0]

    if style != "None":
        styles_dict.pop(style)
        jsonStr = json.dumps(styles_dict)
        with open("./Engine/config_files/Styles.json", "w") as outfile:
            outfile.write(jsonStr)
        print("Saving Styles without this Style")
    else:
        print("Cannot Delete the empty Style")

    return Update_StyleSelect()        

def Update_StyleSelect(*args):
    global styles_dict
    styles_keys= list(styles_dict.keys())
    return  gr.Radio.update(choices=styles_keys)

def save_styles(*args):
    import json
    global styles_dict
    styles_dict

    style=args[0]
    style_pre=args[1]
    style_post=args[2]
    style_name=args[3]      

    if style != "None":
        styles_dict.pop(style)
        styles_dict.update({style_name:f"{style_pre}|{style_post}"})
        jsonStr = json.dumps(styles_dict)
        with open("./Engine/config_files/Styles.json", "w") as outfile:
            outfile.write(jsonStr)
        print("Saving Style")

    return Update_StyleSelect()


