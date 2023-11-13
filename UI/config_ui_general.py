import gradio as gr
import os

from Engine.General_parameters import UI_Configuration as UI_Configuration


models_dir=f"{os.getcwd()}"+'\\models'

def show_general_configuration():
    ui_config=UI_Configuration()
    with gr.Blocks(title="General Config") as ConfigUI:
        with gr.Accordion(label="UI Options",open=False):
            gr.Markdown("Directories configuracion")
            with gr.Row():
                Models_Dir_Select=gr.Textbox(label="Models Directory",lines=1, value=ui_config.models_dir, visible=True, interactive=True)
                Output_Path_Select=gr.Textbox(label="Output Directory",lines=1, value=ui_config.output_path, visible=True, interactive=True)
            with gr.Row():
                Txt2img_Tab = gr.Checkbox(label="Txt2img Tab",value=ui_config.Txt2img_Tab, info="De/Activate TAB(Applied in next run)")
                InPaint_Tab = gr.Checkbox(label="In-Paint Tab",value=ui_config.InPaint_Tab, info="De/Activate TAB(Applied in next run)")
                Img2Img_Tab = gr.Checkbox(label="Img2Img Tab",value=ui_config.Img2Img_Tab, info="De/Activate TAB(Applied in next run)")
                InstructP2P_Tab = gr.Checkbox(label="InstructP2P Tab",value=ui_config.InstructP2P_Tab, info="De/Activate TAB(Applied in next run)")
                Tools_Tab = gr.Checkbox(label="Image Tools Tab",value=ui_config.Tools_Tab, info="De/Activate TAB(Applied in next run)")
                ControlNet_Tab = gr.Checkbox(label="ControlNet Tab",value=ui_config.ControlNet_Tab, info="De/Activate TAB(Applied in next run)")                

            with gr.Row():
                Advanced_Config = gr.Checkbox(label="Advanced Config",value=ui_config.Advanced_Config, info="Deactivate Avanced Options(Applied in next run)")                
                UI_NetworkPort=gr.Textbox(label="Gradio Server Port",lines=1, value=ui_config.GradioPort, visible=True, interactive=True)
            with gr.Row():
                apply_btn = gr.Button("Apply & Save config", variant="primary")
                #loadconfig_btn = gr.Button("Load saved config")
        with gr.Row():                
            from UI import edit_styles_ui
            edit_styles_ui.show_edit_styles_ui()

    UI_options=[Models_Dir_Select,Output_Path_Select,Txt2img_Tab, InPaint_Tab, Img2Img_Tab,InstructP2P_Tab,Tools_Tab, ControlNet_Tab, Advanced_Config,UI_NetworkPort]
    apply_btn.click(fn=applyandsave, inputs=UI_options,outputs=None)
    #loadconfig_btn.click(fn=loadconfig, inputs=Img2Img_Tab,outputs=Img2Img_Tab)

def loadconfig(grImg2Img_Tab):
    ui_config=UI_Configuration()
    ui_config.Txt2img_Tab=False
    return gr.Checkbox.update(value=ui_config.Txt2img_Tab)



def applyandsave(Models_Dir_Select,Output_Path_Select,Txt2img_Tab,
                 InPaint_Tab, Img2Img_Tab,InstructP2P_Tab,Tools_Tab,
                 ControlNet_Tab, Advanced_Config,UI_NetworkPort):
    
    ui_config=UI_Configuration()
    ui_config.models_dir=Models_Dir_Select
    ui_config.output_path=Output_Path_Select
    ui_config.Txt2img_Tab=Txt2img_Tab
    ui_config.InPaint_Tab=InPaint_Tab
    ui_config.Img2Img_Tab=Img2Img_Tab
    ui_config.ControlNet_Tab=ControlNet_Tab
    ui_config.Tools_Tab=Tools_Tab
    ui_config.Advanced_Config=Advanced_Config
    ui_config.InstructP2P_Tab = int(InstructP2P_Tab)
    ui_config.GradioPort = UI_NetworkPort

    #print(ui_config)
    print("Applied and saved, to work with these settings: clean memory and run any pipeline")
    ui_config.save_config_json()


def Generic_Select_Option(Radio_Select):
    config=UI_Configuration()
    if Radio_Select == "Yes":
        config.wildcards_activated=True
        print(params.UI_Configuration().wildcards_activated)
    else:
        config.wildcards_activated=False
        print(params.UI_Configuration().wildcards_activated)

def load_values():
    return

