import gradio as gr
from Engine.General_parameters import Engine_Configuration
from Engine.General_parameters import UI_Configuration
from Engine.General_parameters import running_config

def init_ui():
    ui_config=UI_Configuration()
    with gr.Blocks(title="ONNX Difussers Modular UI",css= css1) as demo:
        if ui_config.Txt2img_Tab:
            with gr.Tab(label="Testing HiRes Txt2img Pipeline") as tab10:
                from UI import HiRes_txt2img_ui as HiRes_txt2img_ui
                HiRes_txt2img_ui.show_HiRes_txt2img_ui()
        if ui_config.Txt2img_Tab:
            with gr.Tab(label="Txt2img Pipelines & Inferences") as tab0:
                from UI import txt2img_ui as txt2img_ui
                txt2img_ui.show_txt2img_ui()
        #if ui_config.Img2Img_Tab:
        if False:            
            with gr.Tab(label="Img2Img") as tab1:
                from UI import Img2Img_ui
                Img2Img_ui.show_Img2Img_ui()
        #if ui_config.InPaint_Tab:
        if False:            
            with gr.Tab(label="InPaint") as tab2:
                from UI import Inpaint_ui
                Inpaint_ui.show_Inpaint_ui()
        #if ui_config.Tools_Tab:
        if False:            
            with gr.Tab(label="Image Tools") as tab3:
                from UI import ui_image_tools
                ui_image_tools.show_input_image_area()
                ui_image_tools.show_danbooru_area()
                ui_image_tools.show_image_resolution_area()
        #if ui_config.InstructP2P_Tab:
        if False:            
            with gr.Tab(label="Instruct Pix2Pix") as tab4:
                from UI import instructp2p_ui
                instructp2p_ui.show_instructp2p_ui()
        #if ui_config.ControlNet_Tab:
        if False:            
            with gr.Tab(label="ControlNet") as tab5:
                from UI import ControlNet_ui
                ControlNet_ui.show_ControlNet_ui()
        if False:            
            with gr.Tab(label="FaceRestoration") as tab6:
                from UI import ui_face_tools
                ui_face_tools.show_input_image_area()

        with gr.Tab(label="Configuration") as tab7:
            from UI import config_ui_general
            from UI import config_ui_ControlNet
            from UI import config_ui_Vae
            from UI import config_ui_TextEncoder

            config_ui_general.show_general_configuration()
            if ui_config.Advanced_Config:
                    from UI import config_ui_engine as config_ui_engine
                    config_ui_engine.show_providers_configuration()
                    config_ui_ControlNet.show_controlnet_models_configuration()
                    config_ui_TextEncoder.show_textenc_models_configuration()
                    config_ui_Vae.show_vae_models_configuration()
                    #from UI import config_ui_wildcards as wilcards_ui_config
                    #wilcards_ui_config.show_wilcards_configuration()

    return demo
	

css1 = """
#title1 {background-color: #00ACAA;text-transform: uppercase; font-weight: bold;}
.feedback textarea {font-size: 24px !important}
#imagen1 {min-height: 600px}
#imagen2 {height: 250px}
#imagen3 {min-width: 512px,min-height: 512px}
"""

css2 = """
#title1 {background-color: #00ACAA;text-transform: uppercase; font-weight: bold;}
.feedback textarea {font-size: 24px !important}
"""


Running_information= running_config().Running_information
Running_information.update({"cancelled":False})
Running_information.update({"model":""})
Running_information.update({"ControlNetModel":""}) #check if used
Running_information.update({"tab":""})
Running_information.update({"Running":False})
Running_information.update({"Save_Latents":False})
Running_information.update({"Load_Latents":False})
Running_information.update({"Latent_Name":""})
Running_information.update({"Latent_Formula":""})
Running_information.update({"Callback_Steps":2})
Running_information.update({"Vae_Config":["model"]*6})
Running_information.update({"Textenc_Config":["model"]*2})
Running_information.update({"offset":1})
Running_information.update({"Style":False})


Engine_Configuration().load_config_json()
demo =init_ui()
demo.launch(server_port=UI_Configuration().GradioPort)


