import gradio as gr
from Engine.General_parameters import ControlNet_config

def show_controlnet_models_configuration():
    ControlNetConfig=ControlNet_config().config

    if True:
        with gr.Accordion(label="Select ControlNet Models Directories",open=False):
            gr.Markdown("Instead of saving duplicates of each ControlNet model for every model, save one instance of generic ControlNet models into a directory and write down here their path.Include full name and extension")
            with gr.Row():
                with gr.Column(scale=1):
                    canny_active = gr.Checkbox(label="Canny Model Activated?", value=ControlNetConfig["canny_active"], interactive=True)
                with gr.Column(scale=8):
                    canny_path=gr.Textbox(label="Canny model full path",lines=1, value=ControlNetConfig["canny_path"], visible=True, interactive=True)
            with gr.Row():
                with gr.Column(scale=1):
                    depth_active = gr.Checkbox(label="Depth Model Activated?", value=ControlNetConfig["depth_active"], interactive=True)
                with gr.Column(scale=8):
                    depth_path=gr.Textbox(label="Depth Model full path",lines=1, value=ControlNetConfig["depth_path"], visible=True, interactive=True)
            with gr.Row():
                with gr.Column(scale=1):
                    hed_active = gr.Checkbox(label="Hed Model Activated?", value=ControlNetConfig["hed_active"], interactive=True)
                with gr.Column(scale=8):
                    hed_path=gr.Textbox(label="Hed Model full path",lines=1, value=ControlNetConfig["hed_path"], visible=True, interactive=True)
            with gr.Row():
                with gr.Column(scale=1):
                    mlsd_active = gr.Checkbox(label="Mlsd Model Activated?", value=ControlNetConfig["mlsd_active"], interactive=True)
                with gr.Column(scale=8):
                    mlsd_path=gr.Textbox(label="Mlsd Model full path",lines=1, value=ControlNetConfig["mlsd_path"], visible=True, interactive=True)
            with gr.Row():
                with gr.Column(scale=1):
                    normal_active = gr.Checkbox(label="Normal Model Activated?", value=ControlNetConfig["normal_active"], interactive=True)
                with gr.Column(scale=8):
                    normal_path=gr.Textbox(label="Normal Model full path",lines=1, value=ControlNetConfig["normal_path"], visible=True, interactive=True)
            with gr.Row():
                with gr.Column(scale=1):
                    openpose_active = gr.Checkbox(label="Openpose Model Activated?", value=ControlNetConfig["openpose_active"], interactive=True)
                with gr.Column(scale=8):
                    openpose_path=gr.Textbox(label="Openpose full path",lines=1, value=ControlNetConfig["openpose_path"], visible=True, interactive=True)
            with gr.Row():
                with gr.Column(scale=1):
                    seg_active = gr.Checkbox(label="Seg Model Activated?", value=ControlNetConfig["seg_active"], interactive=True)
                with gr.Column(scale=8):
                    seg_path=gr.Textbox(label="Seg Model full path",lines=1, value=ControlNetConfig["seg_path"], visible=True, interactive=True)
            with gr.Row():
                save_btn = gr.Button("Apply & Save ControlNet models config")
                load_btn = gr.Button("Load ControlNet models config")

        all_inputs=[
             canny_active,canny_path,depth_active,depth_path,hed_active,hed_path,mlsd_active,mlsd_path,
             normal_active, normal_path,openpose_active,openpose_path,seg_active,seg_path]

        save_btn.click(fn=save_controlnet_config_ui, inputs=all_inputs, outputs=None)
        load_btn.click(fn=load_controlnet_config_ui, inputs=None , outputs=all_inputs)


def load_controlnet_config_ui():
    config=ControlNet_config()
    config.load_config_from_disk()
    return list(ControlNet_config().config.values())


def save_controlnet_config_ui(*args):
    controlnet_config= {
        "canny_active":args[0],
        "canny_path":args[1],
        "depth_active":args[2],
        "depth_path":args[3],
        "hed_active":args[4],
        "hed_path":args[5],
        "mlsd_active":args[6],
        "mlsd_path":args[7],
        "normal_active":args[8],
        "normal_path":args[9],
        "openpose_active":args[10],
        "openpose_path":args[11],
        "seg_active":args[12],
        "seg_path":args[13],
    }
    ControlNet_config().save_controlnet_config(controlnet_config)



