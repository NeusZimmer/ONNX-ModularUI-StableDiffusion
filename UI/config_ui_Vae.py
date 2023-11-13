import gradio as gr
from Engine.General_parameters import running_config
from Engine.General_parameters import VAE_config

def show_vae_models_configuration():
    VaeConfig=load_vae_preferences__ui()
    apply_vae_config_ui(list(VaeConfig.values()))


    if True:
        with gr.Accordion(label="Vae Models Order & Directories",open=False):
            gr.Markdown("""Saving disk space\n
                        Instead of saving duplicates of each VAE (decoder and encoder) files for every model as they are the same, save one instance of generics VAE model\n
                        into a directory and write down their path.\n
                        The system will try to apply your 1st option, if not found it go for 2nd and then for 3rd.""")
            with gr.Row():
                    gr.Markdown("Options for VAE Decoder.",elem_id="title1")
            with gr.Row():
                with gr.Column(scale=8):
                    gr.Markdown("Selected model own VAE decoder.")
                with gr.Column(scale=2):
                    model1_vaedec_order=gr.Slider(1, 3, value=VaeConfig["model1_vaedec_order"], step=1, label="Own model VAE decoder search order", interactive=True)
            with gr.Row():
                with gr.Column(scale=8):
                    model2_vaedec_path=gr.Textbox(label="VAE Decoder model full path",lines=1, value=VaeConfig["model2_vaedec_path"], visible=True, interactive=True)
                with gr.Column(scale=2):
                    model2_vaedec_order=gr.Slider(1, 3, value=VaeConfig["model2_vaedec_order"], step=1, label="This VAE Decoder search order", interactive=True)
            with gr.Row():
                with gr.Column(scale=8):
                    model3_vaedec_path=gr.Textbox(label="VAE Decoder model full path",lines=1, value=VaeConfig["model3_vaedec_path"], visible=True, interactive=True)
                with gr.Column(scale=2):
                    model3_vaedec_order=gr.Slider(1, 3,value=VaeConfig["model3_vaedec_order"], step=1, label="This VAE decoder search order", interactive=True)
            with gr.Row():
                    gr.Markdown("Options for VAE encoder.",elem_id="title1")
            with gr.Row():
                with gr.Column(scale=8):
                    gr.Markdown("Selected model own VAE encoder.")
                with gr.Column(scale=2):
                    model1_vaeenc_order=gr.Slider(1, 3, value=VaeConfig["model1_vaeenc_order"], step=1, label="Own model VAE Encoder search order", interactive=True)
            with gr.Row():
                with gr.Column(scale=8):
                    model2_vaeenc_path=gr.Textbox(label="VAE Encoder model full path",lines=1, value=VaeConfig["model2_vaeenc_path"], visible=True, interactive=True)
                with gr.Column(scale=2):
                    model2_vaeenc_order=gr.Slider(1, 3, value=VaeConfig["model2_vaeenc_order"], step=1, label="VAE Encoder search order", interactive=True)
            with gr.Row():
                with gr.Column(scale=8):
                    model3_vaeenc_path=gr.Textbox(label="VAE Encoder model full path",lines=1, value=VaeConfig["model3_vaeenc_path"], visible=True, interactive=True)
                with gr.Column(scale=2):
                    model3_vaeenc_order=gr.Slider(1, 3, value=VaeConfig["model3_vaeenc_order"], step=1, label="VAE Encoder search order", interactive=True)


            save_btn = gr.Button("Apply & Save VAE models config")
            load_btn = gr.Button("Load VAE models config")

        all_inputs=[
             model1_vaedec_order,model2_vaedec_order,model2_vaedec_path,model3_vaedec_order,model3_vaedec_path,
             model1_vaeenc_order,model2_vaeenc_order,model2_vaeenc_path,model3_vaeenc_order,model3_vaeenc_path]
        save_btn.click(fn=save_vae_config_ui, inputs=all_inputs, outputs=None)
        load_btn.click(fn=load_vae_preferences__ui2, inputs=None , outputs=all_inputs)

def load_vae_preferences__ui2():
    return list(load_vae_preferences__ui().values())

def load_vae_preferences__ui():
    config=VAE_config()
    vaeconfig=config.load_config_from_disk()
    #Fast parse,hard-coded, as they are only 3 elements each, for more, do another approach. -recursive funtion

    if vaeconfig[0]=="model": 
        model1_vaedec_order=1
        model2_vaedec_order=2
        model2_vaedec_path=vaeconfig[1]
        model3_vaedec_order=3
        model3_vaedec_path=vaeconfig[2]
    elif vaeconfig[1]=="model":
        model2_vaedec_order=1
        model2_vaedec_path=vaeconfig[0]
        model1_vaedec_order=2
        model3_vaedec_order=3
        model3_vaedec_path=vaeconfig[2]
    elif vaeconfig[2]=="model":
        model2_vaedec_order=1
        model2_vaedec_path=vaeconfig[0]
        model3_vaedec_order=2
        model3_vaedec_path=vaeconfig[1]
        model1_vaedec_order=3

    if vaeconfig[3]=="model": 
        model1_vaeenc_order=1
        model2_vaeenc_order=2
        model2_vaeenc_path=vaeconfig[4]
        model3_vaeenc_order=3
        model3_vaeenc_path=vaeconfig[5]
    elif vaeconfig[4]=="model":
        model2_vaeenc_order=1
        model2_vaeenc_path=vaeconfig[3]
        model1_vaeenc_order=2
        model3_vaeenc_order=3
        model3_vaeenc_path=vaeconfig[5]
    elif vaeconfig[5]=="model":
        model2_vaeenc_order=1
        model2_vaeenc_path=vaeconfig[3]
        model3_vaeenc_order=2
        model3_vaeenc_path=vaeconfig[4]
        model1_vaeenc_order=3

    all_inputs={
        "model1_vaedec_order":model1_vaedec_order,
        "model2_vaedec_order":model2_vaedec_order,"model2_vaedec_path":model2_vaedec_path,
        "model3_vaedec_order":model3_vaedec_order,"model3_vaedec_path":model3_vaedec_path,
        "model1_vaeenc_order":model1_vaeenc_order,
        "model2_vaeenc_order":model2_vaeenc_order,"model2_vaeenc_path":model2_vaeenc_path,
        "model3_vaeenc_order":model3_vaeenc_order,"model3_vaeenc_path":model3_vaeenc_path}

    return dict(all_inputs)


def apply_vae_config_ui(*args):
    _save_vae_config_ui(False, *args)

def save_vae_config_ui(*args):
    _save_vae_config_ui(True, *args)

def _save_vae_config_ui(save=True,*args):
    if not save:
        args=args[0] #is tupla, select the list of args.
    model1_vaedec_order=int(args[0])
    model2_vaedec_order=int(args[1])
    model2_vaedec_path=args[2]
    model3_vaedec_order=int(args[3])
    model3_vaedec_path=args[4]

    model1_vaeenc_order=int(args[5])
    model2_vaeenc_order=int(args[6])
    model2_vaeenc_path=args[7]
    model3_vaeenc_order=int(args[8])
    model3_vaeenc_path=args[9]



    vae_config =[None] * 6
    vae_config[model1_vaedec_order-1]="model"
    vae_config[model2_vaedec_order-1]=model2_vaedec_path
    vae_config[model3_vaedec_order-1]=model3_vaedec_path
    vae_config[3+model1_vaeenc_order-1]="model"
    vae_config[3+model2_vaeenc_order-1]=model2_vaeenc_path
    vae_config[3+model3_vaeenc_order-1]=model3_vaeenc_path

    Running_information= running_config().Running_information
    Running_information.update({"Vae_Config":vae_config})
    if save:
        VAE_config().save_VAE_config(vae_config)




