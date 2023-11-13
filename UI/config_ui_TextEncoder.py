import gradio as gr
from Engine.General_parameters import running_config
from Engine.General_parameters import TextEnc_config

def show_textenc_models_configuration():
    TextencConfig=load_textenc_preferences__ui()
    apply_textenc_config_ui(list(TextencConfig.values()))


    if True:
        with gr.Accordion(label="Text Encoder Models Order & Directories",open=False):
            gr.Markdown("""Saving disk space\n
                        Instead of saving one duplicate for every model as they are usually the same one(two options available only), save one instance of a Text Encoder model\n
                        into a directory and write down their path.\n
                        The system will try to apply your 1st option, if not found it will go for 2nd.""")
            with gr.Row():
                    gr.Markdown("Options for Text Encoder.",elem_id="title1")
            with gr.Row():
                with gr.Column(scale=8):
                    gr.Markdown("Selected model own Text encoder.")
                with gr.Column(scale=2):
                    model1_textenc_order=gr.Slider(1, 5, value=TextencConfig["model1_textenc_order"], step=1, label="Own model Text Encoder search order", interactive=True)
            with gr.Row():
                with gr.Column(scale=8):
                    model2_textenc_path=gr.Textbox(label="Text Encoder clip-slip1",lines=1, value=TextencConfig["model2_textenc_path"], visible=True, interactive=True)
                with gr.Column(scale=2):
                    model2_textenc_order=gr.Slider(1, 5, value=TextencConfig["model2_textenc_order"], step=1, label="This Text Encoder search order", interactive=True)
            with gr.Row():
                with gr.Column(scale=8):
                    model3_textenc_path=gr.Textbox(label="Text Encoder clip-slip2",lines=1, value=TextencConfig["model3_textenc_path"], visible=True, interactive=True)
                with gr.Column(scale=2):
                    model3_textenc_order=gr.Slider(1, 5, value=TextencConfig["model3_textenc_order"], step=1, label="This Text Encoder search order", interactive=True)
            with gr.Row():
                with gr.Column(scale=8):
                    model4_textenc_path=gr.Textbox(label="Text Encoder clip-slip3",lines=1, value=TextencConfig["model4_textenc_path"], visible=True, interactive=True)
                with gr.Column(scale=2):
                    model4_textenc_order=gr.Slider(1, 5, value=TextencConfig["model4_textenc_order"], step=1, label="This Text Encoder search order", interactive=True)
            with gr.Row():
                with gr.Column(scale=8):
                    model5_textenc_path=gr.Textbox(label="Text Encoder clip-slip4",lines=1, value=TextencConfig["model5_textenc_path"], visible=True, interactive=True)
                with gr.Column(scale=2):
                    model5_textenc_order=gr.Slider(1, 5, value=TextencConfig["model5_textenc_order"], step=1, label="This Text Encoder search order", interactive=True)                                        

            save_btn = gr.Button("Apply & Save Text Encoder models config")
            load_btn = gr.Button("Load Text Encoder models config")

        all_inputs=[model1_textenc_order,
                    model2_textenc_order,model2_textenc_path,
                    model3_textenc_order,model3_textenc_path,
                    model4_textenc_order,model4_textenc_path,
                    model5_textenc_order,model5_textenc_path]
        save_btn.click(fn=save_textenc_config_ui, inputs=all_inputs, outputs=None)
        load_btn.click(fn=load_textenc_preferences__ui2, inputs=None , outputs=all_inputs)

def load_textenc_preferences__ui2():
    return list(load_textenc_preferences__ui().values())

def load_textenc_preferences__ui():
    config=TextEnc_config()
    textenc_config=config.load_config_from_disk()
    #Fast parse,hard-coded, as they are -*were- only 2 elements each, for more, do another approach. -recursive funtionv(to do)

    
    if textenc_config[0]=="model": 
        model1_textenc_order=1
        model2_textenc_order=2
        model2_textenc_path=textenc_config[1]
        model3_textenc_order=3
        model3_textenc_path=textenc_config[2]
        model4_textenc_order=4
        model4_textenc_path=textenc_config[3]
        model5_textenc_order=5
        model5_textenc_path=textenc_config[4]
    elif textenc_config[1]=="model":
        model2_textenc_order=1
        model2_textenc_path=textenc_config[0]
        model1_textenc_order=2
        model3_textenc_order=3
        model3_textenc_path=textenc_config[2]
        model4_textenc_order=4
        model4_textenc_path=textenc_config[3]
        model5_textenc_order=5
        model5_textenc_path=textenc_config[4]
    elif textenc_config[2]=="model":
        model2_textenc_order=1
        model2_textenc_path=textenc_config[0]
        model3_textenc_order=2
        model3_textenc_path=textenc_config[1]
        model1_textenc_order=3
        model4_textenc_order=4
        model4_textenc_path=textenc_config[3]
        model5_textenc_order=5
        model5_textenc_path=textenc_config[4]
    elif textenc_config[3]=="model":
        model2_textenc_order=1
        model2_textenc_path=textenc_config[0]
        model3_textenc_order=2
        model3_textenc_path=textenc_config[1]
        model4_textenc_order=3
        model4_textenc_path=textenc_config[2]
        model1_textenc_order=4
        model5_textenc_order=5
        model5_textenc_path=textenc_config[4]        
    elif textenc_config[4]=="model":
        model2_textenc_order=1
        model2_textenc_path=textenc_config[0]
        model3_textenc_order=2
        model3_textenc_path=textenc_config[1]
        model4_textenc_order=3
        model4_textenc_path=textenc_config[2]
        model5_textenc_order=4
        model5_textenc_path=textenc_config[3]              
        model1_textenc_order=5



    all_inputs={
        "model1_textenc_order":model1_textenc_order,
        "model2_textenc_order":model2_textenc_order,
        "model2_textenc_path":model2_textenc_path,
        "model3_textenc_order":model3_textenc_order,
        "model3_textenc_path":model3_textenc_path,
        "model4_textenc_order":model4_textenc_order,
        "model4_textenc_path":model4_textenc_path,
        "model5_textenc_order":model5_textenc_order,
        "model5_textenc_path":model5_textenc_path}

    return dict(all_inputs)


def apply_textenc_config_ui(*args):
    _save_textenc_config_ui(False, *args)

def save_textenc_config_ui(*args):
    _save_textenc_config_ui(True, *args)

def _save_textenc_config_ui(save=True,*args):
    if not save:
        args=args[0] #is tupla, select the list of args.
    model1_textenc_order=int(args[0])
    model2_textenc_order=int(args[1])
    model2_textenc_path=args[2]
    model3_textenc_order=int(args[3])    
    model3_textenc_path=args[4]
    model4_textenc_order=int(args[5])        
    model4_textenc_path=args[6]
    model5_textenc_order=int(args[7])        
    model5_textenc_path=args[8]
        
    
    textenc_config =[None] * 5  #number of paths antes eran 2
    textenc_config[model1_textenc_order-1]="model"
    textenc_config[model2_textenc_order-1]=model2_textenc_path
    textenc_config[model3_textenc_order-1]=model3_textenc_path
    textenc_config[model4_textenc_order-1]=model4_textenc_path
    textenc_config[model5_textenc_order-1]=model5_textenc_path


    Running_information= running_config().Running_information
    Running_information.update({"Textenc_Config":textenc_config})
    if save:
        TextEnc_config().save_TextEnc_config(textenc_config)




