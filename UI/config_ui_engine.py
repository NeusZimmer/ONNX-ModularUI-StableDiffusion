import gradio as gr
import onnxruntime as ort
from Engine.General_parameters import Engine_Configuration

global debug
debug = False

def show_providers_configuration():
    with gr.Blocks(title="ONNX Difussers UI-2") as test:
        own_code="{'device_id': 1}"
        Engine_Config=Engine_Configuration()
        with gr.Accordion(label="Select which provider will load&execute each process/pipeline",open=False):
            gr.Markdown("Device provider options(dict), i.e. a python dict {'device_id': 1} will select the 2nd device provider, for 2 GPUs available")
            with gr.Row():
                MAINPipe_Select=gr.Radio(ort.get_available_providers(), label="MAINPipes provider", info="Where to load the main pipes",value=Engine_Config.MAINPipe_provider['provider'])
                MAINPipe_own_code=gr.Textbox(label="Pipeline provider",info="Device provider options(dict)",lines=1, value=Engine_Config.MAINPipe_provider['provider_options'], visible=True, interactive=True)
                #MAINPipe_Select.change(fn=Generic_Select_OwnCode,inputs=MAINPipe_Select,outputs=MAINPipe_own_code)
            with gr.Row():
                Sched_Select=gr.Radio(ort.get_available_providers(), label="Scheduler provider", info="Where to load the schedulers",value=Engine_Config.Scheduler_provider['provider'])
                Sched_own_code=gr.Textbox(label="Pipeline provider",info="Device provider options(dict)",lines=1, value=Engine_Config.Scheduler_provider['provider_options'], visible=True)
                #Sched_Select.change(fn=Generic_Select_OwnCode,inputs=Sched_Select,outputs=Sched_own_code)
            with gr.Row():
                ControlNet_Select=gr.Radio(ort.get_available_providers(), label="ControlNet provider", info="Where to load ControlNet",value=Engine_Config.ControlNet_provider['provider'])
                ControlNet_own_code=gr.Textbox(label="Pipeline provider",info="Device provider options(dict)",lines=1, value=Engine_Config.ControlNet_provider['provider_options'], visible=True)
                #ControlNet_Select.change(fn=Generic_Select_OwnCode,inputs=ControlNet_Select,outputs=ControlNet_own_code)
            with gr.Row():
                VAEDec_Select=gr.Radio(ort.get_available_providers(), label="VAEDecoder provider", info="Where to load the VAE Decoder",value=Engine_Config.VAEDec_provider['provider'])
                VAEDec_own_code=gr.Textbox(label="Pipeline provider",info="Device provider options(dict)",lines=1, value=Engine_Config.VAEDec_provider['provider_options'], visible=True)
                #VAEDec_Select.change(fn=Generic_Select_OwnCode,inputs=VAEDec_Select,outputs=VAEDec_own_code)
            with gr.Row():
                VAEEnc_Select=gr.Radio(ort.get_available_providers(), label="VAEEncoder provider", info="Where to load the VAE Encoder",value=Engine_Config.VAEEnc_provider['provider'])
                VAEEnc_own_code=gr.Textbox(label="Pipeline provider",info="Device provider options(dict)",lines=1, value=Engine_Config.VAEEnc_provider['provider_options'], visible=True)
                #VAEDec_Select.change(fn=Generic_Select_OwnCode,inputs=VAEDec_Select,outputs=VAEDec_own_code)
            with gr.Row():
                TEXTEnc_Select=gr.Radio(ort.get_available_providers(), label="TEXTEncoder provider", info="Where to load the Text Encoder",value=Engine_Config.TEXTEnc_provider['provider'])
                TEXTEnc_own_code=gr.Textbox(label="Pipeline provider",info="Device provider options(dict)",lines=1, value=Engine_Config.TEXTEnc_provider['provider_options'], visible=True)
                #TEXTEnc_Select.change(fn=Generic_Select_OwnCode,inputs=TEXTEnc_Select,outputs=TEXTEnc_own_code)
            with gr.Row():
                DeepDanbooru_Select=gr.Radio(ort.get_available_providers(), label="DeepDanbooru provider", info="Where to load DeepDanbooru queries",value=Engine_Config.DeepDanBooru_provider)
                gr.Markdown("DeepDanbooru provider. If already loaded for a query, unload it to apply changes")
            with gr.Row():
                apply_btn = gr.Button("Apply providers config", variant="primary")
                apply_btn.click(fn=apply_config, inputs=[MAINPipe_Select, MAINPipe_own_code,Sched_Select,Sched_own_code,ControlNet_Select,ControlNet_own_code, VAEDec_Select,VAEDec_own_code,TEXTEnc_Select,TEXTEnc_own_code,DeepDanbooru_Select,VAEEnc_Select,VAEEnc_own_code] , outputs=None)
                save_btn = gr.Button("Save to config file", elem_id="test_button")
                save_btn.click(fn=save_config_disk, inputs=None, outputs=None)
                load_btn = gr.Button("Load from config file", elem_id="test_button")
                load_btn.click(fn=load_config_disk, inputs=None, outputs=[MAINPipe_Select, MAINPipe_own_code,Sched_Select,Sched_own_code,ControlNet_Select,ControlNet_own_code, VAEDec_Select,VAEDec_own_code,TEXTEnc_Select,TEXTEnc_own_code,DeepDanbooru_Select, VAEEnc_Select,VAEEnc_own_code])


def apply_config(MAINPipe_Select, MAINPipe_own_code,Sched_Select,Sched_own_code,ControlNet_Select,ControlNet_own_code, 
VAEDec_Select,VAEDec_own_code,TEXTEnc_Select,TEXTEnc_own_code,DeepDanbooru_Select,VAEEnc_Select,VAEEnc_own_code):
    Engine_Config=Engine_Configuration()
    Engine_Config.SetProvider('MAINPipe_provider',get_provider_code('MAINPipe',MAINPipe_Select,MAINPipe_own_code))
    Engine_Config.SetProvider('Scheduler_provider',get_provider_code('Schedulers',Sched_Select,Sched_own_code))
    Engine_Config.SetProvider('ControlNet_provider',get_provider_code('ControlNet',ControlNet_Select,ControlNet_own_code))
    Engine_Config.SetProvider('VAEDec_provider',get_provider_code('VaeDecoder',VAEDec_Select,VAEDec_own_code))
    Engine_Config.SetProvider('VAEEnc_provider',get_provider_code('VaeEncoder',VAEEnc_Select,VAEEnc_own_code))  #Asegurarnos que existe antes de activar la linea
    Engine_Config.SetProvider('TEXTEnc_provider',get_provider_code('TextEncoder',TEXTEnc_Select,TEXTEnc_own_code))
    Engine_Config.DeepDanBooru_provider=DeepDanbooru_Select
    global debug
    if debug:
        print("Applied a new running config:")
        print_current_singleton_data(Engine_Configuration())

def print_current_singleton_data(Engine_Config):
    print(Engine_Config.MAINPipe_provider)
    print(Engine_Config.Scheduler_provider)
    print(Engine_Config.ControlNet_provider)
    print(Engine_Config.VAEDec_provider)
    print(Engine_Config.TEXTEnc_provider)
    print(Engine_Config.DeepDanBooru_provider)

def save_config_disk():
    global debug
    if debug:
        print("Saving to disk the Running Singleton content(Maybe not the UI content):")
        print_current_singleton_data(Engine_Configuration())
    Engine_Configuration().save_config_json()

def load_config_disk():
    global debug
    if debug:
        print("Retrieving from disk the Singleton content:")
        print_current_singleton_data(Engine_Configuration())
    Engine_Configuration().load_config_json()
    #Update gr.radio values for the UI to be able to update it.
    loaded=Engine_Configuration()
    test =  loaded.MAINPipe_provider['provider']
    test2 = loaded.MAINPipe_provider['provider_options']
    test3 = loaded.Scheduler_provider['provider']
    test4 = loaded.Scheduler_provider['provider_options']
    test5 = loaded.ControlNet_provider['provider']
    test6 = loaded.ControlNet_provider['provider_options']
    test7 = loaded.VAEDec_provider['provider']
    test8 = loaded.VAEDec_provider['provider_options']
    test9 = loaded.TEXTEnc_provider['provider']
    test10 = loaded.TEXTEnc_provider['provider_options']
    test11 = loaded.DeepDanBooru_provider
    test12 = loaded.VAEEnc_provider['provider']
    test13 = loaded.VAEEnc_provider['provider_options']
    return test,test2,test3,test4,test5,test6,test7,test8,test9,test10,test11,test12,test13


def get_provider_code_old(Selection,OwnCode):    #Esta funcion es eliminable si no hay que modificar el retorno
    if Selection =="Own_code":
        return OwnCode
    else:
        return str(Selection)


def get_provider_code(pipename,Selection,OwnCode): 
    try:
        provider_options=eval(OwnCode)
    except:
        provider_options=""

    if type(provider_options)==dict:
        OwnCode=provider_options
    else:
        OwnCode=None
    
    #return {pipename+'_provider':Selection, pipename+'_provider_options':provider_options}
    return {'provider':Selection, 'provider_options':provider_options}

def Generic_Select_OwnCode(Btn_Select): #No utilizada, para version antigua que permitia otras opciones
    if Btn_Select == "Own_code":
        return gr.Textbox.update(visible=True)
    else:
        return gr.Textbox.update(visible=False)

