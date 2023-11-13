import gradio as gr
import onnxruntime as ort
from Engine.General_parameters import Engine_Configuration

global debug
debug = True

def show_providers_configuration():
    with gr.Blocks(title="ONNX Difussers UI-2") as test:
        own_code="('DmlExecutionProvider', {'device_id': 0,})"
        with gr.Accordion(label="Select which provider will load&execute each process/pipeline",open=False):
            gr.Markdown("When using Own Code areas, always make sure to include at least one space within the code, if not loading will fail")
            with gr.Row():
                MAINPipe_Select=gr.Radio(ort.get_available_providers()+["Own_code"], label="MAINPipes provider", info="Where to load the main pipes",value="CPUExecutionProvider")
                MAINPipe_own_code=gr.Textbox(label="Pipeline provider",info="Own code for provider parameter",lines=1, value=own_code, visible=False, interactive=True)
                MAINPipe_Select.change(fn=Generic_Select_OwnCode,inputs=MAINPipe_Select,outputs=MAINPipe_own_code)
            with gr.Row():
                Sched_Select=gr.Radio(ort.get_available_providers()+["Own_code"], label="Scheduler provider", info="Where to load the schedulers",value="CPUExecutionProvider")
                Sched_own_code=gr.Textbox(label="Pipeline provider",info="Own code for provider parameter",lines=1, value=own_code, visible=False)
                Sched_Select.change(fn=Generic_Select_OwnCode,inputs=Sched_Select,outputs=Sched_own_code)
            with gr.Row():
                ControlNet_Select=gr.Radio(ort.get_available_providers()+["Own_code"], label="ControlNet provider", info="Where to load ControlNet",value="CPUExecutionProvider")
                ControlNet_own_code=gr.Textbox(label="Pipeline provider",info="Own code for provider parameter",lines=1, value=own_code, visible=False)
                ControlNet_Select.change(fn=Generic_Select_OwnCode,inputs=ControlNet_Select,outputs=ControlNet_own_code)
            with gr.Row():
                VAEDec_Select=gr.Radio(ort.get_available_providers()+["Own_code"], label="VAEDecoder provider", info="Where to load the VAE",value="CPUExecutionProvider")
                VAEDec_own_code=gr.Textbox(label="Pipeline provider",info="Own code for provider parameter",lines=1, value=own_code, visible=False)
                VAEDec_Select.change(fn=Generic_Select_OwnCode,inputs=VAEDec_Select,outputs=VAEDec_own_code)
            with gr.Row():
                TEXTEnc_Select=gr.Radio(ort.get_available_providers()+["Own_code"], label="TEXTEncoder provider", info="Where to load the Text Encoder",value="CPUExecutionProvider")
                TEXTEnc_own_code=gr.Textbox(label="Pipeline provider",info="Own code for provider parameter",lines=1, value=own_code, visible=False)
                TEXTEnc_Select.change(fn=Generic_Select_OwnCode,inputs=TEXTEnc_Select,outputs=TEXTEnc_own_code)
            with gr.Row():
                DeepDanbooru_Select=gr.Radio(ort.get_available_providers(), label="DeepDanbooru provider", info="Where to load DeepDanbooru queries",value="CPUExecutionProvider")
                gr.Markdown("DeepDanbooru provider. If already loaded for a query, unload it to apply changes")
            with gr.Row():
                apply_btn = gr.Button("Apply providers config", variant="primary")
                apply_btn.click(fn=apply_config, inputs=[MAINPipe_Select, MAINPipe_own_code,Sched_Select,Sched_own_code,ControlNet_Select,ControlNet_own_code, VAEDec_Select,VAEDec_own_code,TEXTEnc_Select,TEXTEnc_own_code,DeepDanbooru_Select] , outputs=None)
                save_btn = gr.Button("Save to config file", elem_id="test_button")
                save_btn.click(fn=save_config_disk, inputs=None, outputs=None)
                load_btn = gr.Button("Load from config file", elem_id="test_button")
                load_btn.click(fn=load_config_disk, inputs=None, outputs=[MAINPipe_Select, MAINPipe_own_code,Sched_Select,Sched_own_code,ControlNet_Select,ControlNet_own_code, VAEDec_Select,VAEDec_own_code,TEXTEnc_Select,TEXTEnc_own_code,DeepDanbooru_Select])


def apply_config(MAINPipe_Select, MAINPipe_own_code,Sched_Select,Sched_own_code,ControlNet_Select,ControlNet_own_code, 
VAEDec_Select,VAEDec_own_code,TEXTEnc_Select,TEXTEnc_own_code,DeepDanbooru_Select):
    Engine_Config=Engine_Configuration()
    Engine_Config.MAINPipe_provider=get_provider_code(MAINPipe_Select,MAINPipe_own_code)
    Engine_Config.Scheduler_provider=get_provider_code(Sched_Select,Sched_own_code)
    Engine_Config.ControlNet_provider=get_provider_code(ControlNet_Select,ControlNet_own_code)
    Engine_Config.VAEDec_provider=get_provider_code(VAEDec_Select,VAEDec_own_code)
    Engine_Config.TEXTEnc_provider=get_provider_code(TEXTEnc_Select,TEXTEnc_own_code)
    Engine_Config.DeepDanBooru_provider=DeepDanbooru_Select
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
    test ="Own_code" if (" " in loaded.MAINPipe_provider) else loaded.MAINPipe_provider
    test2 = loaded.MAINPipe_provider if (" " in loaded.MAINPipe_provider) else ""
    test3 ="Own_code" if (" " in loaded.Scheduler_provider) else loaded.Scheduler_provider
    test4 = loaded.Scheduler_provider if (" " in loaded.Scheduler_provider) else ""
    test5 ="Own_code" if (" " in loaded.ControlNet_provider) else loaded.ControlNet_provider
    test6 = loaded.ControlNet_provider if (" " in loaded.ControlNet_provider) else ""
    test7 ="Own_code" if (" " in loaded.VAEDec_provider) else loaded.VAEDec_provider
    test8 = loaded.VAEDec_provider if (" " in loaded.VAEDec_provider) else ""
    test9 ="Own_code" if (" " in loaded.TEXTEnc_provider) else loaded.TEXTEnc_provider
    test10 = loaded.TEXTEnc_provider if (" " in loaded.TEXTEnc_provider) else ""
    test11 = loaded.DeepDanBooru_provider

    return test,test2,test3,test4,test5,test6,test7,test8,test9,test10,test11


def get_provider_code(Selection,OwnCode):    #Esta funcion es eliminable si no hay que modificar el retorno
    if Selection =="Own_code":
        return OwnCode
    else:
        return str(Selection)

def Generic_Select_OwnCode(Btn_Select):
    if Btn_Select == "Own_code":
        return gr.Textbox.update(visible=True)
    else:
        return gr.Textbox.update(visible=False)

