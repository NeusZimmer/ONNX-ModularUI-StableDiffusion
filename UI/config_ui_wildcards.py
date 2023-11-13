import gradio as gr
from Engine import General_parameters as params

global debug
debug = True


def show_wilcards_configuration():
    with gr.Blocks(title="ONNX Difussers UI-2") as test2:
        with gr.Accordion(label="Wilcards options",open=False):
            gr.Markdown("Still not implemented: Wildcards not recursive & Not working option to change marker of wildcards")
            with gr.Row():
                Wilcards_Select=gr.Radio(["Yes","No","Yes,not recursive"], label="Wilcards options", info="Activate Yes/No",value="Yes")
                Wilcards_Select.change(fn=Generic_Select_Option,inputs=Wilcards_Select,outputs=None)

def Generic_Select_Option(Radio_Select):
    config=params.UI_Configuration()
    if Radio_Select == "Yes":
        config.wildcards_activated=True
        print(params.UI_Configuration().wildcards_activated)
    else:
        config.wildcards_activated=False
        print(params.UI_Configuration().wildcards_activated)

