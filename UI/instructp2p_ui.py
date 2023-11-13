import gradio as gr
import os,re,PIL

from Engine.General_parameters import Engine_Configuration
from Engine.General_parameters import running_config
from Engine.General_parameters import UI_Configuration as UI_Configuration
from UI import UI_common_funcs as UI_common
from Engine import pipelines_engines

from PIL import Image, PngImagePlugin

def show_instructp2p_ui():
    ui_config=UI_Configuration()
    model_list = UI_common.get_model_list("ip2p")
    sched_list = get_schedulers_list()
    #sched_list = get_schedulers_list() Modificar a los adecuados
    gr.Markdown("Start typing below and then click **Process** to produce the output.Attention: Image is not automatically saved.")
    with gr.Row(): 
        with gr.Column(scale=13, min_width=650):
            model_drop = gr.Dropdown(model_list, value=(model_list[0] if len(model_list) > 0 else None), label="model folder", interactive=True)
            prompt_t0 = gr.Textbox(value="", lines=2, label="prompt")
            sch_t0 = gr.Radio(sched_list, value=sched_list[0], label="scheduler")
            image_t0 = gr.Image(source="upload",label="input image", type="pil", elem_id="image_p2p")
            iter_t0 = gr.Slider(1, 100, value=1, step=1, label="iteration count", visible=False)
            steps_t0 = gr.Slider(1, 300, value=16, step=1, label="steps")
            guid_t0 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
            height_t0 = gr.Slider(256, 2048, value=512, step=64, label="height")
            width_t0 = gr.Slider(256, 2048, value=512, step=64, label="width")
            eta_t0 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=True)
            seed_t0 = gr.Textbox(value="", max_lines=1, label="seed")
            fmt_t0 = gr.Radio(["png", "jpg"], value="png", label="image format", visible=False)
        with gr.Column(scale=11, min_width=550):
            with gr.Row():
                gen_btn = gr.Button("Process", variant="primary", elem_id="gen_button")
                #cancel_btn = gr.Button("Cancel",info="Cancel at end of current iteration",variant="stop", elem_id="gen_button")
                memory_btn = gr.Button("Release memory", elem_id="mem_button")
            with gr.Row():
                image_out = gr.Gallery(value=None, label="output images")

            with gr.Row():
                status_out = gr.Textbox(value="", label="status")

  
    #cancel_btn.click(fn=UI_common.cancel_iteration,inputs=None,outputs=None)

    list_of_All_Parameters=[model_drop,prompt_t0,sch_t0,image_t0,iter_t0,steps_t0,guid_t0,height_t0,width_t0,eta_t0,seed_t0,fmt_t0]
    gen_btn.click(fn=generate_click, inputs=list_of_All_Parameters, outputs=[image_out,status_out])
    #sch_t0.change(fn=select_scheduler, inputs=sch_t0, outputs= None)  #Atencion cambiar el DDIM ETA si este se activa
    memory_btn.click(fn=UI_common.clean_memory_click, inputs=None, outputs=None)


def get_model_list():
    model_list = []
    try:
        with os.scandir(UI_Configuration().models_dir) as scan_it:
            for entry in scan_it:
                if entry.is_dir():
                    model_list.append(entry.name)
    except:
        model_list.append("Models directory does not exist, configure it")
    return model_list

def get_schedulers_list():
    sched_config = pipelines_engines.SchedulersConfig()
    sched_list =sched_config.available_schedulers
    return sched_list

def select_scheduler(sched_name,model_path):
    return pipelines_engines.SchedulersConfig().scheduler(sched_name,model_path)

def generate_click(model_drop,prompt_t0,sch_t0,image_t0,iter_t0,steps_t0,guid_t0,height_t0,width_t0,eta_t0,seed_t0,fmt_t0):
    from Engine.pipelines_engines import instruct_p2p_pipe

    Running_information= running_config().Running_information
    Running_information.update({"Running":True})
    model_path=ui_config=UI_Configuration().models_dir+"\\"+model_drop

    if (Running_information["model"] != model_drop or Running_information["tab"] != "instruct_p2p"):
        UI_common.clean_memory_click()
        Running_information.update({"model":model_drop})
        Running_information.update({"tab":"instruct_p2p"})

    instruct_p2p_pipe().initialize(model_path,sch_t0)
    input_image = resize_and_crop(image_t0, height_t0, width_t0)

    # generate seed for iteration
    instruct_p2p_pipe().create_seed(seed_t0)
    #images= []
    #information=[]
    #counter=1
    #img_index=get_next_save_index()
    images,info = instruct_p2p_pipe().run_inference(
            prompt=prompt_t0,
            input_image=input_image,
            steps=steps_t0,
            guid=guid_t0,
            eta=eta_t0)
    Running_information.update({"Running":False})
    info=dict(info)
    info['Sched:']=sch_t0
    #information.append(info)
    return images,info
                    

def get_next_save_index():
    output_path=UI_Configuration().output_path
    dir_list = os.listdir(output_path)
    if len(dir_list):
        pattern = re.compile(r"([0-9][0-9][0-9][0-9][0-9][0-9])-([0-9][0-9])\..*")
        match_list = [pattern.match(f) for f in dir_list]
        next_index = max([int(m[1]) if m else -1 for m in match_list]) + 1
    else:
        next_index = 0
    return next_index


def save_image(batch_images,info,next_index):
    output_path=UI_Configuration().output_path

    info_png = f"{info}"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("parameters",info_png)
    prompt=info["prompt"]
    short_prompt = prompt.strip("<>:\"/\\|?*\n\t")
    short_prompt = re.sub(r'[\\/*?:"<>|\n\t]', "", short_prompt)
    short_prompt = short_prompt[:49] if len(short_prompt) > 50 else short_prompt

    os.makedirs(output_path, exist_ok=True)
    """dir_list = os.listdir(output_path)
    if len(dir_list):
        pattern = re.compile(r"([0-9][0-9][0-9][0-9][0-9][0-9])-([0-9][0-9])\..*")
        match_list = [pattern.match(f) for f in dir_list]
        next_index = max([int(m[1]) if m else -1 for m in match_list]) + 1
    else:
        next_index = 0"""
    for image in batch_images:
        image.save(os.path.join(output_path,f"{next_index:06}-00.{short_prompt}.png",),optimize=True,pnginfo=metadata,)
    



    
def resize_and_crop(input_image: PIL.Image.Image, height: int, width: int):
    input_width, input_height = input_image.size

    # nearest neighbor for upscaling
    if (input_width * input_height) < (width * height):
        resample_type = Image.NEAREST
    # lanczos for downscaling
    else:
        resample_type = Image.LANCZOS

    if height / width > input_height / input_width:
        adjust_width = int(input_width * height / input_height)
        input_image = input_image.resize((adjust_width, height),
                                         resample=resample_type)
        left = (adjust_width - width) // 2
        right = left + width
        input_image = input_image.crop((left, 0, right, height))
    else:
        adjust_height = int(input_height * width / input_width)
        input_image = input_image.resize((width, adjust_height),
                                         resample=resample_type)
        top = (adjust_height - height) // 2
        bottom = top + height
        input_image = input_image.crop((0, top, width, bottom))
    return input_image