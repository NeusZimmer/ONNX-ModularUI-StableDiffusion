import gradio as gr
import os,gc,re,PIL

from Engine.General_parameters import Engine_Configuration
from Engine.General_parameters import running_config
from Engine.General_parameters import UI_Configuration
from Engine.General_parameters import ControlNet_config
from UI import UI_common_funcs as UI_common
from Engine import SchedulersConfig
#from Engine import pipelines_engines
from Engine import ControlNet_pipe

from PIL import Image, PngImagePlugin


def show_ControlNet_ui():
    ui_config=UI_Configuration()
    model_list = UI_common.get_model_list("controlnet")    
    sched_list = get_schedulers_list()
    controlnet_models= dict(ControlNet_config().available_controlnet_models())
    gr.Markdown("Start typing below and then click **Process** to produce the output.")
    with gr.Row(): 
        with gr.Column(scale=13, min_width=650):
            model_drop = gr.Dropdown(model_list, value=(model_list[0] if len(model_list) > 0 else None), label="model folder", interactive=True)
            with gr.Row():
                ControlNET_drop = gr.Dropdown(controlnet_models.keys(),value=next(iter(controlnet_models)), label="ControNET Type", interactive=True)
                reload_controlnet_btn = gr.Button("Reload ControlNet model list")

            prompt_t0 = gr.Textbox(value="", lines=2, label="prompt")
            neg_prompt_t0 = gr.Textbox(value="", lines=2, label="negative prompt")
            sch_t0 = gr.Radio(sched_list, value=sched_list[0], label="scheduler", interactive=True)

            image_t0 = gr.Image(source="upload",label="input image", type="pil", elem_id="image_inpaint", visible=True)
            #pose_image_t0 = gr.Image(source="upload",label="ControlNET Image",type="pil")

            with gr.Row():
                iter_t0 = gr.Slider(1, 100, value=1, step=1, label="iteration count")
                batch_t0 = gr.Slider(1, 4, value=1, step=1, label="batch size",visible=False)
            steps_t0 = gr.Slider(1, 300, value=16, step=1, label="steps")
            guid_t0 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
            controlnet_conditioning_scale_t0= gr.Slider(0, 1, value=0.5, step=0.1, label="controlnet_conditioning_scale")
            with gr.Row():
                height_t0 = gr.Slider(256, 2048, value=512, step=64, label="height")
                width_t0 = gr.Slider(256, 2048, value=512, step=64, label="width")
            eta_t0 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=False)
            seed_t0 = gr.Textbox(value="", max_lines=1, label="seed")
            #fmt_t0 = gr.Radio(["png", "jpg"], value="png", label="image format")
        with gr.Column(scale=11, min_width=550):
            with gr.Row():
                gen_btn = gr.Button("Process Full from Image", variant="primary", elem_id="gen_button")
                #gen_btn2 = gr.Button("Process to get pose image", variant="primary", elem_id="gen_button")
                #gen_btn3 = gr.Button("Process from pose image", variant="primary", elem_id="gen_button")
            with gr.Row():
                clear_btn = gr.Button("Cancel",info="Cancel at end of current iteration",variant="stop", elem_id="gen_button")
                memory_btn = gr.Button("Release memory", elem_id="mem_button")
            with gr.Row():
                image_out = gr.Gallery(value=None, label="output images")
            with gr.Row():
                status_out = gr.Textbox(value="", label="status")

    list_of_All_Parameters=[model_drop,prompt_t0,neg_prompt_t0,sch_t0,image_t0,iter_t0,batch_t0,steps_t0,guid_t0,height_t0,width_t0,eta_t0,seed_t0,ControlNET_drop,controlnet_conditioning_scale_t0]
    gen_btn.click(fn=generate_click, inputs=list_of_All_Parameters, outputs=[image_out,status_out])
    #gen_btn1.click(fn=generate_click, inputs=list_of_All_Parameters, outputs=[pose_image_t0,status_out])
    #gen_btn2.click(fn=generate_click, inputs=list_of_All_Parameters, outputs=[image_out,status_out])
    #sch_t0.change(fn=select_scheduler, inputs=sch_t0, outputs= None)  #Atencion cambiar el DDIM ETA si este se activa
    memory_btn.click(fn=UI_common.clean_memory_click, inputs=None, outputs=None)
    clear_btn.click(fn=UI_common.cancel_iteration,inputs=None,outputs=None)
    reload_controlnet_btn.click(fn=ReloadControlNet, inputs=None , outputs=ControlNET_drop)


def ReloadControlNet():
    ControlNet_config().load_config_from_disk()
    controlnet_models= dict(ControlNet_config().available_controlnet_models())
    controlnet_models_keys= list(controlnet_models.keys())
    return gr.Dropdown.update(choices=controlnet_models_keys, value=controlnet_models_keys[0])


def get_schedulers_list():
    sched_config = SchedulersConfig()
    sched_list =sched_config.schedulers_controlnet_list()
    return sched_list

def select_scheduler(sched_name,model_path):
    return SchedulersConfig().scheduler(sched_name,model_path)


def generate_click(model_drop,prompt_t0,neg_prompt_t0,sch_t0,image_t0,iter_t0,batch_t0,steps_t0,guid_t0,height_t0,width_t0,eta_t0,seed_t0,ControlNET_drop,controlnet_conditioning_scale):

    Running_information= running_config().Running_information
    Running_information.update({"Running":True})

    input_image = resize_and_crop(image_t0, height_t0, width_t0)
    #pose_image_t0 = resize_and_crop(pose_image_t0, height_t0, width_t0)

    model_path=UI_Configuration().models_dir+"\\"+model_drop

    if (Running_information["model"] != model_drop or Running_information["tab"] != "controlnet"):
        UI_common.clean_memory_click()
        Running_information.update({"model":model_drop})
        Running_information.update({"tab":"controlnet"})
    ControlNet_pipe().initialize(model_path,sch_t0,ControlNET_drop)


    ControlNet_pipe().create_seeds(seed_t0,iter_t0,False)
    images= []
    information=[]
    counter=1
    img_index=get_next_save_index()

    for seed in ControlNet_pipe().seeds:
        if running_config().Running_information["cancelled"]:
            running_config().Running_information.update({"cancelled":False})
            break

        print(f"Iteration:{counter}/{iter_t0}")
        counter+=1
        batch_images,info = ControlNet_pipe().run_inference(
            prompt_t0,
            neg_prompt=neg_prompt_t0,
            input_image=input_image,
            #pose_image=pose_image_t0,
            height=height_t0,
            width=width_t0,
            steps=steps_t0,
            guid=guid_t0,
            eta=eta_t0,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            seed=seed)

        images.extend([batch_images])
        info=dict(info)
        info['Sched:']=sch_t0
        info['CnetModel:']=ControlNET_drop
        information.append(info)
        save_image([batch_images],info,img_index)
        img_index+=1
    Running_information.update({"Running":False})
    return images,information


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