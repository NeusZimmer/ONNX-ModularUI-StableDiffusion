import gradio as gr
import os,gc,re
from Engine.General_parameters import Engine_Configuration as Engine_config
from Engine.General_parameters import running_config
from Engine.General_parameters import UI_Configuration
from Engine import pipelines_engines
from UI import UI_common_funcs as UI_common
from Engine import engine_common_funcs as Engine_common
from PIL import Image, PngImagePlugin
from modules.modules import preProcess_modules
from UI import UI_placement_areas

global next_prompt
global processed_images
processed_images=[]
next_prompt=None
global number_of_passes

global list_modules
list_modules=[]
list_modules=preProcess_modules().check_available_modules("hires")


def show_HiRes_txt2img_ui():
    global list_modules
    model_list = UI_common.get_model_list("txt2img")
    sched_list = get_schedulers_list()
    ui_config=UI_Configuration()
    gr.Markdown("Start typing below and then click **Generate** to see the output.")
    with gr.Row(): 
        with gr.Column(scale=13, min_width=650):
            model_drop = gr.Dropdown(model_list, value=(model_list[0] if len(model_list) > 0 else None), label="HiRes model", interactive=True)
            model_list.append("None")
            low_model_drop = gr.Dropdown(model_list, value='None', label="LowRes model (None=Same as HiRes)", interactive=True)
            with gr.Accordion("Additional options",open=False):
                #reload_vae_btn = gr.Button("VAE Decoder:Apply Changes & Reload")
                #reload_textenc_btn = gr.Button("Text Encoder Reload")           
                reload_lowres_btn = gr.Button("Reload Low-res model")
                reload_hires_btn = gr.Button("Reload HiRes Model")           
                copy_hires_to_lowres_btn = gr.Button("Use HiRes model as LowRes")                
                test_vae_split_btn = gr.Button("Test VAE Split",visible=False)
                #reload_model_btn = gr.Button("Model:Apply new model & Fast Reload Pipe")

            sch_t0 = gr.Radio(sched_list, value=sched_list[0], label="scheduler")
            prompt_t0 = gr.Textbox(value="", lines=2, label="prompt")
            neg_prompt_t0 = gr.Textbox(value="", lines=2, label="negative prompt")

            
            with gr.Accordion(label="Process IMG To latent",open=False):
                image_in = gr.Image(label="input image", type="pil", elem_id="image_init")
                convert_to_latent_btn = gr.Button("Conver IMG to latent, size of Hi-Res output")
            with gr.Accordion(label="Latents experimentals",open=False):
                #multiplier = gr.Slider(0, 1, value=0.18215, step=0.05, label="Multiplier, blurry the ingested latent, 1 to do not modify", interactive=True)
                #offset_t0 = gr.Slider(0, 100, value=1, step=1, label="Offset Steps for the scheduler", interactive=True)
                #latents_experimental1 = gr.Checkbox(label="Save generated latents ", value=False, interactive=True)
                latents_experimental2 = gr.Checkbox(label="Load latent from a generation", value=False, interactive=True)
                name_of_latent = gr.Textbox(value="1:LastGenerated_Latent.npy", lines=1, label="Names of Numpy File Latents (1:x.npy,2:y.pt,3:noise-(width)x(height))")
                latent_formula = gr.Textbox(value="1", lines=1, label="Formula for the sumatory of latents")


            with gr.Row():
                gr.Markdown("Common parameters")
            with gr.Row():
                strength_t0 = gr.Slider(0, 1, value=0.8, step=0.05, label="Strength, to apply to the latent", interactive=True)
            with gr.Row():
                iter_t0 = gr.Slider(1, 100, value=1, step=1, label="iteration count")
                guid_t0 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
                batch_t0 = gr.Slider(1, 4, value=1, step=1, label="batch size", interactive=False,visible=False)
            with gr.Row():                
                eta_t0 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=True)
                seed_t0 = gr.Textbox(value="", max_lines=1, label="seed")
                fmt_t0 = gr.Radio(["png", "jpg"], value="png", label="image format", interactive=False,visible=False)
            
            with gr.Row():
                gr.Markdown("HiRes parameters")
            with gr.Row():                         
                height_t1 = gr.Slider(64, 2048, value=512, step=64, label="HiRes height")
                width_t1 = gr.Slider(64, 2048, value=512, step=64, label="HiRes width")
            with gr.Row():                
                steps_t1 = gr.Slider(1, 100, value=16, step=1, label="HiRes steps")
                hires_passes_t1 = gr.Slider(1, 10, value=2, step=1, label="HiRes passes")
                hires_pass_variation = gr.Slider(-1, 1, value=0, step=0.1, label="Variation Strengh between between passes")

            with gr.Row():
                gr.Markdown("First pass parameters")
            with gr.Row():                
                height_t0 = gr.Slider(64, 2048, value=512, step=64, label="height")
                width_t0 = gr.Slider(64, 2048, value=512, step=64, label="width")
                steps_t0 = gr.Slider(1, 3000, value=16, step=1, label="steps")                

            with gr.Row():
                gr.Markdown("Other parameters")
                save_textfile=gr.Checkbox(label="Save prompt into a txt file")
                save_low_res=gr.Checkbox(label="Save generated Low-Res Img")

        with gr.Column(scale=11, min_width=550):
            with gr.Row():
                gen_btn = gr.Button("Generate", variant="primary", elem_id="gen_button")
                #clear_btn = gr.Button("Cancel",info="Cancel at end of current iteration",variant="stop", elem_id="gen_button")
                clear_btn = gr.Button("Cancel",variant="stop", elem_id="gen_button")
                memory_btn = gr.Button("Release memory", elem_id="mem_button")

            """if ui_config.wildcards_activated:
                from UI import styles_ui
                styles_ui.show_styles_ui()
                with gr.Accordion(label="Live Prompt & Wildcards for multiple iterations",open=False):
                    with gr.Row():
                        next_wildcard = gr.Textbox(value="",lines=4, label="Next Prompt", interactive=True)
                        discard = gr.Textbox(value="", label="Discard", visible=False, interactive=False)
                    with gr.Row():
                        wildcard_show_btn = gr.Button("Show next prompt", elem_id="wildcard_button")
                        wildcard_gen_btn = gr.Button("Regenerate next prompt", variant="primary", elem_id="wildcard_button")
                        wildcard_apply_btn = gr.Button("Use edited prompt", elem_id="wildcard_button")"""
            with gr.Accordion(label="Prompt Processing tools",open=False):
                with gr.Accordion(label="Live Prompt for multiple iterations & Prompt pre-generation",open=False):
                    with gr.Row():
                        next_wildcard = gr.Textbox(value="",lines=4, label="Next Prompt", interactive=True)
                        discard = gr.Textbox(value="", label="Discard", visible=False, interactive=False)
                    with gr.Row():
                        wildcard_show_btn = gr.Button("Show next prompt", elem_id="wildcard_button")
                        wildcard_gen_btn = gr.Button("Regenerate next prompt", variant="primary", elem_id="wildcard_button")
                        wildcard_apply_btn = gr.Button("Use edited prompt", elem_id="wildcard_button")
                #Show modules areas
                UI_placement_areas.show_prompt_preprocess_area(list_modules)

            with gr.Row():
                image_out = gr.Gallery(value=None, label="output images")
            with gr.Accordion(label="Low Res output images",open=False):
                with gr.Row():
                    low_res_image_out = gr.Gallery(value=None, label="Low res output images")
            with gr.Row():
                status_out = gr.Textbox(value="", label="status")
            with gr.Row():
                Selected_image_status= gr.Textbox(value="", label="status",visible=True)
                Selected_image_index= gr.Number(show_label=False, visible=False)

    with gr.Row():
        UI_placement_areas.show_footer_area(list_modules)
  
    image_out.select(fn=get_select_index, inputs=[image_out,status_out], outputs=[Selected_image_index,Selected_image_status])
    clear_btn.click(fn=UI_common.cancel_iteration,inputs=None,outputs=None)
    #clear_btn.click(fn=cancel,inputs=None,outputs=None)    

    #reload_vae_btn.click(fn=change_vae,inputs=model_drop,outputs=None)
    #reload_textenc_btn.click(fn=change_textenc,inputs=model_drop,outputs=None)


    list_of_All_Parameters=[model_drop,low_model_drop,prompt_t0,neg_prompt_t0,sch_t0,iter_t0,batch_t0,steps_t0,steps_t1,guid_t0,height_t0,width_t0,height_t1,width_t1,eta_t0,seed_t0,fmt_t0,strength_t0,hires_passes_t1,save_textfile, save_low_res,latent_formula,name_of_latent,latents_experimental2,hires_pass_variation]    

    memory_btn.click(fn=UI_common.clean_memory_click, inputs=None, outputs=None)    
    reload_hires_btn.click(fn=reload_hires,inputs=model_drop,outputs=None)
    reload_lowres_btn.click(fn=reload_lowres,inputs=low_model_drop,outputs=None)
    copy_hires_to_lowres_btn.click(fn=copy_hires,inputs=None,outputs=None)
    test_vae_split_btn.click(fn=vae_output_to_numpy,inputs=None,outputs=image_out)    

    gen_btn.click(fn=generate_click,inputs=list_of_All_Parameters,outputs=[image_out,status_out,low_res_image_out])
    convert_to_latent_btn.click(fn=convert_click,inputs=[image_in,height_t1,width_t1],outputs=None)

    
    wildcard_gen_btn.click(fn=gen_next_prompt, inputs=prompt_t0, outputs=[discard,next_wildcard])
    wildcard_show_btn.click(fn=show_next_prompt, inputs=None, outputs=next_wildcard)
    wildcard_apply_btn.click(fn=apply_prompt, inputs=next_wildcard, outputs=None)


def cancel():
    import onnxruntime as ort
    from Engine.Pipelines.txt2img_hires import txt2img_hires_pipe

    Inference_Session = txt2img_hires_pipe().hires_pipe.unet.model
    sess_options = Inference_Session.get_session_options()


    print(f"test1:{sess_options.enable_cpu_mem_arena}")
    print(f"test2:{Inference_Session}")

    Runtime_options = ort.RunOptions()
    #print(f"test2:{Runtime_options}")  
    Runtime_options.terminate=True


    Inference_Session.RunOptions=Runtime_options

def vae_output_to_numpy():
    import numpy as np
    from Engine import txt2img_hires_pipe
    pipe=txt2img_hires_pipe()
    
    latents=np.load(f"./latents/LastGenerated_Latent.npy")
    latents = 1 / 0.18215 * latents
    print("Shape")
    print(latents.shape)

    """image = np.concatenate(
        [pipe.hires_pipe.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
    )"""
    latents1=latents[:,:,:,0:32]
    latents2=latents[:,:,32:,0:32]
    latents3=latents[:,:,:,32:]
    latents4=latents[:,:,32:,32:]

    image1=pipe.hires_pipe.vae_decoder(latent_sample=latents1)[0]
    #image2=pipe.hires_pipe.vae_decoder(latent_sample=latents2)[0]
    image3=pipe.hires_pipe.vae_decoder(latent_sample=latents3)[0]
    #image4=pipe.hires_pipe.vae_decoder(latent_sample=latents4)[0]        


    image1 = numpy_to_pil(normalize(image1))
    #image2 = numpy_to_pil(normalize(image2))
    image3 = numpy_to_pil(normalize(image3))
    #image4 = numpy_to_pil(normalize(image4))
    images=image1+image3#+image2+image4
    print(type(images))

    return images

def normalize(image2):
    import numpy as np
    image2 = np.clip(image2 / 2 + 0.5, 0, 1)
    return image2.transpose((0, 2, 3, 1))

def numpy_to_pil(images):
    """
    Converts a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def copy_hires():
    from Engine import txt2img_hires_pipe
    pipe=txt2img_hires_pipe()
    #pipe.hires_pipe.unet_session=pipe.hires_pipe.low_unet_session
    pipe.hires_pipe.low_unet_session=pipe.hires_pipe.unet_session
    print("Copied current Hires model onto lowres model")

def reload_lowres(low_model_drop):
    from Engine import txt2img_hires_pipe
    pipe=txt2img_hires_pipe()
    model_path=UI_Configuration().models_dir+"\\"+low_model_drop
    
    pipe.reload_partial_model(model_path,"low")

def reload_hires(model_drop):
    from Engine import txt2img_hires_pipe
    pipe=txt2img_hires_pipe()
    model_path=UI_Configuration().models_dir+"\\"+model_drop
    pipe.reload_partial_model(model_path,"high")

def change_textenc(model_drop):
    print("Pending to be adapted")
    """from Engine.Pipelines.txt2img_hires import txt2img_hires_pipe
    from Engine import Vae_and_Text_Encoders
    pipe=txt2img_hires_pipe().hires_pipe
    textenc=Vae_and_Text_Encoders()
    pipe.text_encoder=textenc.load_textencoder(f"{UI_Configuration().models_dir}\{model_drop}")    """
    return

def change_vae(model_drop):
    print("Pending to be adapted")
    """from Engine.Pipelines.txt2img_hires import txt2img_hires_pipe
    from Engine import Vae_and_Text_Encoders
    pipe=txt2img_hires_pipe().hires_pipe
    vae=Vae_and_Text_Encoders()
    pipe.vae_decoder=vae.load_vaedecoder(f"{UI_Configuration().models_dir}\{model_drop}")
    pipe.vae_encoder=vae.load_vaeencoder(f"{UI_Configuration().models_dir}\{model_drop}")"""
    return


def get_select_index(image_out,status_out, evt:gr.SelectData):
    status_out=eval(status_out)
    global number_of_passes
    number_of_passes
    resto=evt.index % number_of_passes
    index= (evt.index-resto)/number_of_passes

    return index,status_out[int(index)]
    #return evt.index,status_out[int(index)]

def gallery_view(images,dict_statuses):
    return images[0]


def get_schedulers_list():
    sched_config = pipelines_engines.SchedulersConfig()
    sched_list =sched_config.available_schedulers
    return sched_list

def select_scheduler(sched_name,model_path):
    return pipelines_engines.SchedulersConfig().scheduler(sched_name,model_path)

def gen_next_prompt(prompt_t0,initial=False):
    global next_prompt
    Running_information= running_config().Running_information    
    style=Running_information["Style"]

    style_pre =""
    style_post=""
    if style:
        styles=style.split("|")
        style_pre =styles[0]
        style_post=styles[1]

    if (initial):
        next_prompt=None
        prompt=prompt_t0
    else:
        if (next_prompt != None):
            prompt=next_prompt
        else:
            prompt=prompt_process(prompt_t0)

        next_prompt=prompt_process(prompt_t0)
    prompt = style_pre+" " +prompt+" " +style_post
    #print(f"Prompt:{prompt}")
    return prompt,next_prompt

def apply_prompt(prompt):
    global next_prompt
    next_prompt=prompt

def prompt_process_old(prompt):
    from Scripts import wildcards
    wildcard=wildcards.WildcardsScript()
    new_prompt,discarded=wildcard.process(prompt)
    return new_prompt

def prompt_process(prompt):
    #"prompt_process"
    global list_modules
    for module in list_modules:
        if module['func_processing']=="prompt_process": #Area of modules for processing prompts
            prompt=module['call'](prompt) #modules[2]= show, #modules[3] =process

    return prompt


def show_next_prompt():
    global next_prompt
    return next_prompt


def generate_click(
    model_drop,low_model_drop,prompt_t0,neg_prompt_t0,sch_t0,
    iter_t0,batch_t0,steps_t0,steps_t1,guid_t0,height_t0,
    width_t0,height_t1,width_t1,eta_t0,seed_t0,fmt_t0,strength,
    hires_passes_t1,save_textfile, save_low_res,
    latent_formula,name_of_latent,latents_experimental2,hires_pass_variation):

    from Engine.General_parameters import running_config

    if latents_experimental2:
        running_config().Running_information.update({"Load_Latents":True})
        running_config().Running_information.update({"Latent_Name":name_of_latent})
        running_config().Running_information.update({"Latent_Formula":latent_formula})
    else:
        running_config().Running_information.update({"Load_Latents":False})

    #from Engine.Pipelines.txt2img_hires import txt2img_hires_pipe
    from Engine import txt2img_hires_pipe

    global number_of_passes
    number_of_passes=hires_passes_t1

    Running_information= running_config().Running_information
    Running_information.update({"Running":True})
    model_path=UI_Configuration().models_dir+"\\"+model_drop

    if low_model_drop != 'None': low_model_path=UI_Configuration().models_dir+"\\"+low_model_drop
    else: low_model_path=None


    if Running_information["tab"] != "hires_txt2img":
        """if (Running_information["model"] != model_drop or Running_information["tab"] != "hires_txt2img"):        
        if (Running_information["tab"] == "txt2img") and (Running_information["model"] == model_drop):
            print("Converting in memory model txt2img to hires pipeline for faster loading")
            from Engine import txt2img_pipe
            pipe_old=txt2img_pipe().txt2img_pipe
            #print(f"txt2img:{pipe_old.unet}")
            new_pipe=txt2img_hires_pipe().Convert_from_txt2img(pipe_old,model_path,sch_t0)
            #print(f"hirestxt2img:{new_pipe.unet}")
            Running_information.update({"tab":"hires_txt2img"})
        else:"""
        UI_common.clean_memory_click()
        Running_information.update({"model":model_drop})
        Running_information.update({"tab":"hires_txt2img"})



    txt2img_hires_pipe().initialize(model_path,low_model_path,sch_t0)
    txt2img_hires_pipe().create_seeds(seed_t0,iter_t0,False)
    images= []
    images_low= []    
    information=[]
    counter=1
    img_index=Engine_common.get_next_save_index(output_path=UI_Configuration().output_path)
    strength_var=hires_pass_variation
    for seed in txt2img_hires_pipe().seeds:
        if running_config().Running_information["cancelled"]:
            break
        prompt,discard=gen_next_prompt(prompt_t0)
        print(f"Iteration:{counter}/{iter_t0}")
        counter+=1
        #batch_images,info = txt2img_hires_pipe().run_inference(
        lowres_image,hires_images,info = txt2img_hires_pipe().run_inference(
            prompt,
            neg_prompt_t0,
            hires_passes_t1,
            height_t0,
            width_t0,
            height_t1,
            width_t1,
            steps_t0,
            steps_t1,
            guid_t0,
            eta_t0,
            batch_t0,
            seed,
            strength,
            strength_var)
        
        for hires_image in hires_images:
            images.append(hires_image)
        images_low.append(lowres_image)
        info=dict(info)
        info['Sched:']=sch_t0
        info['HiResPasses']=hires_passes_t1
        info['HiResSteps:']=steps_t1
        info['Strength:']=strength
        information.append(info)

        style= running_config().Running_information["Style"]
        Engine_common.save_image(hires_images,info,img_index,UI_Configuration().output_path,style,save_textfile)
        #img_index+=1
        if save_low_res:
            Engine_common.save_image([lowres_image],info,img_index,UI_Configuration().output_path,low_res=True)
        img_index+=1          
    
    running_config().Running_information.update({"cancelled":False})
    gen_next_prompt("",True)
    Running_information.update({"Running":False})
    #return images,information

    return images,information,images_low


def convert_click(image,height,width):
    from Engine import txt2img_hires_pipe
    import numpy as np
    path="./latents"
    image=txt2img_hires_pipe().resize_and_crop(image,height,width)
  
    image=np.array(image)
    print(f"Resultant size of image(H/W):{image.shape[0]}x{image.shape[1]}")

    image = np.array(image).astype(np.float32) / 255.0

    image = 2.0 * image - 1.0
 

    image = np.expand_dims(image, axis=0)
    image = image.transpose(0, 3, 1, 2)

    # encode the init image into latents and scale the latents
    if txt2img_hires_pipe().hires_pipe!=None:
        vaeencoder=txt2img_hires_pipe().hires_pipe.vae_encoder
        init_latents = txt2img_hires_pipe().hires_pipe.vae_encoder(sample=image)[0]
    else:
        print("Initialize a model to process an image")
        """from Engine import Vae_and_Text_Encoders
        vaeencoder_session=Vae_and_Text_Encoders().load_vaeencoder("")

        input_names = {input_key.name: idx for idx, input_key in enumerate(vaeencoder_session.get_inputs())}
        print(input_names)
        output_names={output_key.name: idx for idx, output_key in enumerate(vaeencoder_session.get_outputs())}
        
        print(output_names)
        init_latents = vaeencoder_session.run({'latent_sample': 0},{'sample': image})[0]"""


    #print(f"init_latents:{type(init_latents)}")
    #print("Memory size of numpy array in bytes:",init_latents.nbytes)
    init_latents = 0.18215 * init_latents
    #print(init_latents)  
    savename=f"{path}/LastGenerated_latent.npy"
    np.save(savename, init_latents)
    print(f"Converted in:{savename}")
    return



