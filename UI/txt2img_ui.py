import gradio as gr
import os,gc,re
from Engine.General_parameters import Engine_Configuration as Engine_config
from Engine.General_parameters import running_config
from Engine.General_parameters import UI_Configuration
from Engine import pipelines_engines
from UI import UI_common_funcs as UI_common
from Engine import engine_common_funcs as Engine_common
from PIL import Image, PngImagePlugin


global next_prompt
global processed_images
processed_images=[]
next_prompt=None


def show_txt2img_ui():
    model_list = UI_common.get_model_list("txt2img")
    sched_list = get_schedulers_list()
    ui_config=UI_Configuration()
    gr.Markdown("Start typing below and then click **Generate** to see the output.")
    with gr.Row(): 
        with gr.Column(scale=13, min_width=650):
            model_drop = gr.Dropdown(model_list, value=(model_list[0] if len(model_list) > 0 else None), label="model folder", interactive=True)

            with gr.Accordion("Partial Reloads",open=False):
                reload_vae_btn = gr.Button("VAE Decoder:Apply Changes & Reload")
                reload_model_btn = gr.Button("Model:Apply new model & Fast Reload Pipe")
            with gr.Accordion(label="Latents experimentals",open=False):
                multiplier = gr.Slider(0, 1, value=0.18215, step=0.05, label="Multiplier, blurry the ingested latent, 1 to do not modify", interactive=True)
                strengh_t0 = gr.Slider(0, 1, value=0.8, step=0.05, label="Strengh, or % of steps to apply the latent", interactive=True)
                offset_t0 = gr.Slider(0, 100, value=1, step=1, label="Offset Steps for the scheduler", interactive=True)
                latents_experimental1 = gr.Checkbox(label="Save generated latents ", value=False, interactive=True)
                latents_experimental2 = gr.Checkbox(label="Load latent from a generation", value=False, interactive=True)
                name_of_latent = gr.Textbox(value="", lines=1, label="Name of Numpy- Latent")
                latent_formula = gr.Textbox(value="", lines=1, label="Formula for the sumatory of latents")
                latent_to_img_btn = gr.Button("Convert all latents to imgs", variant="primary", elem_id="gen_button")
            prompt_t0 = gr.Textbox(value="", lines=2, label="prompt")
            neg_prompt_t0 = gr.Textbox(value="", lines=2, label="negative prompt")
            sch_t0 = gr.Radio(sched_list, value=sched_list[0], label="scheduler")

            with gr.Row():
                iter_t0 = gr.Slider(1, 100, value=1, step=1, label="iteration count")
                batch_t0 = gr.Slider(1, 4, value=1, step=1, label="batch size", interactive=False)
            steps_t0 = gr.Slider(1, 3000, value=16, step=1, label="steps")
            guid_t0 = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
            height_t0 = gr.Slider(64, 2048, value=512, step=64, label="height")
            width_t0 = gr.Slider(64, 2048, value=512, step=64, label="width")
            eta_t0 = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=True)
            seed_t0 = gr.Textbox(value="", max_lines=1, label="seed")
            fmt_t0 = gr.Radio(["png", "jpg"], value="png", label="image format")
        with gr.Column(scale=11, min_width=550):
            with gr.Row():
                gen_btn = gr.Button("Generate", variant="primary", elem_id="gen_button")
                #clear_btn = gr.Button("Cancel",info="Cancel at end of current iteration",variant="stop", elem_id="gen_button")
                clear_btn = gr.Button("Cancel",variant="stop", elem_id="gen_button")
                memory_btn = gr.Button("Release memory", elem_id="mem_button")

            if ui_config.wildcards_activated:
                from UI import styles_ui
                styles_ui.show_styles_ui()
                with gr.Accordion(label="Live Prompt & Wildcards for multiple iterations",open=False):
                    with gr.Row():
                        next_wildcard = gr.Textbox(value="",lines=4, label="Next Prompt", interactive=True)
                        discard = gr.Textbox(value="", label="Discard", visible=False, interactive=False)
                    with gr.Row():
                        wildcard_show_btn = gr.Button("Show next prompt", elem_id="wildcard_button")
                        wildcard_gen_btn = gr.Button("Regenerate next prompt", variant="primary", elem_id="wildcard_button")
                        wildcard_apply_btn = gr.Button("Use edited prompt", elem_id="wildcard_button")
            with gr.Row():
                image_out = gr.Gallery(value=None, label="output images")

            with gr.Row():
                status_out = gr.Textbox(value="", label="status")
            with gr.Row():
                Selected_image_status= gr.Textbox(value="", label="status",visible=True)
                Selected_image_index= gr.Number(show_label=False, visible=False)
            with gr.Accordion(label="Edit/Save gallery images",open=False):
                with gr.Row():
                    delete_btn = gr.Button("Delete", elem_id="gallery_button")
                    save_btn = gr.Button("Save", elem_id="gallery_button")
                    #previous_btn = gr.Button("Previous", elem_id="gallery_button")
                    #next_btn = gr.Button("Next", elem_id="gallery_button")
  
    image_out.select(fn=get_select_index, inputs=[image_out,status_out], outputs=[Selected_image_index,Selected_image_status])
    delete_btn.click(fn=delete_selected_index, inputs=[Selected_image_index,status_out], outputs=[image_out,status_out])
    clear_btn.click(fn=UI_common.cancel_iteration,inputs=None,outputs=None)
    reload_vae_btn.click(fn=change_vae,inputs=model_drop,outputs=None)
    reload_model_btn.click(fn=change_model,inputs=model_drop,outputs=None)

    list_of_All_Parameters=[model_drop,prompt_t0,neg_prompt_t0,sch_t0,iter_t0,batch_t0,steps_t0,guid_t0,height_t0,width_t0,eta_t0,seed_t0,fmt_t0,multiplier,strengh_t0,name_of_latent,latent_formula,offset_t0]
    gen_btn.click(fn=generate_click, inputs=list_of_All_Parameters, outputs=[image_out,status_out])
    #sch_t0.change(fn=select_scheduler, inputs=sch_t0, outputs= None)  #Atencion cambiar el DDIM ETA si este se activa
    memory_btn.click(fn=UI_common.clean_memory_click, inputs=None, outputs=None)    
    

    latents_experimental1.change(fn=_activate_latent_save, inputs=latents_experimental1, outputs= None)
    latents_experimental2.change(fn=_activate_latent_load, inputs=[latents_experimental2,name_of_latent], outputs= None)
    latent_to_img_btn.click(fn=_latent_to_img,inputs=None,outputs=None)

    if ui_config.wildcards_activated:
        wildcard_gen_btn.click(fn=gen_next_prompt, inputs=prompt_t0, outputs=[discard,next_wildcard])
        wildcard_show_btn.click(fn=show_next_prompt, inputs=None, outputs=next_wildcard)
        wildcard_apply_btn.click(fn=apply_prompt, inputs=next_wildcard, outputs=None)

def _latent_to_img():
    import numpy as np
    latent_path="./latents"
    latent_list = []
    try:
        with os.scandir(latent_path) as scan_it:
            for entry in scan_it:
                if ".npy" in entry.name:
                    latent_list.append(entry.name)
        print(latent_list)                    
    except:
        print("Not numpys found.Wrong Directory or no files inside?")        


    vaedec=pipelines_engines.Vae_and_Text_Encoders().vae_decoder

    if vaedec==None:
        print("NO VAE LOADED, if you got a default VAE path configurated it will be loaded, if not, will raise an exception")
        vaedec= pipelines_engines.Vae_and_Text_Encoders().load_vaedecoder("") 

    for latent in latent_list:
        loaded_latent=np.load(f"./latents/{latent}")
        loaded_latent = 1 / 0.18215 * loaded_latent
        image=vaedec(latent_sample=loaded_latent)[0]  
        name= latent[:-3]
        name= name+"png"
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))
        image = Engine_common.numpy_to_pil(image)[0]
        image.save(f"./latents/{name}",optimize=True)

    return

def _activate_latent_save(activate_latent_save):
    from Engine.General_parameters import running_config
    running_config().Running_information.update({"Save_Latents":activate_latent_save})

def _activate_latent_load(activate_latent_load,name_of_latent):
    from Engine.General_parameters import running_config
    running_config().Running_information.update({"Load_Latents":activate_latent_load})


def change_vae(model_drop):
    from Engine.pipelines_engines import txt2img_pipe
    from Engine.pipelines_engines import Vae_and_Text_Encoders
    pipe=txt2img_pipe().txt2img_pipe
    vae=Vae_and_Text_Encoders()
    pipe.vae_decoder=vae.load_vaedecoder(f"{UI_Configuration().models_dir}\{model_drop}")
    return

def change_model(model_drop):
    #from Engine.pipelines_engines import txt2img_pipe
    from Engine.Pipelines.txt2img_pipeline import  txt2img_pipe
    pipe=txt2img_pipe().txt2img_pipe
    modelpath=f"{UI_Configuration().models_dir}\{model_drop}"
    pipe.unet=None
    gc.collect()
    pipe.unet=txt2img_pipe().reinitialize(modelpath)

    Running_information= running_config().Running_information
    Running_information.update({"model":model_drop})

    return



def delete_selected_index(Selected_image_index,status_out):
    global processed_images
    Selected_image_index=int(Selected_image_index)
    status_out=eval(status_out)
    processed_images.pop(Selected_image_index)
    status_out.pop(Selected_image_index)
    return processed_images,status_out


def get_select_index(image_out,status_out, evt:gr.SelectData):
    status_out=eval(status_out)
    return evt.index,status_out[evt.index]

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
            prompt=wildcards_process(prompt_t0)

        next_prompt=wildcards_process(prompt_t0)
    prompt = style_pre+" " +prompt+" " +style_post
    #print(f"Prompt:{prompt}")
    return prompt,next_prompt

def apply_prompt(prompt):
    global next_prompt
    next_prompt=prompt

def wildcards_process(prompt):
    from Scripts import wildcards
    wildcard=wildcards.WildcardsScript()
    new_prompt,discarded=wildcard.process(prompt)
    return new_prompt

def show_next_prompt():
    global next_prompt
    return next_prompt

def generate_click(
    model_drop,prompt_t0,neg_prompt_t0,sch_t0,
    iter_t0,batch_t0,steps_t0,guid_t0,height_t0,
    width_t0,eta_t0,seed_t0,fmt_t0,multiplier,
    strengh,name_of_latent,latent_formula,offset_t0):

    #from Engine.pipelines_engines import txt2img_pipe
    #from Engine.Pipelines.txt2img_pipeline import  txt2img_pipe
    from Engine import txt2img_pipe

    Running_information= running_config().Running_information
    Running_information.update({"Running":True})
    Running_information.update({"Latent_Name":name_of_latent})
    Running_information.update({"Latent_Formula":latent_formula})
    Running_information.update({"offset":offset_t0})


    if Running_information["Load_Latents"]:
        Running_information.update({"Latent_Name":name_of_latent})

    model_path=UI_Configuration().models_dir+"\\"+model_drop

    if Running_information["tab"] != "txt2img":  #Dejar asi hasta que modifique la rutina de cambio rapido de pipeline
        """if (Running_information["model"] != model_drop or Running_information["tab"] != "txt2img"):
        if (Running_information["tab"] == "hires_txt2img") and (Running_information["model"] == model_drop):
            print("Converting in memory model Hires to txt2img pipeline for faster loading")
            from Engine.Pipelines.txt2img_hires import txt2img_hires_pipe
            pipe_old=txt2img_hires_pipe().hires_pipe
            #print(f"hirestxt2img:{pipe_old.unet}")
            new_pipe=txt2img_pipe().Convert_from_hires_txt2img(pipe_old,model_path,sch_t0)
            #print(f"txt2img:{new_pipe.unet}")
            Running_information.update({"tab":"txt2img"})
        else:"""
        UI_common.clean_memory_click()
        Running_information.update({"model":model_drop})
        Running_information.update({"tab":"txt2img"})
        


    pipe=txt2img_pipe().initialize(model_path,sch_t0)
    txt2img_pipe().create_seeds(seed_t0,iter_t0,False)
    images= []
    information=[]
    counter=1
    img_index=Engine_common.get_next_save_index(output_path=UI_Configuration().output_path)
    for seed in txt2img_pipe().seeds:
        if running_config().Running_information["cancelled"]:
            break
        prompt,discard=gen_next_prompt(prompt_t0)
        print(f"Iteration:{counter}/{iter_t0}")
        counter+=1
        batch_images,info = txt2img_pipe().run_inference(
            prompt,
            neg_prompt_t0,
            height_t0,
            width_t0,
            steps_t0,
            guid_t0,
            eta_t0,
            batch_t0,
            seed,multiplier,strengh)
        images.extend(batch_images)
        info=dict(info)
        info['Sched:']=sch_t0
        info['Multiplier:']=multiplier
        information.append(info)
        style= running_config().Running_information["Style"]
        Engine_common.save_image(batch_images,info,img_index,UI_Configuration().output_path,style)        
        #Engine_common.save_image(batch_images,info,img_index,UI_Configuration().output_path)
        img_index+=1

    running_config().Running_information.update({"cancelled":False})
    global processed_images
    processed_images=images
    gen_next_prompt("",True)
    Running_information.update({"Running":False})
    return images,information
    #return separadas,information

