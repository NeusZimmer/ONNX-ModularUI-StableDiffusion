import os,gc,re
from Engine.General_parameters import running_config
from Engine.General_parameters import Engine_Configuration as Engine_config
#from Engine import pipelines_engines

import gradio as gr
from PIL import Image #, PngImagePlugin

from Engine.General_parameters import UI_Configuration
from Engine import SchedulersConfig
from Engine import engine_common_funcs as Engine_common

from UI import UI_common_funcs as UI_common
from UI import UI_placement_areas
from modules.modules import preProcess_modules


global list_modules
global number_of_passes
global next_prompt
global processed_images
processed_images=[]
next_prompt=None
list_modules=[]
list_modules=preProcess_modules().check_available_modules("hires")

######### UI Layout ########################################################

def show_HiRes_txt2img_ui():
    global list_modules
    model_list = UI_common.get_model_list("txt2img")
    sched_list = get_schedulers_list()
    ui_config=UI_Configuration()
    gr.Markdown("Start typing below and then click **Generate** to see the output.")
    with gr.Row(): 
        with gr.Column(scale=13, min_width=650):
            model_drop = gr.Dropdown(model_list, value=(model_list[0] if len(model_list) > 0 else None), label="HiRes model", interactive=True)
            with gr.Row():
                model_list=["None"]+model_list
                low_model_drop = gr.Dropdown(model_list, value='None', label="LowRes model (None=Same as HiRes)", interactive=True)
                model_update_btn=gr.Button("Reload List").click(fn=renew_models,inputs=None,outputs=[model_drop,low_model_drop])
            with gr.Accordion("Additional options",open=False):
                reload_lowres_btn = gr.Button("Reload Low-res model")
                reload_hires_btn = gr.Button("Reload HiRes Model")           
                copy_hires_to_lowres_btn = gr.Button("Use HiRes model as LowRes")
                reload_scheduler_btn=gr.Button("Reload Scheduler",visible=True).click(fn=reload_scheduler,inputs=None,outputs=None)
                test_vae_split_btn = gr.Button("Test VAE Split",visible=False)
                #reload_model_btn = gr.Button("Model:Apply new model & Fast Reload Pipe")
                reload_vae_dec_btn = gr.Button("VAE Decoder:Apply Changes & Reload")
                reload_vae_enc_btn = gr.Button("VAE Encoder:Apply Changes & Reload")
                reload_textenc_btn = gr.Button("Text Encoder Reload")

            scheduler = gr.Radio(sched_list, value=sched_list[0], label="scheduler")
            prompt = gr.Textbox(value="", lines=2, label="prompt")
            neg_prompt = gr.Textbox(value="", lines=2, label="negative prompt")

            with gr.Row():
                UI_placement_areas.show_tools_area(list_modules)

            with gr.Accordion(label="Latents experimentals",open=False):
                latents_experimental2 = gr.Checkbox(label="Load latent from a generation", value=False, interactive=True)
                name_of_latent = gr.Textbox(value="1:LastGenerated_Latent.npy", lines=1, label="Names of Numpy File Latents (1:x.npy,2:y.pt,3:noise-(width)x(height))")
                latent_formula = gr.Textbox(value="1", lines=1, label="Formula for the sumatory of latents")


            with gr.Row():
                gr.Markdown("Common parameters")
            with gr.Row():
                strength = gr.Slider(0, 1, value=0.6, step=0.05, label="Strength of the low-res/input image", interactive=True)
            with gr.Row():
                iterations = gr.Slider(1, 100, value=1, step=1, label="iteration count")
                guid = gr.Slider(0, 50, value=7.5, step=0.1, label="guidance")
                batch = gr.Slider(1, 4, value=1, step=1, label="batch size", interactive=False,visible=False)
            with gr.Row():                
                eta = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", interactive=True)
                img_format = gr.Radio(["pil", "latent"], value="pil", label="Output format")
                seed = gr.Textbox(value="", max_lines=1, label="seed")
            
            with gr.Row():
                gr.Markdown("HiRes parameters")
            with gr.Row():                         
                height = gr.Slider(64, 2048, value=512, step=64, label="HiRes height")
                width = gr.Slider(64, 2048, value=512, step=64, label="HiRes width")
            with gr.Row():                
                steps_hires = gr.Slider(1, 100, value=16, step=1, label="HiRes steps")
                hires_passes = gr.Slider(1, 10, value=2, step=1, label="HiRes passes")
                hires_pass_variation = gr.Slider(-1, 1, value=0, step=0.1, label="Variation of Strengh between between passes")

            with gr.Row():
                gr.Markdown("First pass parameters")
            with gr.Row():                
                height_low = gr.Slider(64, 2048, value=512, step=64, label="height")
                width_low = gr.Slider(64, 2048, value=512, step=64, label="width")
                steps_low = gr.Slider(1, 3000, value=16, step=1, label="steps")                

            with gr.Row():
                gr.Markdown("Other parameters")
            with gr.Row():
                upscale_method=gr.Radio(["Torch", "VAE"], value="Torch", label="Upscale method", interactive=True,visible=True)
                save_textfile=gr.Checkbox(label="Save prompt into a txt file")
                save_low_res=gr.Checkbox(label="Save generated Low-Res Img")

        with gr.Column(scale=11, min_width=550):
            with gr.Row():
                gen_btn = gr.Button("Generate", variant="primary", elem_id="gen_button")
                clear_btn = gr.Button("Cancel",variant="stop", elem_id="gen_button").click(fn=UI_common.cancel_iteration,inputs=None,outputs=None)
                memory_btn = gr.Button("Release memory", elem_id="mem_button")
            with gr.Accordion(label="Prompt Processing tools",open=False):
                with gr.Accordion(label="Live Prompt for multiple iterations & Prompt pre-generation",open=False):
                    with gr.Row():
                        next_wildcard = gr.Textbox(value="",lines=4, label="Next Prompt", interactive=True)
                        discard = gr.Textbox(value="", label="Discard", visible=False, interactive=False)
                    with gr.Row():
                        wildcard_show_btn = gr.Button("Show next prompt", elem_id="wildcard_button")
                        wildcard_gen_btn = gr.Button("Regenerate next prompt", variant="primary", elem_id="wildcard_button")
                        wildcard_apply_btn = gr.Button("Use edited prompt", elem_id="wildcard_button")
                ######## Show prompt preprocess areas ######## 
                UI_placement_areas.show_prompt_preprocess_area(list_modules)
                ######## ########  ######## ########  ########
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

    ######## Show footer areas #############################
    with gr.Row():
        UI_placement_areas.show_footer_area(list_modules)
    ########################################################
        
    image_out.select(fn=get_select_index, inputs=[image_out,status_out], outputs=[Selected_image_index,Selected_image_status])



    reload_vae_dec_btn.click(fn=change_vae_dec,inputs=model_drop,outputs=None)
    reload_vae_enc_btn.click(fn=change_vae_enc,inputs=model_drop,outputs=None)
    reload_textenc_btn.click(fn=change_textenc,inputs=model_drop,outputs=None)


    list_of_All_Parameters=[model_drop,low_model_drop,prompt,neg_prompt,scheduler,iterations,batch,steps_low,steps_hires,guid,height_low,width_low,height,width,eta,seed,img_format,strength,hires_passes,save_textfile, save_low_res,latent_formula,name_of_latent,latents_experimental2,hires_pass_variation,upscale_method]    

    memory_btn.click(fn=UI_common.clean_memory_click, inputs=None, outputs=None)    
    reload_hires_btn.click(fn=reload_hires,inputs=model_drop,outputs=None)
    reload_lowres_btn.click(fn=reload_lowres,inputs=low_model_drop,outputs=None)
    copy_hires_to_lowres_btn.click(fn=copy_hires,inputs=None,outputs=low_model_drop)
    test_vae_split_btn.click(fn=vae_output_to_numpy,inputs=None,outputs=image_out)    

    gen_btn.click(fn=generate_click,inputs=list_of_All_Parameters,outputs=[image_out,status_out,low_res_image_out])


    
    wildcard_gen_btn.click(fn=gen_next_prompt, inputs=prompt, outputs=[discard,next_wildcard])
    wildcard_show_btn.click(fn=show_next_prompt, inputs=None, outputs=next_wildcard)
    wildcard_apply_btn.click(fn=apply_prompt, inputs=next_wildcard, outputs=None)




######### UI Creation and Management Funcs ########################################################
def get_schedulers_list():
    sched_config = SchedulersConfig()
    sched_list =sched_config.available_schedulers
    return sched_list

def renew_models():
    model_list = UI_common.get_model_list("txt2img")
    return gr.Dropdown.update(choices=model_list, value=(model_list[0] if len(model_list) > 0 else None)),gr.Dropdown.update(choices=["None"]+model_list, value="None"),

#def select_scheduler(sched_name,model_path):
#    return SchedulersConfig().scheduler(sched_name,model_path)

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

####################################################################################
#UNDEFINED & TEST FUNCS

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


######## RELOAD AREAS ##########################################

def reload_scheduler():
    SchedulersConfig().reload()

def copy_hires():
    from Engine import txt2img_hires_pipe
    pipe=txt2img_hires_pipe()
    pipe.use_hires_as_lowres()
    #pipe.hires_pipe.unet_session=pipe.hires_pipe.low_unet_session
    #pipe.hires_pipe.low_unet_session=pipe.hires_pipe.unet_session
    print("Copied current Hires model onto lowres model")
    return gr.Dropdown.update(value="None")

def reload_lowres(low_model_drop):
    from Engine import txt2img_hires_pipe
    pipe=txt2img_hires_pipe()
    model_path=UI_Configuration().models_dir+"\\"+low_model_drop
    if low_model_drop=="None":
        pipe.use_hires_as_lowres()
        print("Using Hires model as lowres model")
    else:
        pipe.reload_partial_model(model_path,"low")

def reload_hires(model_drop):
    from Engine import txt2img_hires_pipe
    pipe=txt2img_hires_pipe()
    model_path=UI_Configuration().models_dir+"\\"+model_drop
    pipe.reload_partial_model(model_path,"high")

def change_textenc(model_drop):
    from Engine import txt2img_hires_pipe
    from Engine import Vae_and_Text_Encoders
    class_engine=Vae_and_Text_Encoders()
    textenc_session=class_engine.load_textencoder(f"{UI_Configuration().models_dir}\{model_drop}")
    txt2img_hires_pipe().hires_pipe.text_encoder=class_engine.convert_session_to_model(textenc_session,"textenc")
    return

def change_vae_dec(model_drop):
    from Engine import txt2img_hires_pipe
    from Engine import Vae_and_Text_Encoders
    class_engine=Vae_and_Text_Encoders()
    vae_session=class_engine.load_vaedecoder(f"{UI_Configuration().models_dir}\{model_drop}")
    txt2img_hires_pipe().hires_pipe.vae_decoder=class_engine.convert_session_to_model(vae_session,"vaedec")
    return

def change_vae_enc(model_drop):
    from Engine import txt2img_hires_pipe
    from Engine import Vae_and_Text_Encoders
    class_engine=Vae_and_Text_Encoders()
    vae_session=class_engine.load_vaeencoder(f"{UI_Configuration().models_dir}\{model_drop}")
    txt2img_hires_pipe().hires_pipe.vae_encoder=class_engine.convert_session_to_model(vae_session,"vaeenc")
    return


######## PROMPT PROCESSING ##########################################

def gen_next_prompt(prompt,initial=False):
    global next_prompt
    prompt_i=prompt
    if (initial):
        next_prompt=None
        prompt=prompt_process(prompt)
    else:
        if (next_prompt != None):
            prompt=next_prompt
        else:
            prompt=prompt_process(prompt)

        next_prompt=prompt_process(prompt_i)

    return prompt,next_prompt

def prompt_process(prompt):
    global list_modules
    for module in list_modules:
        if module['func_processing']=="prompt_process": #Area of modules for processing prompts
            prompt=module['call'](prompt) #modules[2]= show, #modules[3] =process

    return prompt

def show_next_prompt():
    global next_prompt
    return next_prompt

def apply_prompt(prompt):
    global next_prompt
    next_prompt=prompt


###############################################################
###############################################################
########               IMAGE GENERATION                ########
###############################################################
###############################################################


def generate_click(
    model_drop,low_model_drop,prompt,neg_prompt,scheduler,
    iterations,batch,steps_low,steps_hires,guid,height_low,
    width_low,height,width,eta,seed,img_format,strength,
    hires_passes,save_textfile, save_low_res,
    latent_formula,name_of_latent,latents_experimental2,hires_pass_variation,upscale_method):

    # We just create a dict with the same variable names as keys

    list_of_All_Parameters="model_drop,low_model_drop,prompt,neg_prompt,scheduler,iterations,batch,steps_low,steps_hires,guid,height_low,width_low,height,width,eta,seed,img_format,strength,hires_passes,save_textfile,save_low_res,latent_formula,name_of_latent,latents_experimental2,hires_pass_variation,upscale_method"
    names=list_of_All_Parameters.split(',')

    retorno={}
    for elemento in names:
        try:
            retorno[elemento]=eval(elemento)
        except:
            print("Error converting to dict.")

    return generate_click2(kwargs=retorno)

def generate_click2(kwargs):
    from Engine.General_parameters import running_config

    if kwargs['latents_experimental2']:
        running_config().Running_information.update({"Load_Latents":True})
        running_config().Running_information.update({"Latent_Name":kwargs['name_of_latent']})
        running_config().Running_information.update({"Latent_Formula":kwargs['latent_formula']})
    else:
        running_config().Running_information.update({"Load_Latents":False})

    from Engine import txt2img_hires_pipe

    global number_of_passes
    number_of_passes= kwargs['hires_passes']

    Running_information= running_config().Running_information
    Running_information.update({"Running":True})


    model_path=UI_Configuration().models_dir+"\\"+kwargs['model_drop']
    if kwargs['low_model_drop'] != 'None': low_model_path=UI_Configuration().models_dir+"\\"+kwargs['low_model_drop']
    else: low_model_path=None


    if Running_information["tab"] != "hires_txt2img":
        """if (Running_information["model"] != kwargs['model_drop'] or Running_information["tab"] != "hires_txt2img"):        
        if (Running_information["tab"] == "txt2img") and (Running_information["model"] == kwargs['model_drop']):
            print("Converting in memory model txt2img to hires pipeline for faster loading")
            from Engine import txt2img_pipe
            pipe_old=txt2img_pipe().txt2img_pipe
            #print(f"txt2img:{pipe_old.unet}")
            new_pipe=txt2img_hires_pipe().Convert_from_txt2img(pipe_old,model_path,scheduler)
            #print(f"hirestxt2img:{new_pipe.unet}")
            Running_information.update({"tab":"hires_txt2img"})
        else:"""
        UI_common.clean_memory_click()
        Running_information.update({"model":kwargs['model_drop']})
        Running_information.update({"tab":"hires_txt2img"})



    txt2img_hires_pipe().initialize(model_path,low_model_path,kwargs['scheduler'])
    txt2img_hires_pipe().create_seeds(kwargs['seed'],kwargs['iterations'],False)


    if type(txt2img_hires_pipe().seeds)==list:
        kwargs['seed']=txt2img_hires_pipe().seeds
    else:
        kwargs['seed']=txt2img_hires_pipe().seeds.tolist()  #seeds are a numpy array, 1 dimension or a list, no other options?


    images= []
    images_low= []    
    information=[]
    counter=1
    img_index=Engine_common.get_next_save_index(output_path=UI_Configuration().output_path)


    ############### CALLS FOR MODULES INITIAL_DATA  ###############
    global list_modules
    for module in list_modules:
        if module['func_processing']=="initial_data": #Area of modules for processing the initial dict
            kwargs=module['call'](kwargs)

    ############### ############### ############### ###############
    
    for seed in txt2img_hires_pipe().seeds:
        if running_config().Running_information["cancelled"]:
            break
        prompt,_discard=gen_next_prompt(kwargs['prompt']) #make sure we always use the initial received prompt
        #neg_prompt=prompt_process(kwargs['neg_prompt']) #publish with this one commented instead of next
        neg_prompt=kwargs['neg_prompt'] 
        print(f"Iteration:{counter}/{kwargs['iterations']}")
        counter+=1
        
        #lowres_image,hires_images,info = txt2img_hires_pipe().run_inference(
        lowres_image,hires_images = txt2img_hires_pipe().run_inference(
            prompt,
            neg_prompt,
            #kwargs['neg_prompt'],
            kwargs['hires_passes'],
            kwargs['height_low'],
            kwargs['width_low'],
            kwargs['height'],
            kwargs['width'],
            kwargs['steps_low'],
            kwargs['steps_hires'],
            kwargs['guid'],
            kwargs['eta'],
            kwargs['batch'],
            seed,
            kwargs['strength'],
            kwargs['hires_pass_variation'], #strengh variation
            kwargs['upscale_method'],
            kwargs['img_format'],
            )
        
        
        #info=dict(info)
        info=kwargs.copy()
        info['prompt']= prompt
        info['neg_prompt']=neg_prompt

        info['seed']=seed

        """info['Sched:']= kwargs['scheduler']
        info['HiResPasses']=kwargs['hires_passes']
        info['HiResSteps:']=kwargs['steps_hires']
        info['Strength:']=kwargs['strength']"""
        information.append(info)

        style= running_config().Running_information["Style"]
        #print(kwargs['img_format'])
        if kwargs['img_format']=='pil':
            for hires_image in hires_images:
                images.append(hires_image)
            images_low.append(lowres_image)
            Engine_common.save_image(hires_images,info,img_index,UI_Configuration().output_path,style,kwargs['save_textfile'])
            if kwargs['save_low_res']:
                Engine_common.save_image([lowres_image],info,img_index,UI_Configuration().output_path,low_res=True)
        else:
            #salvar latents
            import os, numpy
            path="./latents"
            numpy.save(f"{path}/{seed}-Low-Res_img.npy", numpy.array(lowres_image))  #aqui lowres
            with open(os.path.join(path,f"{seed}-info_img.txt"), 'w',encoding='utf8') as txtfile:
                txtfile.write(str(info))
            c=0
            for latent in  hires_images:
                numpy.save(f"{path}/{seed}-Hi-Res_img_{c}.npy", numpy.array(latent))
                c+=1
            from PIL import Image
            fake_image=Image.fromarray(numpy.ones([64,64,3],dtype=numpy.uint8))
            for hires_image in hires_images:
                images.append(fake_image)
            images_low.append(fake_image)

        img_index+=1          
    
    running_config().Running_information.update({"cancelled":False})
    gen_next_prompt("",True)
    Running_information.update({"Running":False})


    return images,information,images_low

