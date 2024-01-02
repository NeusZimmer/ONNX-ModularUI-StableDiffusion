
import gradio as gr


#### MODULE NECESSARY FUNCTIONS (__init__ , show and __call__) ####
def __init__(*args):
    __name__='ImageToNumpyModule'
    print(args[0])
    #here a check of access of initializacion  if needed
    # must return a dict: tabs: in which tabs is to be shown,ui_position: area within the UI tab and func_processing where is to be processed the data
    return {
        'tabs':["hires"],
        'ui_position':"tools_area",
        'func_processing':None}

def is_global_ui():
    return False

def is_global_function():
    return False

def show():
    show_ImageToNumpyModule_ui()

def __call__(datos):
    return datos



##### MAIN MODULE CODE #####


def show_ImageToNumpyModule_ui(*args):
    args= [""] if not args else args

    with gr.Accordion(label="Process IMG To latent",open=False):
        with gr.Row():                 
            image_in = gr.Image(label="input image", type="pil", elem_id="image_init")
        with gr.Row():
            height = gr.Slider(64, 2048, value=512, step=64, label="Output Height")
            width = gr.Slider(64, 2048, value=512, step=64, label="Output Width")
        with gr.Row():            
            convert_to_latent_btn = gr.Button("Convert IMG to latent/numpy on disk")
            analyze_btn = gr.Button("Show info on image and conversion")
        with gr.Row():                        
            information = gr.Textbox(value=args[0], lines=1, label="Information",interactive=False)


        convert_to_latent_btn.click(fn=convert_click,inputs=[image_in,height,width],outputs=information)
        analyze_btn.click(fn=get_info,inputs=[image_in,height,width],outputs=information)

    return

def get_info(image,height,width):
    image_size=image.size
    image_ratio= str(image_size[0]/image_size[1])[0:4]

    destination_ratio=str(width/height)[0:4]

    return f"Image Size(H/W):{(image_size[1],image_size[0])} is ratio:{image_ratio}, destination ratio:{destination_ratio}"

def convert_click(image,height,width):
    import numpy as np
    from Engine.General_parameters import running_config

    Running_information= running_config().Running_information
    vaeencoder=False

    if Running_information["tab"] == "hires_txt2img":
        from Engine import txt2img_hires_pipe
        if txt2img_hires_pipe().hires_pipe!=None:
            vaeencoder=txt2img_hires_pipe().hires_pipe.vae_encoder
    """elif Running_information["tab"] == "txt2img":  #txt2img no tiene encoder usar inpaint o load uno para el proceso?
        from Engine import txt2img_pipe
        if txt2img_pipe().txt2img_pipe!=None:
            vaeencoder=txt2img_pipe().txt2img_pipe.vae_encoder"""


    image=resize_and_crop(image,height,width)
    image=np.array(image)
    image = np.array(image).astype(np.float32) / 255.0
    image = 2.0 * image - 1.0
    image = np.expand_dims(image, axis=0)
    image = image.transpose(0, 3, 1, 2)

    # encode the init image into latents and scale the latents

    if vaeencoder==False:
        from Engine import Vae_and_Text_Encoders
        vaeencoder_session=Vae_and_Text_Encoders().load_vaeencoder("")
        try:
            init_latents = vaeencoder_session.run(['latent_sample'],{'sample': image})[0]
        except:
            return ("Initialize a model to process an image or configure a standard vae encoder")

        """
        #Esto es para sacar los nombres para poder ejecutar la sesion.
        input_names = {input_key.name: idx for idx, input_key in enumerate(vaeencoder_session.get_inputs())}
        print(input_names)
        output_names={output_key.name: idx for idx, output_key in enumerate(vaeencoder_session.get_outputs())}
        print(output_names)
        init_latents = None"""

    else:
        init_latents = vaeencoder(sample=image)[0]

    init_latents = 0.18215 * init_latents
    path="./latents"
    savename=f"{path}/LastGenerated_latent.npy"
    np.save(savename, init_latents)

    return f"Converted in:{savename} with size of (H/W):{image.shape[0]}x{image.shape[1]}"

def resize_and_crop(input_image, height, width):
    from PIL import Image
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

if __name__ == "__main__":
    print("This is a module not intended to run as standalone")
    with gr.Blocks(title="Image Converter") as demo:
        show_ImageToNumpyModule_ui("This is a module not intended to run as standalone, will fail loading the model for converting the image or saving to disk")
    demo.launch()

    pass
