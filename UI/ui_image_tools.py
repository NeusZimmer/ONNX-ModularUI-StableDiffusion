import gradio as gr
import gc,os
from Scripts import deepdanbooru_onnx as DeepDanbooru_Onnx
from Scripts import codeformer_onnx as codeformer_onnx
from Scripts import facedetector_onnx as facedetector_onnx
from Scripts import image_slicer
from Scripts import superresolution as SPR
from Engine import pipelines_engines

global debug
global danbooru
global facedect
global image_in
global image_out
danbooru = None
facedect = None
debug = False
global faces
faces = []
global new_faces
new_faces = []
global boxes
boxes = []
global index
index=0



def show_input_image_area():
    global image_in
    global image_out
    with gr.Row():
        #with gr.Column(variant="compact"):
        image_in = gr.Image(label="input image", type="pil", elem_id="image_init")
#with gr.Column():
    with gr.Row():
        image_out = gr.Image(label="Output image", type="pil", elem_id="image_out")
def show_danbooru_area():
    with gr.Accordion(label="Danbooru tagging",open=False):
        with gr.Row():
            apply_btn = gr.Button("Analyze image with Deep DanBooru", variant="primary")
            mem_btn = gr.Button("Unload from memory")
        with gr.Row():
            results = gr.Textbox(value="", lines=8, label="Results")

    mem_btn.click(fn=unload_DanBooru, inputs=results , outputs=results)
    apply_btn.click(fn=analyze_DanBooru, inputs=image_in , outputs=results)


def analyze_DanBooru(image):
    global danbooru
    if danbooru == None:
        danbooru = DeepDanbooru_Onnx.DeepDanbooru()
    results=danbooru(image)
    results2=str(results.keys())
    results2=results2.replace("'","")
    results2=results2.replace("dict_keys([","")
    results2=results2.replace("])","")
    #return list(results)
    return results2

def unload_DanBooru(results):
    global danbooru
    danbooru= None
    gc.collect
    return results+"\nUnloaded from memory of provider"

def show_image_resolution_area():
    global image_out
    with gr.Accordion(label="Up-Scale & Resize",open=False):
        with gr.Row():
            with gr.Accordion(label="Stable Diffusion Upscale",open=False):  
                resolution_btn2 = gr.Button("Resize image with Stable Diffusion 4x", variant="primary")
                resolution_prompt = gr.Textbox(value="", lines=8, label="Prompt for Stable Diffusion Upscale")
            with gr.Accordion(label="ONNX Upscale",open=False):  
                resolution_btn = gr.Button("Resize image with ONNX", variant="primary")
                with gr.Row():
                    gr.Markdown("Choose number of divisions to be applied (higher means higher resolution)"+"\Image Size: height=512xcolums,512xrows, (keeping proportions based on its larger size)")
                    checkbox1 = gr.Checkbox(label="3 Steps", info="Try to reduce the slicing/join lines, inscreases time a lot")
                    img_rows = gr.Slider(2, 30, value=4, label="Row Divisions", step=1)
                    img_cols = gr.Slider(2, 30, value=4, label="Column Divisions",step=1)
                with gr.Row():
                    Mem_btn = gr.Button("Clean SuperResolution Memory")
                    delete_btn = gr.Button("Delete Temp dir")


    with gr.Accordion(label="Face Restoration & Area Inpainting",open=False):      
        with gr.Row():
            with gr.Column():
                with gr.Row():                
                    facedect_btn = gr.Button("Analyze faces")
                with gr.Row():
                    with gr.Column():
                        image_out2 = gr.Gallery(label="Output Faces")
                    with gr.Column():
                        with gr.Accordion(label="Face restoration",open=False):                           
                            with gr.Row():       
                                select_face_btn = gr.Button("Select Area & Upscale")
                            with gr.Row():       
                                paste_faces_btn = gr.Button("Paste faces", variant="primary")
                            """with gr.Row():
                            with gr.Accordion(label="Area Inpainting",open=False):"""      
                            with gr.Row():
                                point_x = gr.Slider(0, 4096, value=64, label="X Axis", step=64,interactive=True)
                                point_y = gr.Slider(0, 4096, value=64, label="Y Axis", step=64,interactive=True)
                                side_size = gr.Slider(64, 1024, value=64, label="Box Size", step=64,interactive=True)
                            with gr.Row():
                                select_area_btn = gr.Button("Select Box from point in image and show")   
                                select_area_btn2 = gr.Button("Put output image as input")
                                send_gfpgan = gr.Button("Send full image to Gfpgan")
                        with gr.Accordion(label="Mask Detection",open=True):
                            detect_masks_btn = gr.Button("Masks detection")
                            list_of_masks = gr.Gallery(label="Output Masks")
                                                            
                with gr.Row():
                    face_image_in = gr.Image(label="Face & Mask", tool="sketch", type="pil", interactive=True)
                with gr.Row():
                    generate_face_btn = gr.Button("Send to Inpaint, Generate Face", variant="primary")
                    checkbox2 = gr.Checkbox(label="Inpaint",value=True, info="Inpainting Active")
                    checkbox3 = gr.Checkbox(label="GFPGan",value=False, info="GFPGan Active")
                    steps = gr.Slider(1, 100, value=20, label="Steps", step=1)
                    seed_in = gr.Textbox(label="Seed",value="", lines=1)      
                with gr.Row():
                    prompt_in = gr.Textbox(label="Prompt",value="", lines=1)
                    neg_prompt_in = gr.Textbox(label="Neg Prompt",value="blurry, bad anatomy, undefined, jpeg artifacts, noise", lines=1)                    
        with gr.Row():
            mem_facedect_btn = gr.Button("Unload from memory")
    """with gr.Row():
        convert_to_numpy_btn = gr.Button("Resize to and then convert to numpy", variant="primary")
        height = gr.Slider(64, 2048, value=512,step=64, label="Height")
        width = gr.Slider(64, 2048, value=512,step=64, label="Width")"""

    detect_masks_btn.click(fn=detect_masks_fn, inputs=face_image_in , outputs=list_of_masks)
    mem_facedect_btn.click(fn=unload_facedect, inputs=None , outputs=None)
    facedect_btn.click(fn=analyze_Faces, inputs=image_in , outputs=[image_out,image_out2])
    paste_faces_btn.click(fn=Paste_Faces, inputs=image_in , outputs=[image_out,image_out2])
    select_face_btn.click(fn=select_face, inputs=None , outputs=face_image_in)
    image_out2.select(fn=selected_index,inputs=None)
    #inpaint_face_btn.click(fn=inpaint_face, inputs=[image_in,prompt_in], outputs=[image_out,image_out2])
    #convert_to_numpy_btn.click(fn=img_to_numpy, inputs=[image_in,height,width] , outputs=None)
    generate_face_btn.click(fn=generate_click, inputs=[prompt_in,neg_prompt_in,seed_in,face_image_in,steps,checkbox2,checkbox3] , outputs=[face_image_in,seed_in])
    resolution_btn2.click(fn=Resize_Image2, inputs=[image_in,resolution_prompt], outputs=image_out)
    resolution_btn.click(fn=Resize_Image, inputs=[image_in,img_rows,img_cols,checkbox1], outputs=image_out)
    Mem_btn.click(fn=Clean_SPR_Mem, inputs=None, outputs=image_out)
    delete_btn.click(fn=DeleteTemp, inputs=None , outputs=None)
    select_area_btn.click(fn=Select_Box, inputs=[image_in,point_x,point_y,side_size] , outputs=[image_out,image_out2])
    select_area_btn2.click(fn=exchange_image, inputs=[image_out] , outputs=[image_in])
    send_gfpgan.click(fn=gfpgan_image, inputs=[image_out] , outputs=[image_out])


def detect_masks_fn(image_in):
    image_in=image_in['image']
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry['vit_b'](checkpoint="sam_vit_b_01ec64.pth")
    sam.to('cpu')
    output_mode = "binary_mask"
    amg_kwargs={
        "pred_iou_thresh": 0.5,
        "stability_score_thresh": 0.5,
        "stability_score_offset": 0.5,
    }
    #generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)   
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    import numpy as np
    image_in=image_in.convert("RGB")
    image = np.asarray(image_in)
    masks = generator.generate(image)
    #masks=masks*255
    from PIL import Image
    mascaras=[]
    for i, mask_data in enumerate(masks):
        print(f"Mascara:{i}")
        mask = mask_data["segmentation"]
        mascaras.append(Image.fromarray(mask*255))

    return mascaras

    """import onnxruntime as ort
    
    from segment_anything.utils.onnx import SamOnnxModel    
    sess_options = ort.SessionOptions()
    #sess_options.log_severity_level=3
    provider=['DmlExecutionProvider', 'CPUExecutionProvider'] 
    onnx_model_path="./facebook_sam.onnx"""
    """checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)"""
    """
    ort_session = ort.InferenceSession(onnx_model_path,providers=provider)
    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }
    masks, _, low_res_logits = ort_session.run(None, ort_inputs)"""

    #return (image_in,image_in)

def gfpgan_image(image_t0):
    print("Entrada:")
    print(image_t0)
    print(type(image_t0))
    if isinstance(image_t0, list) or isinstance(image_t0, dict):
        result= image_t0["image"]
        #mask= image_t0["mask"].convert("RGB")
    else:
        from PIL import Image
        result= image_t0
        #mask= Image.new("RGB", (result.size), (0, 0, 0))
    return generate_click_gfpgan(result,True)

def exchange_image(image_out):
    return image_out

def Select_Box(image_in,point_x,point_y,side_size):
    global facedect
    if facedect == None:
        facedect = facedetector_onnx.FaceDetector()
    imgwithboxes,box,imgarea =facedect.select_box(image_in,point_x, point_y,side_size)

    global faces
    global new_faces    
    global index
    global boxes
    index=0
    import numpy as np
    faces=[]
    boxes=[]

    faces.append(imgarea)
    boxes.append(box)
    new_faces=[None]*(len(faces))    
    boxes=np.asarray(boxes)
 
    return imgwithboxes,faces

def selected_index(evt:gr.SelectData):
    global index
    index=evt.index

def img_to_numpy(image,height,width):
    from PIL import Image
    import PIL
    import numpy as np
    from Engine import pipelines_engines

    vaeenc=pipelines_engines.Vae_and_Text_Encoders().vae_encoder
    if vaeenc==None:
        print("NO VAE LOADED")
        import Engine.pipelines_engines as Engines
        model_path="C:\\AMD_ML\\models\\stable-diffusion-x4-dddsd-onnx"
        vaeenc= Engines.Vae_and_Text_Encoders().load_vaeencoder(model_path) 
    

    image =image.resize([width,height], Image.Resampling.LANCZOS)
    image = np.expand_dims(image,axis=0)
    image = np.array(image).astype(np.float32) / 255.0

    image = image.transpose(0, 3, 1, 2)
    print(image.shape)

    image_latents = vaeenc(sample=image)[0]
    #image = image * 0.08333
    #image = 2.0 * image - 1.0
    image = image * 0.18215    
    np.save(f"./latents/test.npy", image_latents)

def select_face():
    global faces
    global index
    from PIL import Image
    global boxes
    box = boxes[index]
    image=faces[index]

    image =image.resize([512, 512], Image.Resampling.LANCZOS)
    #image =image.resize([256, 256], Image.Resampling.LANCZOS)
    #image.save(f"./facetempindex{index}_{box[2]-box[0]}_{box[0]}x{box[2]}.png", "png")
    #SPR.superresolution_process(f"./facetemp1_{box[0]*box[2]}.png")
    #image=Image.open(f"./facetempindex_{index}_{box[2]-box[0]}_{box[0]}x{box[2]}.png")
    faces[index]=image
    return image

def analyze_Faces(image):
    global facedect
    if facedect == None:
        facedect = facedetector_onnx.FaceDetector()
    global faces
    global new_faces
    global boxes

    img_with_boxes, faces, boxes =facedect(image)
    new_faces=[None]*(len(faces))
    """for box in boxes:
        print(box)
        print(box[2]-box[0])
        print(box[3]-box[1])"""
    return img_with_boxes, faces

def Paste_Faces(image):
    global new_faces
    global boxes
    from PIL import Image

    for i in range(boxes.shape[0]):
        box=boxes[i]
        side_size=box[2]-box[0]
        restored_face = new_faces[i]
        if restored_face != None:
            #print(f"Pasting area:{i}")
            restored_face =restored_face.resize([side_size, side_size], Image.Resampling.LANCZOS)
            img_with_boxes= facedect.paste_face(image,boxes[i],restored_face)
    return img_with_boxes, faces

def generate_click(prompt_t0,neg_prompt_in,seed_in,image_t0,steps,checkbox2,checkbox3):
    #width=256
    #height=width
    seed=""
    if isinstance(image_t0, list) or isinstance(image_t0, dict):
        result= image_t0["image"]
        mask= image_t0["mask"].convert("RGB")
    else:
        from PIL import Image
        print("Mascara vacia")
        result= image_t0
        mask= Image.new("RGB", (result.size), (0, 0, 0))
    
    if checkbox2:
        result,seed=generate_click_inpaint(prompt_t0,neg_prompt_in,seed_in,result,mask,steps)
        #def _process_mask(input_image,new_face,input_mask):
        result=_process_mask(image_t0["image"],result,mask)
    if checkbox3:
        result2=generate_click_gfpgan(result)
        result2=result2.resize(result.size)
        mask=mask.resize(result.size)
        result=_process_mask(result,result2,mask)


    global index
    global new_faces
    #result =result.resize([width,height], Image.Resampling.LANCZOS)
    new_faces[index]=result
    return result,seed


def generate_click1(prompt_t0,image_t0,steps):
    import numpy as np
    import cv2
    import PIL
    codeformer=codeformer_onnx.CodeFormer()
    image=image_t0["image"]

    new_face = codeformer(image)

    global index
    global new_faces
    new_faces[index]=new_face
    return new_face


def generate_click_gfpgan(image_t0, full=False):
    if full:
        GFPGAN=codeformer_onnx.GFPGANFaceAugment("./Scripts/anime-realesrgan-x4-default.onnx")
    else:
        #GFPGAN=codeformer_onnx.GFPGANFaceAugment("./Scripts/GFPGANv1.4.onnx")    
        GFPGAN=codeformer_onnx.FaceCorrection(mode='dml',style="GPEN")
        #GFPGAN=codeformer_onnx.GFPGANFaceAugment("./Scripts/codeformer.onnx")

    #image=image_t0["image"]
    image=image_t0
    if full:
        new_face = GFPGAN.forward(image)
    else:
        new_face= GFPGAN(image)

        
    return new_face




def generate_click_inpaint(prompt_t0,neg_prompt_in,seed_in,image_t0,mask_t0,steps):
    from Engine.pipelines_engines import inpaint_pipe
    from Engine.General_parameters import running_config
    from Engine.General_parameters import UI_Configuration

    Running_information= running_config().Running_information
    Running_information.update({"Running":True})
    model_drop="MergedModel-fp16-inpainting"
    #model_drop="stable-diffusion-onnx-v2-inpainting"
    legacy_t0 = False
    neg_prompt_t0 = neg_prompt_in
    #neg_prompt_t0 = 
    iter_t0=1
    steps_t0=steps
    guid_t0=8
    #height_t0=256
    #width_t0=256
    height_t0=512
    width_t0=512    
    batch_t0=1
    eta_t0 =0
    seed_t0 = int(seed_in) if seed_in!="" else None
    sch_t0="DDIM"
    input_image = image_t0

    from PIL import Image
    input_mask= mask_t0.resize((height_t0,width_t0) )   
    input_mask = mask_t0.convert("RGB")

    print(f"The model configurated for restoring faces is:{model_drop},i'm sure the name will be different for you. When you got an inpainting model you need to change this to your one, line 322 of ui_image_tools.py inside UI folder")

    if (Running_information["model"] != model_drop or Running_information["tab"] != "inpaint"):
        from UI import UI_common_funcs as UI_common
        UI_common.clean_memory_click()        
        Running_information.update({"model":model_drop})
        Running_information.update({"tab":"inpaint"})

    model_path=ui_config=UI_Configuration().models_dir+"\\"+model_drop
    inpaint_pipe().initialize(model_path,sch_t0,legacy_t0)

    inpaint_pipe().create_seeds(seed_in,iter_t0,False)

    for seed in inpaint_pipe().seeds:
        if running_config().Running_information["cancelled"]:
            running_config().Running_information.update({"cancelled":False})
            break

        print(f"Inpainting face")
        batch_images, _ = inpaint_pipe().run_inference(
            prompt_t0,
            neg_prompt_t0,
            input_image,
            input_mask,
            height_t0,
            width_t0,
            steps_t0,
            guid_t0,
            eta_t0,
            batch_t0,
            seed,
            legacy_t0)

    Running_information.update({"Running":False})

    new_face=batch_images[0]

    new_face=_process_mask(input_image,new_face,input_mask)
    """global index
    global new_faces
    new_faces[index]=new_face
    generate_click2("",new_face,1)"""
    return new_face,inpaint_pipe().seeds[0]

def _process_mask(input_image,new_face,input_mask):
    import numpy as np
    import cv2
    import PIL

    old_face = np.array(input_image.convert('RGB')).copy()
    old_face = cv2.cvtColor(old_face, cv2.COLOR_RGB2BGR)

    new_face = np.array(new_face.convert('RGB'))
    new_face = cv2.cvtColor(new_face, cv2.COLOR_RGB2BGR)
    input_mask = np.asarray(input_mask.convert('RGB'))
    input_mask = cv2.cvtColor(input_mask, cv2.COLOR_RGB2GRAY)
    _,input_mask = cv2.threshold(input_mask, 120, 255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_or(new_face,new_face,mask=input_mask)
    input_mask= cv2.bitwise_not(input_mask)
    background = cv2.bitwise_or(old_face,old_face,mask=input_mask)
    new_face = cv2.add(background,foreground)

    new_face = cv2.normalize(new_face, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    new_face=cv2.cvtColor(new_face, cv2.COLOR_BGR2RGB)
    new_face = PIL.Image.fromarray((new_face*255).astype(np.uint8))

    return new_face


def unload_facedect():
    from UI import UI_common_funcs as UI_common
    global facedect
    facedect= None
    UI_common.clean_memory_click()
    gc.collect
    print("Face detection unload")
    return 

def ResizeAndJoin(row,col):
    for files in sorted(os.listdir("./Temp")):
        if "TemporalImage" in files:
            SPR.superresolution_process("./Temp/"+files)
    tiles=image_slicer.open_images_in("./Temp")
    img=image_slicer.join(tiles,row,col)
    return img

def Clean_SPR_Mem():
    SPR.clean_superresolution_memory()
    gc.collect()

def DeleteTemp():
    image_slicer.delete_temp_dir()

def adjustate_column_marks(img,image,img_cols,img_rows,pixels):
    halfof_tile_width_size=(image.size[0]/img_cols)/2
    left=halfof_tile_width_size
    right=image.size[0]-halfof_tile_width_size
    img_columnmarks_correction=image.crop((left,0,right,image.size[1]))
    image_slicer.slice(img_columnmarks_correction,row=img_rows,col=img_cols-1)
    img_columnmarks_correction= None
    for files in sorted(os.listdir("./Temp")):
        if "TemporalImage" in files:
            SPR.superresolution_process("./Temp/"+files)
            image_slicer.create_slice_of_substitute_for_column_joint("./Temp/"+files,pixels)
    tiles=image_slicer.open_images_in("./Temp")
    img=image_slicer.substitute_image_joint_vertical_marks(img,tiles,pixels,img_cols)
    return img

    

def adjustate_row_marks(img,image,img_cols,img_rows,pixels):
    halfof_tile_height_size=(image.size[1]/img_rows)/2
    left = 0
    right = image.size[0]
    top = halfof_tile_height_size
    down = image.size[1]-halfof_tile_height_size
    img_rowmarks_correction=image.crop((left,top,right,down))
    image_slicer.slice(img_rowmarks_correction,row=img_rows-1,col=img_cols)

    for files in sorted(os.listdir("./Temp")):
        if "TemporalImage" in files:
            SPR.superresolution_process("./Temp/"+files)
            image_slicer.create_slice_of_substitute_for_row_joint("./Temp/"+files,pixels)

    tiles=image_slicer.open_images_in("./Temp")
    img=image_slicer.substitute_image_joint_horizontal_marks(img,tiles,pixels,img_rows)
    return img

def Resize_Image(image,img_rows,img_cols,checkbox1):
    pixels=8  #Manual Adjustment
    #Adjust img size to col&rows multiples, and make sure both tile sizes are divisible by 2
    image=image_slicer.adjust_image_size(image,img_rows,img_cols)

    #tiles=image_slicer.slice(image,row=img_rows,col=img_cols)
    #tiles=None  #Modify previous to save memory if not used?
    image_slicer.slice(image,row=img_rows,col=img_cols)
    img=ResizeAndJoin(row=img_rows,col=img_cols)

    DeleteTemp()

    Mark_Adjustement = checkbox1
    if Mark_Adjustement:
        image=adjustate_column_marks(img,image,img_cols,img_rows,pixels)
        DeleteTemp()
        adjustate_row_marks(img,image,img_cols,img_rows,pixels)
        DeleteTemp()
        img=blur_Image(img)
        img=Sharpen_Image(img) #Create option for them, they delay too much the creation

    return img


def Resize_Image2(image,resolution_prompt):
    import requests
    from PIL import Image
    from io import BytesIO
    #from diffusers import StableDiffusionUpscalePipeline
    from diffusers import OnnxStableDiffusionUpscalePipeline

    import torch
    import Engine.pipelines_engines as Engines
    model_path="C:\\AMD_ML\\models\\stable-diffusion-x4-upscaler-onnx"
    #VAE= Engines.Vae_and_Text_Encoders().load_vaedecoder(model_path)
    #textenc= Engines.Vae_and_Text_Encoders().load_textencoder(model_path)
    #scheduler=Engines.SchedulersConfig().scheduler("DPMS_Heun",model_path)
    #low_res_scheduler= Engines.SchedulersConfig().low_res_scheduler()
    #pipeline =OnnxStableDiffusionUpscalePipeline.from_pretrained("C:\\AMD_ML\\models\\stable-diffusion-x4-upscaler-onnx",low_res_scheduler= low_res_scheduler,scheduler=scheduler, vae=VAE,text_encoder=textenc,provider="DmlExecutionProvider")  #sess_options=sess_options 
    pipeline =OnnxStableDiffusionUpscalePipeline.from_pretrained("C:\\AMD_ML\\models\\stable-diffusion-x4-upscaler-onnx", provider="DmlExecutionProvider")
    
    #model_id = "stabilityai/stable-diffusion-x4-upscaler"
    #pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, revision="fp16")
    #pipeline = pipeline.to("cpu")

    low_res_img = image
    low_res_img = low_res_img.resize((192, 192))
    prompt = resolution_prompt

    upscaled_image = pipeline(prompt=prompt, image=low_res_img,num_inference_steps=4).images[0]


    return upscaled_image
    #return image


def blur_Image(img):
    import cv2 as cv
    import numpy as np
    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    blur = cv.GaussianBlur(img,(5,5),0)
    #blur = cv.medianBlur(img,5)
    #blur = cv.bilateralFilter(img,9,75,75)
    #blur=cv.blur(img,(5,5))
    return blur

def Sharpen_Image(img):
    import numpy as np
    import cv2 as cv
    img = np.array(img) 
    if is_grayscale(img):
        height, width = img.shape
    else:
        img = cv.cvtColor(img, cv.CV_8U)
        height, width, n_channels = img.shape

    result = np.zeros(img.shape, img.dtype)
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            if is_grayscale(img):
                sum_value = 5 * img[j, i] - img[j + 1, i] - img[j - 1, i] \
                            - img[j, i + 1] - img[j, i - 1]
                result[j, i] = saturated(sum_value)
            else:
                for k in range(0, n_channels):
                    sum_value = 5 * img[j, i, k] - img[j + 1, i, k]  \
                                - img[j - 1, i, k] - img[j, i + 1, k]\
                                - img[j, i - 1, k]
                    result[j, i, k] = saturated(sum_value)
    
    return result


def is_grayscale(my_image):
    return len(my_image.shape) < 3

def saturated(sum_value):
    if sum_value > 255:
        sum_value = 255
    if sum_value < 0:
        sum_value = 0
    return sum_value




def clean_memory_click_borrar():
    print("Cleaning memory")
    pipelines_engines.Vae_and_Text_Encoders().unload_from_memory()
    pipelines_engines.txt2img_pipe().unload_from_memory()
    pipelines_engines.inpaint_pipe().unload_from_memory()
    pipelines_engines.instruct_p2p_pipe().unload_from_memory()
    pipelines_engines.img2img_pipe().unload_from_memory()
    pipelines_engines.ControlNet_pipe().unload_from_memory()
    gc.collect()