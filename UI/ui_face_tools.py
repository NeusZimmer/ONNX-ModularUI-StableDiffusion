import gradio as gr
import gc,os
import numpy as np
from Engine import pipelines_engines

global facedect
global faces
global full_faces_info
global new_faces
global boxes
global index
global analyser    
analyser=None
facedect = None
faces = []
new_faces = []
boxes = []
index=0



def show_input_image_area():
    with gr.Row():
        with gr.Column(variant="compact"):
            image_in = gr.Image(label="Input image", type="pil", elem_id="image_init")
        with gr.Column(variant="compact"):
            gallery_extracted_faces = gr.Gallery(label="Extracted Faces")
    facedect_btn = gr.Button("Extract faces information")
    select_face_btn = gr.Button("Select face")


    with gr.Row():
        with gr.Column(variant="compact"):
            selected_face_in = gr.Image(label="Original Face", type="pil", elem_id="imagen2")
        with gr.Column(variant="compact"):
            with gr.Row(): 
                face_in = gr.Image(label="Reference Face", type="pil",elem_id="imagen2") 
    with gr.Row(): 
        process_face_btn = gr.Button("Process Selected face",variant="primary")
        threshold = gr.Slider(1, 255, value=20, label="Threshold", info="Strength of pasted image(inversed), Increase until the pasted borders are not shown")  

            
        
    with gr.Row():
        selected_face_index = gr.Textbox(label="selected_face_index",value="", lines=1,visible=False, interactive=False)
        selected_face_gender = gr.Textbox(label="selected_face_gender",value="", lines=1,visible=False, interactive=True)   
        selected_face_age = gr.Textbox(label="selected_face_age",value="", lines=1,visible=False, interactive=True) 
        selected_face_box = gr.Textbox(label="selected_face_box",value="", lines=1,visible=False, interactive=False) 
    with gr.Row():
        new_face_info = gr.Textbox(label="New Face info",value="", lines=4,visible=False) 
    with gr.Row():
        update_face_btn = gr.Button("Update Gender & Age of Selected face",visible=False)  
    with gr.Row():
        new_face_out = gr.Image(label="Swapped Face& Create Mask",tool="sketch", type="pil", interactive=True,elem_id="imagen1")
    with gr.Row():
        re_process_face_btn = gr.Button("Apply mask to processed face",variant="primary")
    with gr.Row():
        new_face_out2 = gr.Image(label="Swapped Face", type="pil", interactive=True,elem_id="imagen1")          
    with gr.Row():
        paste_face_btn = gr.Button("Paste back new face",visible=False)  
        unload_btn = gr.Button("Unload") 
    
    facedect_btn.click(fn=face_detection, inputs=image_in, outputs=gallery_extracted_faces)
    select_face_btn.click(fn=select_face, inputs=None, outputs=[selected_face_in,selected_face_box,selected_face_index,selected_face_gender,selected_face_age])
    process_face_btn.click(fn=swap_face, inputs=[image_in,selected_face_index,face_in,threshold], outputs=new_face_out)
    re_process_face_btn.click(fn=re_swap_face, inputs=[image_in,selected_face_index,face_in,threshold,new_face_out], outputs=new_face_out2)
    paste_face_btn.click(fn=paste_back_face, inputs=None)
    gallery_extracted_faces.select(fn=selected_index,inputs=None, outputs=None)
    unload_btn.click(fn=unload,inputs=None, outputs=None)
    update_face_btn.click(fn=update_face,inputs=[selected_face_index,selected_face_gender,selected_face_age], outputs=None)





def update_face(index,gender,age):
    print("Update face") 
    index=int(index)
    gender= int(gender)
    age= int(age)
    gender= 0 if gender==0 else 1
    global full_faces_info
    full_faces_info[index]['age']=age
    full_faces_info[index]['gender']=gender
    return 

def unload():
    global analyser
    analyser=None

    
def face_detection(image):
    print("face_detection")    
    global faces
    global analyser
    global full_faces_info
    from Scripts.faceanalyser import faceanalyser
    if analyser==None:
        analyser=faceanalyser.face_analyser()

    full_faces_info=analyser.get(image)
    #print(full_faces_info[0].keys())
    faces=[]
    image=np.asarray(image)
    from PIL import Image
    for face in full_faces_info:
        box=face['bbox']
        #print(box)
        extracted_face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        faces.append(Image.fromarray(extracted_face))

    #image=Image.fromarray(image)
    #return image,faces
    return faces




def select_face():
    print("select_face")
    global faces
    global boxes
    global full_faces_info
    return faces[index],full_faces_info[index],index,full_faces_info[index]['gender'],full_faces_info[index]['age']

def re_swap_face(*args):  #[image_in,selected_face_index,face_in,new_face_out]
    print("swap_face")
    import insightface
    global full_faces_info
    temp_frame=args[0]
    target_face=full_faces_info[int(args[1])]
    source_face=args[2]
    threshold=args[3]
    mask=args[4]["mask"].convert("RGB")

    global analyser
    source_face=analyser.get(source_face)[0]

    model_path = './Scripts/facedetector/inswapper_128_fp16.onnx'
    #face_processor = insightface.model_zoo.get_model(model_path, providers = ['DmlExecutionProvider'])
    face_processor = insightface.model_zoo.get_model(model_path, providers = ['CPUExecutionProvider'])
    #imagen=face_processor.get(np.asarray(temp_frame), target_face, source_face, paste_back = False)
    imagen=get_face(np.asarray(temp_frame), target_face, source_face,face_processor, paste_back = True,threshold=threshold,mask=np.asarray(mask),apply_mask=True)

    return imagen

def swap_face(*args):  #[image_in,selected_face_index,face_in]
    print("swap_face")
    import insightface
    global full_faces_info
    temp_frame=args[0]
    target_face=full_faces_info[int(args[1])]
    source_face=args[2]
    threshold=args[3]
    global analyser
    source_face=analyser.get(source_face)[0]

    model_path = './Scripts/facedetector/inswapper_128_fp16.onnx'
    #face_processor = insightface.model_zoo.get_model(model_path, providers = ['DmlExecutionProvider'])
    face_processor = insightface.model_zoo.get_model(model_path, providers = ['CPUExecutionProvider'])
    #imagen=face_processor.get(np.asarray(temp_frame), target_face, source_face, paste_back = False)
    imagen=get_face(np.asarray(temp_frame), target_face, source_face,face_processor, paste_back = True,threshold=threshold)

    return imagen

def paste_back_face():
    print("paste_back_face")
    return

def selected_index(evt:gr.SelectData):
    #print("selected_index")
    global index
    index=evt.index
    return 



def get_face(img, target_face, source_face,face_processor, paste_back=True,threshold=20,mask=None,apply_mask=False):
    import cv2
    from insightface.utils import face_align

    #aimg, M = face_align.norm_crop2(img, target_face.kps, face_processor.input_size[0])
    aimg_init, M = face_align.norm_crop2(img, target_face.kps, 672)
    aimg=cv2.resize(aimg_init,[128, 128], interpolation = cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / face_processor.input_std, face_processor.input_size,
                                    (face_processor.input_mean, face_processor.input_mean, face_processor.input_mean), swapRB=True)
    latent = source_face.normed_embedding.reshape((1,-1))
    latent = np.dot(latent, face_processor.emap)
    latent /= np.linalg.norm(latent)
    pred = face_processor.session.run(face_processor.output_names, {face_processor.input_names[0]: blob, face_processor.input_names[1]: latent})[0]
    img_fake = pred.transpose((0,2,3,1))[0]
    bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
    if not paste_back:
        return bgr_fake, M
    else:
        from Scripts import superresolution as SPR
        from PIL import Image
        aimg=aimg_init
        bgr_fake_PIL=SPR.superresolution_process_pil(Image.fromarray(bgr_fake))
        bgr_fake= np.array(bgr_fake_PIL)

        target_img = img
        fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)
        fake_diff[:2,:] = 0
        fake_diff[-2:,:] = 0
        fake_diff[:,:2] = 0
        fake_diff[:,-2:] = 0
        IM = cv2.invertAffineTransform(M)
        img_white = np.full((aimg.shape[0],aimg.shape[1]), 255, dtype=np.float32)
        bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        img_white[img_white>20] = 255
        #fthresh = 10
        fthresh = threshold     

        fake_diff[fake_diff<fthresh] = 0
        fake_diff[fake_diff>=fthresh] = 255
        img_mask = img_white
        mask_h_inds, mask_w_inds = np.where(img_mask==255)
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h*mask_w))
        k = max(mask_size//10, 10)
        #k = max(mask_size//20, 6)
        #k = 6
        cv2.imwrite("fake_diff.png", fake_diff)         
        cv2.imwrite("img_mask.png", img_mask)
        if apply_mask:
            cv2.imwrite("mask.png", mask)

            #mask= np.asarray(mask.convert('RGB'))
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            #img_mask=img_mask - mask
            fake_diff=fake_diff - mask
            fake_diff[fake_diff<fthresh] = 0
            fake_diff[fake_diff>=fthresh] = 255            

        cv2.imwrite("fake_diff.png", fake_diff) 
        kernel = np.ones((k,k),np.uint8)
        img_mask = cv2.erode(img_mask,kernel,iterations = 1)
        kernel = np.ones((2,2),np.uint8)
        fake_diff = cv2.dilate(fake_diff,kernel,iterations = 1)
        k = max(mask_size//20, 5)
        #k = 3
        #k = 3
        kernel_size = (k, k)
        blur_size = tuple(2*i+1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        k = 5
        kernel_size = (k, k)
        blur_size = tuple(2*i+1 for i in kernel_size)
        fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
        img_mask /= 255
        fake_diff /= 255
        img_mask = fake_diff #uncommented from original file
        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
        fake_merged = img_mask * bgr_fake + (1-img_mask) * target_img.astype(np.float32)
        fake_merged = fake_merged.astype(np.uint8)
        return fake_merged


