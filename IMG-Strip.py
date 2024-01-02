import gradio as gr
from pathlib import Path
from PIL import Image
#from DIS_ISNET.models import *


Metadata_value="Created by @"

def init_ui():
    downloads_path = str(Path.home() / "Downloads")
    with gr.Blocks(title="Additional Tools") as img_strip:
        with gr.Tab(label="Remove & Extract Metadata") as Metadata_tab:
            image_in = gr.Image(label="input image", type="pil", elem_id="image_init")
            save_dir = gr.Textbox(label="Save to Dir",value=downloads_path, lines=1)
            old_metadata = gr.Textbox(label="Existing metadata",value="", lines=4)
            new_metadata = gr.Textbox(label="Metadata to overwrite",value=Metadata_value, lines=4)
            threshold = gr.Slider(0, 255, value=20, step=1, label="Mask threshold", interactive=True)
            extract_metadata_btn = gr.Button("Extract Metadata", variant="primary")
            process_btn = gr.Button("Process Metadata & Save Image", variant="primary")
            convert_btn = gr.Button("Extract mask of main objects")
            image_out = gr.Image(label="Object")
            image_out2 = gr.Image(label="Detected Mask")       


        process_btn.click(fn=delete_metadata_saveimg, inputs=[image_in,save_dir,new_metadata] , outputs=None)
        extract_metadata_btn.click(fn=extract_metadata, inputs=image_in , outputs=old_metadata)
        convert_btn.click(fn=test_fn, inputs=[image_in,threshold] , outputs=[image_out2,image_out])
    return img_strip
	


def test_fn(image,threshold):
    #import torch.onnx
    #import torch
    #import torch.nn.functional as F
    #from torchvision.transforms.functional import normalize

    import numpy as np
    import cv2
    import onnxruntime    


    input_size=[1024,1024]  #size of model input width X height

    input_image = np.array(image.convert('RGB'))
    #input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    input_image_resized_np=cv2.resize(input_image,input_size)

    input_shape=input_image.shape[0:2]   #size of input image height X width 

    if len(input_image_resized_np.shape) < 3:
        input_image_resized_np = input_image_resized_np[:, :, np.newaxis]

    input_image_resized_np=input_image_resized_np.transpose(2,0,1)#.astype(np.uint8)
    input_image_resized_np=np.expand_dims(input_image_resized_np, 0)
    input_image_resized_np=input_image_resized_np/255.0

    min_val = np.min(input_image_resized_np)
    max_val = np.max(input_image_resized_np)
    input_image_resized_np = (input_image_resized_np - min_val) / (max_val - min_val) #Normalization


    #session = onnxruntime.InferenceSession("model_isnet.onnx_fp16.onnx",providers=['DmlExecutionProvider', 'CPUExecutionProvider'])        
    #results = session.run(["output"], {"image": input_image_resized_np.astype(np.float16})
    session = onnxruntime.InferenceSession("./modules/dis-isnet/model_isnet.onnx",providers=['CUDAExecutionProvider','DmlExecutionProvider', 'CPUExecutionProvider'])
    results = session.run(["output"], {"image": input_image_resized_np.astype(np.float32)})

    result=results[0]
    #result=torch.from_numpy(result)
    #result=torch.squeeze(F.upsample(result,input_shape,mode='bilinear'),0)  #from pth inference converted to numpy

    result=np.squeeze(result)
    max=result.max()
    min=result.min()
    result =(result-min)/(max-min)      #Normalization
    result = (result *255).astype(np.uint8)


    result_3channels=cv2.merge([result,result,result]) #create 3 duplicated channels for each color to be considered as image
    result_3channels=cv2.resize(result_3channels,(input_shape[1],input_shape[0]))
    result,_,_=cv2.split(result_3channels)
    
    result=np.expand_dims(result, axis=2)


    #result =result.transpose(1,2,0).astype(np.uint8)
    _,result = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY)  #mirar el threshold, esta en 10, pero inicial en 120
    #final=cv2.merge([input_image,result])
    result1=cv2.bitwise_or(input_image,input_image,mask=result) ##pensar si dejar fondo(input_image) o borrarlo (result1)
    #result=cv2.merge([input_image,result]) 
    result=cv2.merge([result1,result])
    #result=cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    image_applied_mask=Image.fromarray(result)

    return result_3channels,image_applied_mask

    #return result_3channels,final


def test_fn2(image):   #pruebas de cambio de pth a onnx
    
    import torch.onnx
    import torch
    import numpy as np
    import cv2
    #import torch.nn as nn
    #from torchvision import models
    import torch.nn.functional as F
    from torchvision.transforms.functional import normalize

    input_size=[1024,1024]


    model_path = "./DIS_ISNET/isnet.pth"
    #model=ISNetDIS()
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    #print("Dict")
    #print(model)
    new_image = np.array(image.convert('RGB'))
    im=new_image
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

    #im= np.array(image).copy()  #convert to skimage opposite(to PIL) : Image.fromarray(image)
    
    with torch.no_grad():
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_shp=im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
        im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
        image = torch.divide(im_tensor,255.0)
        image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

        if torch.cuda.is_available():
            image=image.cuda()

        import onnxruntime
        session = onnxruntime.InferenceSession("model_isnet.onnx",providers=['CUDAExecutionProvider','DmlExecutionProvider', 'CPUExecutionProvider'])
        results = session.run(["output"], {"image": image.numpy()})

        result=results[0]
        result=torch.from_numpy(result)
        #print(type(result))  
        #print(result.shape)
   
        #result=model(image)
        #print(type(result))  
        #print(type(result[0][0]))
        #print(result.shape)

        #result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
        result=torch.squeeze(F.upsample(result,im_shp,mode='bilinear'),0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result-mi)/(ma-mi)
        result=(result*255)
        
        
        result=result.permute(1,2,0).cpu().data.numpy().astype(np.uint8)
        result2=cv2.merge([result,result,result])
        result = cv2.cvtColor(result2, cv2.COLOR_RGB2GRAY)
        _,result = cv2.threshold(result, 120, 255, cv2.THRESH_BINARY)
        #print(result.shape)

        """model.eval()
        x = torch.randn(1, 3, 1024, 1024, requires_grad=True)
        x.to('cpu')
        #input_names = ["batch,channels,height,width"]"input_ids"
        #input_names = ['b','c','height','width']
        input_names = ['image']
        output_names = ["output0"]
        #torch_out = model(x)
        #torch.onnx.export(model, x, "model_isnet.onnx", export_params=True, opset_version=11, do_constant_folding=True, input_names = ['image'], output_names = ['output'], dynamic_axes = {'image' : {2: 'height',3:'width'}, 'output': {2: 'height',3:'width'}})
        #torch.onnx.export(model, x, "model_isnet.onnx", export_params=True, opset_version=11, do_constant_folding=True, input_names = ['input'], output_names = ['output'])    
        torch.onnx.export(model, x, "model_isnet.onnx", export_params=True, opset_version=11, do_constant_folding=True, input_names = ['image'], output_names = ['output'])"""
        
        

    result=cv2.bitwise_or(new_image,new_image,mask=result)
    result=cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    image2=Image.fromarray(result)
    """

    import onnxruntime
    session = onnxruntime.InferenceSession("model_isnet.onnx",providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
    session.get_modelmeta()
    first_input_name = session.get_inputs()[0].name
    first_output_name = session.get_outputs()[0].name
    print("Modelo onnx")
    print(first_input_name)
    print(first_output_name)


    print("The model expects input shape: ", session.get_inputs()[0].shape)
    print("The shape of the Image is: ", mantener.shape)    
    results = session.run(["output"], {"image": mantener.numpy()})

    result=results[0]
    print(type(result))"""
    """import onnx
    from onnxconverter_common import float16
    model = onnx.load("model_isnet.onnx")
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, "model_isnet.onnx_fp16.onnx")"""

    return result2,image2


def extract_metadata(image):
    #metadata=list(image.info.values())
    metadata=image.info
    return metadata


def delete_metadata_saveimg(image,save_dir, new_metadata=""):
    from PIL import PngImagePlugin
    #exifdata = eval(image.info['parameters'])
    #exifdata = (image.info['parameters'])
    exifdata =str(image.info)
    import hashlib
    new_name= hashlib.sha256(exifdata.encode()).hexdigest()[0:10]

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("ImageInfo",f"{new_metadata}")

    image.save(f"{save_dir}/{new_name}.png",optimize=True,pnginfo=metadata)


img_strip =init_ui()
img_strip.launch(server_port=8080)


