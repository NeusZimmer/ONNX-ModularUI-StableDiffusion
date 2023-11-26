import gradio as gr
import os,shutil
import torch
import onnx

from pathlib import Path
from onnxruntime.transformers.float16 import convert_float_to_float16

global loaded_model
loaded_model=None

global show_debug
show_debug=True


def init_ui():
    with gr.Blocks(title="Stable Difussion ONNX Model Converter UI",css= css1) as converter:
        models=_get_list_of_models(os.getcwd())
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("Select model area to convert")
                Convert_TextEnc=gr.Checkbox(label="Convert Text Encoder", value=False, interactive=True)                          
                Convert_Tokenizer=gr.Checkbox(label="Convert Tokenizer", value=False, interactive=True)             
                Convert_Scheduler=gr.Checkbox(label="Convert Scheduler", value=False, interactive=True)
                Convert_VaeEnc=gr.Checkbox(label="Convert VAE Encoder", value=False, interactive=True)
                Convert_VaeDec=gr.Checkbox(label="Convert VAE Decoder", value=False, interactive=True)
                Convert_UNET=gr.Checkbox(label="Convert UNET Model", value=False, interactive=True) 
                Convert_ControlNet=gr.Checkbox(label="Convert as ControlNet UNET Model", value=False, interactive=False,visible=False)#still dont implemented
                Generate_Config=gr.Checkbox(label="Generate Config File", value=False, interactive=True)
                gr.Markdown("Additional")
                make_fp16=gr.Checkbox(label="Convert to FP16", value=True, interactive=True)
                slicing=gr.Radio(["auto","max"],value="auto",label="Attention Slicing")
                optimize=gr.Checkbox(label="Optimize", value=False, interactive=True)
                device=gr.Radio(["cuda","cpu"],value="cpu",label="Device")

            with gr.Column(scale=5):
                directory_in = gr.Textbox(value=os.getcwd(), lines=1, label="List models from", interactive=True)
                gr.Markdown("Model Selection (safetensors or ckpt)")
                model_drop = gr.Dropdown(models,value=models[0], label="Model Selection", interactive=True)
                reload_availablemodels_btn = gr.Button("Reload list of models")
                with gr.Row():
                    load_model_btn = gr.Button("Load model in memory")
                    unload_model_btn = gr.Button("UnLoad from memory")
                process_model_btn = gr.Button("Convert Select Model areas",variant="primary")                    
                directory_out = gr.Textbox(value=os.getcwd(), lines=1, label="Save Model to", interactive=True)
                
                with gr.Accordion("Additional info & fails with optimization",open=False):
                    markdown_text="""Sometimes, due to out of memory errors,if optimize is selected the conversion wil stop prior finalizing. \nIf that happens, you will still have the extracted files 
                                in the unet subdir,\n Do not panic, you might still continue, go to below, open the area for Postprocessing and introduce the full path (with .onnx)
                                  and optimize or just convert those files to as fp16 (saved into a unet2 directory, delete previous one and change to unet if you want to run the model as fp16),"""
                    gr.Markdown("NOTE: there is no automated load & convert: load a model select your config and then convert it.")
                    gr.Markdown(markdown_text)
            with gr.Column(scale=5):
                informacion = gr.Textbox(value="", lines=12, label="Output Information", interactive=False)
        ################    FOOTER ACTIONS  #####################                
        with gr.Accordion("Processes prior conversion",open=False):
            with gr.Row():            
                new_vae = gr.Textbox(value="", lines=1, label="Change to vae (full path with extension)", interactive=True)
                change_vae_btn = gr.Button("Change Model's vae to this vae")
            with gr.Row():            
                clip_skip = gr.Radio(["2","3","4"],value="1",label="Clip Skip")
                clipskip_btn = gr.Button("Apply clip-skip to text_encoder")
            with gr.Row():            
                save_memory_path = gr.Textbox(value=os.getcwd(), lines=1, label="Path", interactive=True)
                save_memory_btn = gr.Button("Save in memory model to disk")                
            gr.Markdown("Pending: adding LORAS and text inversion prior conversion")
        with gr.Accordion("Processes post conversion",open=False):
            with gr.Row():            
                onnx_path = gr.Textbox(value="", lines=1, label="Optimize model", interactive=True)
                optimize_btn = gr.Button("Optimize model")
                save_asfp16_btn = gr.Button("Save as fp16")


        all_inputs=[model_drop,Convert_TextEnc,Convert_Tokenizer,Convert_Scheduler,Convert_VaeEnc,Convert_VaeDec,Convert_UNET,Convert_ControlNet,Generate_Config,make_fp16,slicing,optimize,device,directory_in,directory_out,informacion]

    ################BUTTON ACTIONS#####################
        reload_availablemodels_btn.click(fn=get_list_of_models,inputs=directory_in,outputs=model_drop)
        load_model_btn.click(fn=_loadmodel,inputs=[model_drop,device],outputs=informacion)
        process_model_btn.click(fn=process_model,inputs=all_inputs,outputs=informacion)
        unload_model_btn.click(fn=unload,inputs=informacion,outputs=informacion)
        save_memory_btn.click(fn=save_memory_todisk, inputs=[save_memory_path,informacion], outputs=informacion)

        #         FOOTER ACTIONS  # 
        change_vae_btn.click(fn=load_vae,inputs=[new_vae,informacion],outputs=informacion)
        clipskip_btn.click(fn=process_clipskip, inputs=[clip_skip,device,informacion], outputs=[informacion,clipskip_btn])
        optimize_btn.click(fn=optimize_onnx_model, inputs=[onnx_path,informacion],outputs=informacion)
        save_asfp16_btn.click(fn=convert_to_fp16_2, inputs=onnx_path,outputs=informacion)
    ################END OF BUTTON ACTIONS#####################



    return converter

css1 = """
#title1 {background-color: #00ACAA;text-transform: uppercase; font-weight: bold;}
.feedback textarea {font-size: 24px !important}
#imagen1 {min-height: 600px}
#imagen2 {height: 250px}
#imagen3 {min-width: 512px,min-height: 512px}
"""

css2 = """
#title1 {background-color: #00ACAA;text-transform: uppercase; font-weight: bold;}
.feedback textarea {font-size: 24px !important}
"""


def optimize_onnx_model(model_path,informacion):
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.optimizer import optimize_model

    if not (model_path.endswith(".onnx") and os.path.isfile(model_path)):
        return informacion+f"\nNo valid input onnx model:{model_path}"
    else:
        informacion=informacion+f"\nValid input onnx model:{model_path}"

    model_dir=os.path.dirname(model_path)
    unet_path=Path(model_path)
    unet_model_path = str(unet_path.absolute().as_posix())
    unet_dir = os.path.dirname(unet_model_path)



    # First we set our optimisation to the ORT Optimizer defaults for unet
    optimization_options = FusionOptions("unet")
    # The ORT optimizer is designed for ORT GPU and CUDA
    # To make things work with ORT DirectML, we disable some options
    # On by default in ORT optimizer, turned off because it has no effect
    optimization_options.enable_qordered_matmul = False
    optimization_options.enable_nhwc_conv = False # On by default in ORT optimizer, turned off as it causes performance issues

    optimizer = optimize_model(
        #input = unet_model_path,
        input =unet_path,
        model_type = "unet",
        #opt_level = 0,
        opt_level = 1,
        optimization_options = optimization_options,
        use_gpu = True,
        only_onnxruntime = True
        #use_gpu = False,
        #only_onnxruntime = False
    )
    #if fp16:
    if True:        
        optimizer.convert_float_to_float16(
            keep_io_types=True, disable_shape_infer=True, op_block_list=['RandomNormalLike']
        )
    optimizer.topological_sort()
    unet=optimizer.model
    del optimizer

    print("Optimizer finalized, trying to save data in the same directory")

    # clean up existing tensor files , remove comments to delete previous files
    shutil.rmtree(unet_dir)
    os.mkdir(unet_dir)

    # collate external tensor files into one
    onnx.save_model(
        unet,
        unet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    return informacion

def _get_list_of_models(path:str)->list:
    import glob
    if path[-1]== "\\" or path[-1]=="/": path=path[0:-1]
    safetensors=glob.glob(f"{path}\\*.safetensors")
    ckpt=glob.glob(f"{path}\\*.ckpt")
    models=safetensors+ckpt
    if len(models)==0: models=["None"]
    return models

def get_list_of_models(path:str):
    models=_get_list_of_models(path)
    return gr.Dropdown.update(choices=models,value=models[0])

def _loadmodel(model,device):
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
    import torch
    show_debug=False
    area="Loading model"
    device= device if torch.cuda.is_available() else "cpu"
    safetensors=True if model.endswith(".safetensors") else False
    debug([model,device,safetensors],area,show_debug)

    global loaded_model
    loaded_model =download_from_original_stable_diffusion_ckpt(
            checkpoint_path_or_dict=model,
            device=device,
            load_safety_checker=False,
            from_safetensors= safetensors,
            local_files_only=True,
            )
    """
    checkpoint_path_or_dict: Union[str, Dict[str, torch.Tensor]],
    original_config_file: str = None,
    image_size: Optional[int] = None,
    prediction_type: str = None,
    model_type: str = None,
    extract_ema: bool = False,
    scheduler_type: str = "pndm",
    num_in_channels: Optional[int] = None,
    upcast_attention: Optional[bool] = None,
    device: str = None,
    from_safetensors: bool = False,
    stable_unclip: Optional[str] = None,
    stable_unclip_prior: Optional[str] = None,
    clip_stats_path: Optional[str] = None,
    controlnet: Optional[bool] = None,
    adapter: Optional[bool] = None,
    load_safety_checker: bool = True,
    pipeline_class: DiffusionPipeline = None,
    local_files_only=False,
    vae_path=None,
    vae=None,
    text_encoder=None,
    tokenizer=None,
    config_files=None,
    """

    #Atencion a esto, añadir a la UI:'load_lora_into_text_encoder', 'load_lora_into_unet ??
    debug(type(loaded_model),area,show_debug)
    debug(loaded_model.__dict__,area,show_debug)

    return ("Model loaded: %s" % model)

def unload(text):
    import gc
    global loaded_model
    loaded_model=None
    gc.collect()
    return text+"\nModel unloaded from memory"

def convert_params_todict(args):
    #just copy the line of list of all inputs and make it a string
    all_inputs="model_drop,Convert_TextEnc,Convert_Tokenizer,Convert_Scheduler,Convert_VaeEnc,Convert_VaeDec,Convert_UNET,Convert_ControlNet,Generate_Config,make_fp16,slicing,optimize,device,directory_in,directory_out,informacion".split(',')
    dictio={}
    for count, item in enumerate(all_inputs):
        dictio.update({item:args[count]}) 

    global show_debug
    debug(dictio,"Params conversion dict",show_debug)

    return dictio

def process_model(*args):
    opset=17 #"The version of the ONNX operator set to use.",

    from diffusers.models.attention_processor import AttnProcessor

    returned_text=""
    params=convert_params_todict(args)

    params["device"]= params["device"] if torch.cuda.is_available() else "cpu"
    global loaded_model
    loaded_model.unet.set_attn_processor(AttnProcessor())


    print(params['slicing'])
    if params['slicing']:
        if params['slicing'] == "max":
            print ("WARNING: attention_slicing max implies Optimization not available")
            #Esto ignorado, he pasado por max y hecho optimizacion, funciona? comprobar diferencia de rendimiento.
            #params['optimize']=False
        loaded_model.enable_attention_slicing(params['slicing'])


    model_name=params["model_drop"].split('\\')[-1]
    model_name=model_name.split('.')[0]
    text1= "max" if params["slicing"]=="max" else "auto"
    text2= "fp16" if params["make_fp16"]==True else "fp32"
    output_path=params["directory_out"]+"\\"+f"{model_name}-{text1}-{text2}"


    if params['Convert_Tokenizer']:
        #Must convert and save to disk in a subdirectory
        convert_tokenizer(loaded_model,output_path)
        returned_text=returned_text+f"\nTokenizer added"

    if params['Convert_Scheduler']:
        #Must convert and save to disk in a subdirectory
        convert_scheduler(loaded_model,output_path)
        returned_text=returned_text+f"\nScheduler added"        

    if params['Generate_Config']:
        #Must convert and save to disk in a subdirectory
        generate_model_config(loaded_model,output_path)
        returned_text=returned_text+f"\nConfig file generated"   

    if params['Convert_TextEnc']:
        #Must convert and save to disk in a subdirectory
        textenc_model_path=convert_text_encoder(loaded_model,output_path,opset,params["device"])
        if params["make_fp16"]:
            print("Converting fp16 simple")
            textenc_model_path=convert_to_fp16(textenc_model_path)
        returned_text=returned_text+f"\nText encoder model converted"

    if params['Convert_VaeEnc']:
        #Must convert and save to disk in a subdirectory
        vae_encoder_path=convert_vae_encoder(loaded_model,output_path,opset,params["device"])
        returned_text=returned_text+f"\nVae Encoded model converted"
    if params['Convert_VaeDec']:
        #Must convert and save to disk in a subdirectory
        vae_decoder_path=convert_vae_decoder(loaded_model,output_path,opset,params["device"])
        returned_text=returned_text+f"\nVae Decoder model converted"
    if params['Convert_UNET']:
        unet_model_path=extract_unet(loaded_model,output_path,opset,params["device"])
        print(unet_model_path)        
        if params['optimize']:            
            print("UNET OPTIMIZATION PROCESS")
            unet_model_path=convert_unet_optimized(unet_model_path,params["make_fp16"])
            returned_text=returned_text+f"\nUnet model optimization completed"
        else:
            #Must convert and save to disk in a subdirectory (created the config file , scheduler, and tokenizer?)
            if params["make_fp16"]:
                print("Converting fp16 simple")
                unet_model_path=convert_to_fp16(unet_model_path)
            else:
                print("Model is extracted file, it will work and allow further processing")
                #save_export_to_onnx32()
            returned_text=returned_text+f"\nUnet model fp32 converted: extracted files instead one weights file"

    returned_text=returned_text+f"\nONNX Model is here:{output_path}"

    return params['informacion']+returned_text
    

def save_memory_todisk(path:str,informacion:str):
    global loaded_model
    loaded_model.save_pretrained(path)
    return informacion+f"\nIm memory model saved to {path}" 

def load_vae(vae_path,text):
    from diffusers.models import AutoencoderKL
    import os, tempfile,torch
    from diffusers import StableDiffusionPipeline
    area="Loading new vae"
    dtype=torch.float32
    global loaded_model
    device=loaded_model.device
    if os.path.exists(vae_path) and vae_path.endswith(".pth"):
        with tempfile.TemporaryDirectory() as tmpdirname:
            debug(tmpdirname,area , True)
            loaded_model.save_pretrained(tmpdirname)        
            vae = AutoencoderKL.from_pretrained(vae_path,low_cpu_mem_usage=False)
            loaded_model = StableDiffusionPipeline.from_pretrained(tmpdirname,
                    torch_dtype=dtype, vae=vae,low_cpu_mem_usage=False).to(device)
    return text+"\nNew vae loaded into the pipeline"

def process_clipskip(value,device,text):
    import json, tempfile,torch
    from diffusers import StableDiffusionPipeline

    value=eval(value)
    dtype=torch.float32
    global loaded_model
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        loaded_model.save_pretrained(tmpdirname)
        print(tmpdirname)
        confname=f"{tmpdirname}/text_encoder/config.json"
        with open(confname, 'r', encoding="utf-8") as f:
            clipconf = json.load(f)
            clipconf['num_hidden_layers'] = clipconf['num_hidden_layers']-value+1
        with open(confname, 'w', encoding="utf-8") as f:
            json.dump(clipconf, f, indent=1)
        loaded_model = StableDiffusionPipeline.from_pretrained(tmpdirname,
            torch_dtype=dtype,low_cpu_mem_usage=False).to(device)    

    return text+"\nClip Skip:%s applied" % value,gr.Button.update(interactive=False)

def debug(element,area,imp=True):
    if imp:
        if type(element)==list:
            print(f"Info from {area}:")
            for info in element: print(info)
        else: print(f"Info from {area}:{element}")


def convert_unet_optimized(unet_path,fp16):
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.optimizer import optimize_model
    #model_dir=os.path.dirname("d:/model1/salsRealism_betaV40-auto-fp32/unet/model.onnx")
    unet_path=Path(unet_path)
    unet_model_path = str(unet_path.absolute().as_posix())
    unet_dir = os.path.dirname(unet_model_path)

    # First we set our optimisation to the ORT Optimizer defaults for unet
    optimization_options = FusionOptions("unet")
    # The ORT optimizer is designed for ORT GPU and CUDA
    # To make things work with ORT DirectML, we disable some options
    # On by default in ORT optimizer, turned off because it has no effect
    optimization_options.enable_qordered_matmul = False
    optimization_options.enable_nhwc_conv = False # On by default in ORT optimizer, turned off as it causes performance issues

    optimizer = optimize_model(
        #input = unet_model_path,
        input =unet_path,
        model_type = "unet",
        opt_level = 0,
        #opt_level = 1,
        optimization_options = optimization_options,
        #use_gpu = True,
        #only_onnxruntime = True
        use_gpu = False,
        only_onnxruntime = False
    )
    if fp16:
        optimizer.convert_float_to_float16(
            keep_io_types=True, disable_shape_infer=True, op_block_list=['RandomNormalLike']
        )
    optimizer.topological_sort()
    unet=optimizer.model
    del optimizer

    # clean up existing tensor files
    shutil.rmtree(unet_dir)
    os.mkdir(unet_dir)
    # collate external tensor files into one

    onnx.save_model(
        unet,
        unet_model_path,
        #"d:/model1/salsRealism_betaV40-auto-fp32/unet2/model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    return unet_model_path


def extract_unet(pipeline,output_path,opset,device):
    global show_debug
    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    unet_path = output_path + "\\unet\\model.onnx"
    unet_path = Path(unet_path)
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    dtype=torch.float32

    onnx_export(
        pipeline.unet,
        model_args=(
            torch.randn(2, unet_in_channels, unet_sample_size,
                unet_sample_size).to(device=device, dtype=dtype),
            torch.randn(2).to(device=device, dtype=dtype),
            torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
            False,
        ),
        output_path=unet_path,
        ordered_input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
        output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {0: "unet_sample_batch", 1: "unet_sample_channels", 2: "unet_sample_height", 3: "unet_sample_width"},
            "timestep": {0: "unet_timestep_batch"},
            "encoder_hidden_states": {0: "unet_ehs_batch", 1: "unet_ehs_sequence"},
        },
        opset=opset,
    )
    unet_model_path = str(unet_path.absolute().as_posix())

    debug("line 465: free some memory for next steps, during the optimization phase it may fail running out of memory.","extraction of models to disk",show_debug)
    del pipeline


    return unet_model_path

def convert_tokenizer(pipe,path):
    #from transformers import CLIPTokenizer
    pipe.tokenizer.save_pretrained(path+"\\tokenizer")

def convert_scheduler(pipe,path):
    #from diffusers.schedulers.scheduling_utils import SchedulerMixin
    pipe.scheduler.save_pretrained(path+"\\scheduler")


def generate_model_config(pipe,path):
    print("Pending to be implemented, just copy one from any other model")
    pipe.save_config(path)
    return

def convert_text_encoder(pipeline,output_path,opset,device):
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    textenc_path=output_path+"\\text_encoder\\model.onnx"
    textenc_path=Path(textenc_path)    
    onnx_export(
        pipeline.text_encoder,
        # casting to torch.int32 https://github.com/huggingface/transformers/pull/18515/files
        model_args=(text_input.input_ids.to(device=device, dtype=torch.int32)),
        output_path=textenc_path,
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "textenc_inputids_batch", 1: "textenc_inputids_sequence"},
        },
        opset=opset,
    )
    textenc_model_path = str(textenc_path.absolute().as_posix())

    return textenc_model_path

def convert_vae_encoder(pipeline,output_path,opset,device):
    dtype=torch.float32
    vaepath=output_path+"\\vae_encoder\\model.onnx"
    vaepath=Path(vaepath)  

    vae_encoder = pipeline.vae
    vae_in_channels = vae_encoder.config.in_channels
    vae_sample_size = vae_encoder.config.sample_size
    # need to get the raw tensor output (sample) from the encoder
    vae_encoder.forward = lambda sample, return_dict: vae_encoder.encode(sample,
        return_dict)[0].sample()
    onnx_export(
        vae_encoder,
        model_args=(
            torch.randn(1, vae_in_channels, vae_sample_size,
                vae_sample_size).to(device=device, dtype=dtype),
            False,
        ),
        output_path=vaepath,
        ordered_input_names=["sample", "return_dict"],
        output_names=["latent_sample"],
        dynamic_axes={
            "sample": {0: "vaeenc_sample_batch", 1: "vaeenc_sample_channels", 2: "vaeenc_sample_height", 3: "vaeenc_sample_width"},
        },
        opset=opset,
    )
       
    return vaepath

def convert_vae_decoder(pipeline,output_path,opset,device):
    dtype=torch.float32
    vaepath=output_path+"\\vae_decoder\\model.onnx"
    vaepath=Path(vaepath) 
    vae_decoder = pipeline.vae
    vae_encoder = pipeline.vae
    unet_sample_size = pipeline.unet.config.sample_size  
    vae_latent_channels = vae_decoder.config.latent_channels
    vae_out_channels = vae_decoder.config.out_channels
    # forward only through the decoder part
    vae_decoder.forward = vae_encoder.decode
    onnx_export(
        vae_decoder,
        model_args=(
            torch.randn(1, vae_latent_channels, unet_sample_size,
                unet_sample_size).to(device=device, dtype=dtype),
            False,
        ),
        output_path=vaepath,
        ordered_input_names=["latent_sample", "return_dict"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "vaedec_sample_batch", 1: "vaedec_sample_channels", 2: "vaedec_sample_height", 3: "vaedec_sample_width"},
        },
        opset=opset,
    )

    return vaepath

def save_export_to_onnx32():
    unet=onnx.load(unet_model_path)

def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
):
    from torch.onnx import export    
    '''export a PyTorch model as an ONNX model'''
    global show_debug
    debug(output_path,"extraction of model files",show_debug)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export(
        model,
        model_args,
        f=output_path.as_posix(),
        input_names=ordered_input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset,
    )

@torch.no_grad()
def convert_to_fp16(
    model_path
):
    '''Converts an ONNX model on disk to FP16'''
    
    model_dir=os.path.dirname(model_path)
    # Breaking down in steps due to Windows bug in convert_float_to_float16_model_path
    onnx.shape_inference.infer_shapes_path(model_path)
    fp16_model = onnx.load(model_path)
    fp16_model = convert_float_to_float16(
        fp16_model, keep_io_types=True, disable_shape_infer=True
    )
    # clean up existing tensor files
    shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    # save FP16 model
    onnx.save(fp16_model, model_path)
    return model_path

@torch.no_grad()
def convert_to_fp16_2(
    model_path
):
    '''Converts an ONNX extracted model on disk to FP16 in a unet2 subdirectory, keeping source files'''
    
    model_dir=os.path.dirname(model_path)
    model_dir2=model_dir[:-5]+"\\unet2"
    os.mkdir(model_dir2)

    model_path2=model_dir2+"\\model.onnx"
    # Breaking down in steps due to Windows bug in convert_float_to_float16_model_path
    onnx.shape_inference.infer_shapes_path(model_path)
    fp16_model = onnx.load(model_path)
    fp16_model = convert_float_to_float16(
        fp16_model, keep_io_types=True, disable_shape_infer=True
    )
    # clean up existing tensor files
    #shutil.rmtree(model_dir)
    #os.mkdir(model_dir)
    # save FP16 model
    onnx.save(fp16_model, model_path2)
    return model_path
if __name__== "__main__":
    converter =init_ui()
    converter.launch(server_port=7860)




