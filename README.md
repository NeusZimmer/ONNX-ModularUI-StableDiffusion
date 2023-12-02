# ONNX-ModularUI-StableDiffusion
This is the Optimum version of a UI for Stable Diffusion, running on ONNX models for faster inference, working on most common GPU vendors: NVIDIA,AMD GPU...as long as they got support into onnxruntime


Point to old version ( works up to diffusers 14.0):https://github.com/NeusZimmer/ONNX-Stable-Diffusion-ModularUI
(TXT2IMG, HIRES-TXT2IMG, IMG2IMG, ControlNet, InstructPix, 3 approaches for face/image restoration, danbooru tagging ...and more)

**Initial Beta version, only including TXT2IMG, IMG2IMG and HIRES approach of TXT2IMG**

**Updates:**
-Added a working UI interface for model conversion, to ease the process.

Supporting: wildcards,pre-defined styles, latent import for hires-txt2img and txt2img inferences, HiRes supports using same or different models for low and hires inferences...
Adapted version of my approach for a working version of stabble diffusion for onnx, able to run on low profile GPUs (also in high-profile), dedicated GPU memory needed:4Gb.

# Install (20-30 min, difficulty low/medium): 

At first, download or clone this repository: ```git clone https://github.com/NeusZimmer/ONNX-ModularUI-StableDiffusion```
Then, as usual, create a python virtual environment for the install ( recommended, if you want to do it over you base python install that's up to you)
Python version: 3.10 ( tested, please feedback if you make it run in 3.11)
```
"cd ONNX-ModularUI-StableDiffusion"
"python -m venv venv"
```
 #(or  ```py -3.10 -m venv venv ``` if running multiple python instances on the same machine)

 Update pip:  
```pip install --upgrade pip```

Install requirements file:  
```pip install -r requirements.txt```

Optional, for face restoration/swapping verssion
```pip install -r additional-requirements.txt```

When finished, I recommend to reinstall again the onnxruntime package, only one of the listed below, according to your GPU vendor, sometimes it did not recognize the adecuate ExecutionProviders (i.e.DMLExecutionProvider) until this package is reinstalled"
 (Select this last package install according to onnx documentation for NVIDIA, Intel Vino, Mac...more info about what package you need to install: https://onnxruntime.ai/docs/install/#python-installs)

```
    #Attention: ONLY ONE OF THE LIST BELOW:
    #AMD (and other DirectMl supported cards):
    pip install onnxruntime-directml
    #CPU:
    pip install onnxruntime
    #CUDA-TENSORRT
    pip install onnxruntime-gpu
    #OpenVino
    pip install onnxruntime-openvino
    pip install openvino
    #Jetson
    #Here: https://elinux.org/Jetson_Zoo#ONNX_Runtim
    #Azure:
    pip install onnxruntime-azure
```
Optional: download the onnx model of your preference and include it into the models folder ( or configure the directory of your preference in the Configuration Tab).
Sample model: "https://civitai.com/models/125580/onnx-base-set-of-model-neusz"
(You will need to download the main model, the vae decoder, vae encoder and text encoder and put each one in the adequate subdir)

Activate the environment  ```activate.bat```
and the run the UI: ```py -O ONNX-StableDiffusion.py```
or ```run.bat```

Point your browser to localhost:7860 to accessing the UI app and enjoy yourself...

#  Utility for Model Conversion: 
 (conversion code extracted from: https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16)
```
cd ConversionTool
py ONNX-SD-ModelConverter.py
```
And point your browser to localhost:7860






