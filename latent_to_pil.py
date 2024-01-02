from Engine import Vae_and_Text_Encoders,engine_common_funcs
from optimum.onnxruntime.modeling_ort import ORTModel
import numpy as np
import os,sys
latent_path="./latents"
latent_list = []
divider=1

for i, arg in enumerate(sys.argv):
    print(f"Argument {i:>6}: {arg}")

try:
    with os.scandir(latent_path) as scan_it:
        for entry in scan_it:
            if ".npy" in entry.name:
                latent_list.append(entry.name)
    print(latent_list)                    
except:
    print("Not numpys found.Wrong Directory or no files inside?")        

vae_path="D:\\models\\ModelosVAE\\vae_decoder-standar"
vaedec_sess = ORTModel.load_model(vae_path+"/model.onnx", provider="DmlExecutionProvider", provider_options={'device_id': 1})

for latent in latent_list:
    try:
        loaded_latent=np.load(f"./latents/{latent}")
        loaded_latent = 1 / 0.18215 * loaded_latent

        import torch.nn.functional as F
        import torch.onnx
        import torch
        latent1=torch.from_numpy(loaded_latent)
        print(latent1.size())
        latent1= F.interpolate(latent1,size=(int(latent1.size()[2]/divider), int(latent1.size()[3]/divider)), mode='bilinear')
        print(latent1.size())
        loaded_latent = latent1.numpy()  
                
        image= vaedec_sess.run(['sample'],{'latent_sample': loaded_latent})[0]
        name= latent[:-3]
        name= name+"png"
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))
        image = engine_common_funcs.numpy_to_pil(image)[0]
        image.save(f"./latents/{name}",optimize=True)
        print(f"Saved:{name}")
    except:
        print(f"Error opening/processing:{latent}")

del vaedec_sess