from optimum.onnxruntime.modeling_ort import ORTModel
from Engine.General_parameters import Engine_Configuration

import json


class Borg1:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state
        
class Vae_and_Text_Encoders(Borg1):
    vae_decoder = None
    vae_encoder = None
    text_encoder = None

    def __init__(self):
        Borg1.__init__(self)

    def __str__(self): return json.dumps(self.__dict__)

    def load_vaedecoder(self,model_path):
        from Engine.General_parameters import running_config        
        """if " " in Engine_Configuration().VAEDec_provider:
            provider =eval(Engine_Configuration().VAEDec_provider)
        else:
            provider =Engine_Configuration().VAEDec_provider"""
        
        provider=Engine_Configuration().VAEDec_provider['provider']
        provider_options=Engine_Configuration().VAEDec_provider['provider_options']

        running_config=running_config()
        import os
        if running_config.Running_information["Vae_Config"]:
            vae_config=running_config.Running_information["Vae_Config"]
            vae_path1= (model_path + "/vae_decoder") if vae_config[0]=="model" else vae_config[0]
            vae_path2= (model_path + "/vae_decoder") if vae_config[1]=="model" else vae_config[1]
            vae_path3= (model_path + "/vae_decoder") if vae_config[2]=="model" else vae_config[2]                       
            vae_path=""

            if os.path.exists(vae_path1): vae_path= vae_path1
            elif os.path.exists(vae_path2): vae_path= vae_path2
            elif os.path.exists(vae_path3): vae_path= vae_path3
            else: raise Exception("No valid vae decoder path"+vae_path)

        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level=3

        self.vae_decoder = None
        print(f"Loading VAE decoder in:{provider}, from {vae_path} with options:{provider_options}" )
        #self.vae_decoder = ORTModel.load_model(vae_path+"/model.onnx", provider,None, provider_options={'device_id': 1})
        #self.vae_decoder = ORTModel.load_model(vae_path+"/model.onnx", provider,sess_options, provider_options={'device_id': 1})
        self.vae_decoder = ORTModel.load_model(vae_path+"/model.onnx", provider,sess_options, provider_options=provider_options)
        return self.vae_decoder

    def load_vaeencoder(self,model_path):
        from Engine.General_parameters import running_config        
        running_config=running_config()
        import os

        vae_config=running_config.Running_information["Vae_Config"]
        vae_path1= (model_path + "/vae_encoder") if vae_config[3]=="model" else vae_config[3]
        vae_path2= (model_path + "/vae_encoder") if vae_config[4]=="model" else vae_config[4]
        vae_path3= (model_path + "/vae_encoder") if vae_config[5]=="model" else vae_config[5]
        vae_path=""

        if os.path.exists(vae_path1): vae_path= vae_path1
        elif os.path.exists(vae_path2): vae_path= vae_path2
        elif os.path.exists(vae_path3): vae_path= vae_path3
        else: raise Exception("No valid vae encoder path:"+vae_path)

        """
        if " " in Engine_Configuration().VAEDec_provider:
            provider =eval(Engine_Configuration().VAEDec_provider)
        else:
            provider =Engine_Configuration().VAEDec_provider"""

        provider=Engine_Configuration().VAEEnc_provider['provider']
        provider_options=Engine_Configuration().VAEEnc_provider['provider_options']

        #vae_path=model_path + "/vae_encoder"
        self.vae_encoder = None

        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level=3
        print(f"Loading VAE encoder in:{provider}, from {vae_path} with options:{provider_options}")
        self.vae_encoder = ORTModel.load_model(vae_path+"/model.onnx", provider,sess_options, provider_options=provider_options)
        

        return self.vae_encoder

    def load_textencoder(self,model_path):
        from Engine.General_parameters import running_config        
        """if " " in Engine_Configuration().TEXTEnc_provider:
            provider = eval(Engine_Configuration().TEXTEnc_provider)
        else:
            provider = Engine_Configuration().TEXTEnc_provider"""

        provider=Engine_Configuration().TEXTEnc_provider['provider']
        provider_options=Engine_Configuration().TEXTEnc_provider['provider_options']

        running_config=running_config()
        import os
        if running_config.Running_information["Textenc_Config"]:
            Textenc_Config=running_config.Running_information["Textenc_Config"]
            Textenc_path1= (model_path + "/text_encoder") if Textenc_Config[0]=="model" else Textenc_Config[0]
            Textenc_path2= (model_path + "/text_encoder") if Textenc_Config[1]=="model" else Textenc_Config[1]                     
            Textenc_path=""
            if os.path.exists(Textenc_path1): Textenc_path= Textenc_path1
            elif os.path.exists(Textenc_path2): Textenc_path= Textenc_path2
            else: raise Exception("No valid Text Encoder path:"+Textenc_path)


        print(f"Loading TEXT encoder in:{provider} from:{Textenc_path} with options:{provider_options}" )
        self.text_encoder = None
        #self.text_encoder  = ORTModel.load_model(Textenc_path+"/model.onnx", provider,None,None)      
        self.text_encoder  = ORTModel.load_model(Textenc_path+"/model.onnx", provider,None,provider_options)      

        return self.text_encoder
    
    def unload_from_memory(self):
        import gc
        self.vae_decoder = None
        self.vae_encoder = None
        self.text_encoder = None
        gc.collect()


