from Engine.General_parameters import Engine_Configuration
from Engine import SchedulersConfig
from Engine import Vae_and_Text_Encoders
from Engine.engine_common_funcs import create_generator,seed_generator

from own_pipes.pipeline_onnx_stable_diffusion_hires_txt2img import OnnxOptimumStableDiffusionHiResPipeline
import numpy as np

"""from diffusers.utils import randn_tensor"""
#from diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion_hires_txt2img import OnnxStableDiffusionHiResPipeline
#from pipes.pipeline_onnx_stable_diffusion_hires_txt2img import OnnxStableDiffusionHiResPipeline



class Borg_hires:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class txt2img_hires_pipe(Borg_hires):
    hires_pipe = None
    model = None
    seeds = []
    latents_list = []

    def __init__(self):
        Borg_hires.__init__(self)
    def __str__(self): 
        import json
        return json.dumps(self.__dict__)

    def initialize(self,model_path,low_model_path,sched_name):
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level=3
        sess_options.enable_cpu_mem_arena=False
        sess_options.enable_mem_reuse= True
        sess_options.enable_mem_pattern = True
        #sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        if self.hires_pipe == None:
            if Vae_and_Text_Encoders().text_encoder == None:
                Vae_and_Text_Encoders().load_textencoder(model_path)
            if Vae_and_Text_Encoders().vae_decoder == None:
                Vae_and_Text_Encoders().load_vaedecoder(model_path)
            if Vae_and_Text_Encoders().vae_encoder == None:
                Vae_and_Text_Encoders().load_vaeencoder(model_path)

            provider=Engine_Configuration().MAINPipe_provider['provider']
            provider_options=Engine_Configuration().MAINPipe_provider['provider_options']
            low_res_provider=Engine_Configuration().LowResPipe_provider['provider']
            low_res_provider_options=Engine_Configuration().LowResPipe_provider['provider_options']

            self.hires_pipe = OnnxOptimumStableDiffusionHiResPipeline(
                model_path,
                provider=provider,
                scheduler=SchedulersConfig().scheduler(sched_name,model_path),
                text_encoder_session=Vae_and_Text_Encoders().text_encoder,
                vae_decoder_session=Vae_and_Text_Encoders().vae_decoder,
                vae_encoder_session=Vae_and_Text_Encoders().vae_encoder,
                sess_options=sess_options,
                provider_options=provider_options,
                low_res_model_path=low_model_path,
                low_res_provider=low_res_provider,
                low_res_provider_options=low_res_provider_options,
            )
        else:
            self.hires_pipe.scheduler=SchedulersConfig().scheduler(sched_name,model_path)

        import functools
        from Engine import lpw_pipe
        self.hires_pipe._encode_prompt = functools.partial(lpw_pipe._encode_prompt, self.hires_pipe)
        return self.hires_pipe

    def Convert_from_txt2img(self,txt_pipe,model_path,sched_name):
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level=3
        sess_options.enable_cpu_mem_arena=False
        sess_options.enable_mem_reuse= True
        sess_options.enable_mem_pattern = True
        #sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL


            #from Engine.General_parameters import Engine_Configuration as en_config
        if Vae_and_Text_Encoders().text_encoder == None:
            Vae_and_Text_Encoders().load_textencoder(model_path)
        if Vae_and_Text_Encoders().vae_decoder == None:
            Vae_and_Text_Encoders().load_vaedecoder(model_path)
        if Vae_and_Text_Encoders().vae_encoder == None:
            Vae_and_Text_Encoders().load_vaeencoder(model_path)

        unet=txt_pipe.unet

        self.hires_pipe = OnnxStableDiffusionHiResPipeline.from_pretrained(
            model_path,
            unet=unet,
            scheduler=SchedulersConfig().scheduler(sched_name,model_path),
            text_encoder=Vae_and_Text_Encoders().text_encoder,
            vae_decoder=Vae_and_Text_Encoders().vae_decoder,
            vae_encoder=Vae_and_Text_Encoders().vae_encoder,
            sess_options=sess_options
        )



        import functools
        from Engine import lpw_pipe
        self.hires_pipe._encode_prompt = functools.partial(lpw_pipe._encode_prompt, self.hires_pipe)


        return self.hires_pipe

    def use_hires_as_lowres(self):
        self.hires_pipe.use_hires_as_lowres()

    def reload_partial_model(self,model_path,LowHigh=None):
        if LowHigh!=None:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level=3
            sess_options.enable_cpu_mem_arena=False
            sess_options.enable_mem_reuse= True
            sess_options.enable_mem_pattern = True

            if Vae_and_Text_Encoders().text_encoder == None:
                Vae_and_Text_Encoders().load_textencoder(model_path)
            if Vae_and_Text_Encoders().vae_decoder == None:
                Vae_and_Text_Encoders().load_vaedecoder(model_path)
            if Vae_and_Text_Encoders().vae_encoder == None:
                Vae_and_Text_Encoders().load_vaeencoder(model_path)

            if LowHigh=="low":
                low_res_provider=Engine_Configuration().LowResPipe_provider['provider']
                low_res_provider_options=Engine_Configuration().LowResPipe_provider['provider_options']            
                self.hires_pipe.reload_lowres(
                    scheduler=SchedulersConfig().scheduler("DDIM",model_path),
                    text_encoder_session=Vae_and_Text_Encoders().text_encoder,
                    vae_decoder_session=Vae_and_Text_Encoders().vae_decoder,
                    vae_encoder_session=Vae_and_Text_Encoders().vae_encoder,
                    sess_options=sess_options,
                    low_res_model_path=model_path,
                    low_res_provider=low_res_provider,
                    low_res_provider_options=low_res_provider_options,
                )
            else: #reload highresmodel
                provider=Engine_Configuration().MAINPipe_provider['provider']
                provider_options=Engine_Configuration().MAINPipe_provider['provider_options']

                self.hires_pipe.reload_hires(
                    model_path,
                    provider=provider,
                    scheduler=SchedulersConfig().scheduler("DDIM",model_path),
                    text_encoder_session=Vae_and_Text_Encoders().text_encoder,
                    vae_decoder_session=Vae_and_Text_Encoders().vae_decoder,
                    vae_encoder_session=Vae_and_Text_Encoders().vae_encoder,
                    sess_options=sess_options,
                    provider_options=provider_options,
                )
        import gc
        gc.collect()

    def create_seeds(self,seed=None,iter=1,same_seeds=False):
        self.seeds=seed_generator(seed,iter)
        if same_seeds:
            for seed in self.seeds:
                seed = self.seeds[0]

    def unload_from_memory(self):
        import gc
        self.hires_pipe= None
        self.model = None
        self.latents_list = None
        gc.collect()


    ########### DELETE THIS FUNCTION
    def seed_generator1(self,seed,iteration_count):
        import numpy as np
        # generate seeds for iterations
        
        if seed == "" or seed == None:
            rng = np.random.default_rng()
            seed = rng.integers(np.iinfo(np.uint32).max)
            seeds = np.array([seed], dtype=np.uint32)
        else:
            seed_list=seed.split(',')
            seeds=[]
            for seedint in seed_list:
                try:
                    seed = int(seedint) & np.iinfo(np.uint32).max
                except ValueError:
                    seed = hash(seedint) & np.iinfo(np.uint32).max
                seeds.append(seed)

        # use given seed for the first iteration
        #seeds = np.array([seed], dtype=np.uint32)

        if iteration_count > len(seeds):
            seed_seq = np.random.SeedSequence(seed)
            seeds = np.concatenate((seeds, seed_seq.generate_state(iteration_count - len(seeds)))) # era restar 1

        return seeds[:iteration_count]
   
    def get_ordered_latents(self):
        from Engine.General_parameters import running_config
        import numpy as np
        name=running_config().Running_information["Latent_Name"]
        name1= name.split(',')
        lista=[0]*len(name1)
        for pair in name1:
            tupla= pair.split(':')
            lista[int(tupla[0])-1]=tupla[1]
        #print("Ordered numpys"+str(lista))
        return lista

    def sum_latents(self,latent_list,formula,generator,resultant_latents,iter=0):
        #print("Processing formula:"+str(formula))
        subformula_latents= None
        while ("(" in formula) or (")" in formula):
            #print("Subformula exists")
            subformula_startmarks=list([pos for pos, char in enumerate(formula) if char == '('])
            subformula_endmarks=list([pos for pos, char in enumerate(formula) if char == ')'])

            if (len(subformula_endmarks) != len(subformula_startmarks)):
                raise Exception("Sorry, Error in formula, check it")

            contador=0
            while (len(subformula_startmarks)>contador) and (subformula_startmarks[contador] < subformula_endmarks[0]):
                contador+=1
            if contador==0: raise Exception("Sorry, Error in formula, check it")

            subformula= formula[(subformula_startmarks[contador-1]+1):subformula_endmarks[0]]
            #print(f"subformula:{iter},{subformula}")
            previous= formula[0:subformula_startmarks[contador-1]]
            posterior=formula[subformula_endmarks[0]+1:]
            formula= f"{previous}|{iter}|{posterior}" 
            iter+=1
            subformula_latents =  self.sum_latents(latent_list,subformula,generator,resultant_latents,iter)
            resultant_latents.append(subformula_latents)


        # Here we got a plain formula
        #print("No subformulas")
        result = self.process_simple_formula(latent_list,formula,generator,resultant_latents)
        return result

    def process_simple_formula(self,latent_list,formula,generator,resultant_latents):
        position=-1
        #print("Simple_formula process")
        for pos, char in enumerate(formula):
            if char in "WwHh":
                position=pos
                break
        if position ==-1 and len(formula)>0:  #No operators, single item
            result=self.load_latent_file(latent_list,formula,generator,resultant_latents)
        else:
            previous=formula[0:position]
            operator=formula[position]
            rest=formula[position+1:]
            #print("previous:"+previous)
            #print("operator:"+operator)
            #print("rest:"+rest)

            result=self.load_latent_file(latent_list,previous,generator,resultant_latents)
            result2 = self.process_simple_formula(latent_list,rest,generator,resultant_latents)

            if (operator=='w'):
                result = self._sum_latents(result,result2,True) #left & right
            elif (operator=='h'):
                result = self._sum_latents(result,result2,False) #Up & Down

        return result

    def load_latent_file(self,latent_list,data,generator,resultant_latents):
        result = ""
        if "|" in data:
            lista=data.split("|")
            index=int(lista[1])
            result = resultant_latents[index]
            #result = "SP:"+resultant_latents[index]
        else:
            index=int(data)
            name=latent_list[int(index)-1]
            if "noise" not in name:
                print(f"Loading latent(idx:name):{index}:{name}")
                result=np.load(f"./latents/{name}")

                """import torch
                latents_dtype = result.dtype
                noise = generator.randn(*result.shape).astype(latents_dtype)
                result = self.hires_pipe.scheduler.add_noise(
                    torch.from_numpy(result), torch.from_numpy(noise), torch.from_numpy(np.array([1]))
                )
                result =result.numpy()"""
            else:
                noise_size=name.split("noise-")[1].split("x")
                print(f"Creating noise block of W/H:{noise_size}")
                noise = (0.1)*(generator.random((1,4,int(int(noise_size[1])/8),int(int(noise_size[0])/8))).astype(np.float32))
                result = noise

        return result

    def _sum_latents(self,latent1,latent2,direction): #direction True=horizontal sum(width), False=vertical sum(height)
        latent_sum= None
        side=""
        try:
            if direction:
                side="Height"
                latent_sum = np.concatenate((latent1,latent2),axis=3) #left & right
            else:
                side="Width"
                latent_sum = np.concatenate((latent1,latent2),axis=2)  #Up & Down
        except:
            size1=f"Latent1={(latent1.shape[3]*8)}x{(latent1.shape[2]*8)}"
            size2=f"Latent2={(latent2.shape[3]*8)}x{(latent2.shape[2]*8)}"
            raise Exception(f"Cannot sum the latents(Width x Height):{size1} and {size2} its {side} must be equal")
        return latent_sum

    def get_initial_latent(self, steps,multiplier,generator,strengh):
        debug = False
        from Engine.General_parameters import running_config
        latent_list=self.get_ordered_latents()
        formula=running_config().Running_information["Latent_Formula"]
        formula=formula.replace(' ', '')
        formula=formula.lower()

        loaded_latent=self.sum_latents(latent_list,formula,generator,[])

        print("Resultant Latent Shape "+"H:"+str(loaded_latent.shape[2]*8)+"x W:"+str(loaded_latent.shape[3]*8))

        return loaded_latent

    def savelatents_todisk(self,path="./latents",seed=0,save_steps=True,contador=1000,callback_steps=2):
        import numpy as np
        if self.latents_list:
            latent_to_save= self.latents_list.pop()
            if save_steps:
                self.savelatents_todisk(path=path,seed=seed,save_steps=save_steps,contador=contador-1,callback_steps=callback_steps)
            np.save(f"{path}/Seed-{seed}_latent_Step-{contador*callback_steps}.npy", latent_to_save)
        return

    def resize_and_crop(self,input_image, height, width):
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

    ################# MAIN CALL FOR INFERENCE #################
    #Cambiar la entrada a un diccionario?

    def run_inference(self,prompt,neg_prompt,hires_passes,height,width,hires_height,hires_width,steps,hires_steps,guid,eta,batch,seed,strength,strength_var,upscale_method,output_format):
        import numpy as np
        from Engine.General_parameters import running_config

        rng = create_generator(int(seed),generator_type='torch')
        #rng = create_generator(int(seed),generator_type='numpy')
        prompt.strip("\n")
        neg_prompt.strip("\n")

        var=self.hires_pipe.scheduler.__str__()[0:12]
        if not ('LCMScheduler' in var):
            hires_steps=int(hires_steps/strength)

        if running_config().Running_information["Load_Latents"]:
            multiplier=1 #no usado?
            loaded_latent=self.get_initial_latent(steps,multiplier,rng,strength)            
        else:
            loaded_latent=None

        lowres_image,hires_images = self.hires_pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            height=height,
            width=width,
            hires_height=hires_height,
            hires_width=hires_width,            
            num_inference_steps=steps,
            num_hires_steps=hires_steps,
            guidance_scale=guid,
            eta=eta,
            num_images_per_prompt=batch,
            prompt_embeds = None,
            negative_prompt_embeds = None,
            latents=loaded_latent,
            output_type= output_format,
            strength=strength,
            strength_var=strength_var,
            hires_steps=hires_passes,
            #callback= self.__callback,
            #callback_steps = running_config().Running_information["Callback_Steps"],
            generator=rng,
            upscale_method=upscale_method
            )

        #dictio={'prompt':prompt,'neg_prompt':neg_prompt,'height':height,'width':width,'steps':steps,'guid':guid,'eta':eta,'batch':batch,'seed':seed}

        #return lowres_image,hires_images,dictio  #WARNING:they could be pil or latents, depending on the requested format
        return lowres_image,hires_images  #WARNING:they could be pil or latents, depending on the requested format
 