
##Here only common functions who does not call UI variables
import os,re,PIL,numpy
from PIL import Image, PngImagePlugin

class generator(numpy.random.RandomState):
    def __init__(self,seed,generator_type='numpy'):  #True = numpy
        numpy_or_torch=True if generator_type=='numpy'else False
        self.seed=seed
        if numpy_or_torch:
            super().__init__(self.seed)
        else:
            import torch
            self.torch_gen = torch.Generator()
            self.torch_gen.manual_seed(self.seed)
            self.randn=self.random
            self.randint

    def randint(self,low,max):
        import torch
        #dato=self.torch_gen.seed()
        dato=torch.random(self.torch_gen,low,max)
        return dato
    
    def random(self,*args):
    #def random(self,shape):
        shape=tuple(args)
        import torch
        #torch.manual_seed(self.seed)
        #retorno=torch.randn(shape)
        retorno=torch.randn(shape, generator=self.torch_gen)
        try:
            return retorno.numpy()
        except:
            pass
        #finally:
        #    self=retorno


def create_generator(seed,generator_type='numpy'):
    import torch
    import numpy as np
    
    if generator_type=='numpy':
        generator1=generator(seed,generator_type='numpy')
    elif generator_type=='torch':
        generator1=generator(seed,generator_type='torch')
    else:
        print("Generator type does not configured or recognized")
    #print(type(generator1))
    return generator1

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    from PIL import Image
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images



def get_next_save_index(output_path):
    #output_path=UI_Configuration().output_path
    dir_list = os.listdir(output_path)
    if len(dir_list):
        pattern = re.compile(r"([0-9][0-9][0-9][0-9][0-9][0-9])-([0-9][0-9])\..*")
        match_list = [pattern.match(f) for f in dir_list]
        next_index = max([int(m[1]) if m else -1 for m in match_list]) + 1
    else:
        next_index = 0
    return next_index


def save_image(batch_images,info,next_index,output_path,style=None,save_textfile=False,low_res=False):
    #output_path=UI_Configuration().output_path
    info_png = f"{info}"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("parameters",info_png)
    prompt=info["prompt"]

    style_pre =""
    style_post=""
    style_pre_len=0
    style_post_len=0

    if style:
        styles=style.split("|")
        style_pre =styles[0]
        style_pre_len=len(style_pre)
        style_post=styles[1]
        style_post_len=len(style_post)
        prompt=prompt[style_pre_len-1:-style_post_len]
        """print(f"prompt:{prompt}")
    else:
        print("No tocado el prompt")"""

    short_prompt1 = prompt.strip("(){}<>:\"/\\|?*\n\t")
    short_prompt1 = re.sub(r'[\\/*?:"<>()|\n\t]', "", short_prompt1)
    short_prompt = short_prompt1[:49] if len(short_prompt1) > 50 else short_prompt1
    os.makedirs(output_path, exist_ok=True)

    i=0
    if low_res:
        low_res_text="-LowRes"
    else:
        low_res_text=""

    process_tags=True
    if not process_tags:
        for image in batch_images:
            if image!=None:
                image.save(os.path.join(output_path,f"{next_index:06}{low_res_text}-0{i}_{short_prompt}.png",),optimize=True,pnginfo=metadata,)
            i+=1
    else:
        for image in batch_images:
            if image!=None:
                image.save(os.path.join(output_path,f"{next_index:06}{low_res_text}-0{i}.png",),optimize=True,pnginfo=metadata,)
                if save_textfile and (i==0):
                    #print(f"low_res_text:{low_res_text}-{low_res_text}")
                    #print(f"low_res_text:{next_index:06}{low_res_text}")
                    with open(os.path.join(output_path,f"{next_index:06}-{low_res_text}-0{i}.txt"), 'w',encoding='utf8') as txtfile:
                        txtfile.write(f"{short_prompt1} \nstyle_pre:{style_pre}\nstyle_post:{style_post}")
            i+=1

def PIL_resize_and_crop(input_image: PIL.Image.Image, height: int, width: int):
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


def load_tokenizer_and_config(model_path):
        import json
        CONFIG_NAME="model_index.json"
        config_file=model_path+"/"+CONFIG_NAME
        config=None
        try:
            import json
            with open(config_file, "r", encoding="utf-8") as reader:
                text = reader.read()
                config=json.loads(text)
        except:
            raise OSError(f"model_index.json not found in {model_path} local folder")
        
        from transformers import CLIPTokenizer
        return CLIPTokenizer.from_pretrained(model_path+"/"+'tokenizer'),config

def seed_generator(seed,iteration_count):
    import numpy as np
    # generate seeds for iterations
    if seed == "" or seed == None:
        rng = np.random.default_rng()
        seed = rng.integers(np.iinfo(np.uint32).max)
    else:
        try:
            seed = int(seed) & np.iinfo(np.uint32).max
        except ValueError:
            seed = hash(seed) & np.iinfo(np.uint32).max

    # use given seed for the first iteration
    seeds = np.array([seed], dtype=np.uint32)

    if iteration_count > 1:
        seed_seq = np.random.SeedSequence(seed)
        seeds = np.concatenate((seeds, seed_seq.generate_state(iteration_count - 1)))

    return seeds