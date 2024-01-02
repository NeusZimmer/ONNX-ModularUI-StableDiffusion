"""
Initialize:

Return: dict of 3 elements
1st: a list of tab names where you want it to be included,
2nd:  the area where to be shown in the UI
3rd: the point where to execute the call 

Additionally you need two funcions: one named "__call__" and another named "__show__"
if no need for an specific function, create one with the return being the same as the imput or a pass function    
    function __show__: the gradio elements to be shown into the UI component in the specified TAB & AREA, 
            def show(*args):
                pass
    function __call__: this will be called to process a dictionay of parameters or a specific parameter, depending on where is to be included
            def __call__(*args):
                return args

tabs names:["txt2img","hires"] --


"""



def __init__(*args):
    __name__='LatentFormulaProcessor'
    print(args[0])
    #here a check of access of initializacion  if needed
    return {
        'tabs':["hires"], #Añadir tambien a txt2img
        'ui_position':None, #por definir
        'func_processing':None}

def __call__(*args):
    #what to do when the module is called 
    print("Processing Latent Formula Processor Module")
    if args:
        datos=args[0]
        if type(datos)==dict:
            print("LLega dict")
        elif type(datos)==str:
            print("LLega string")
        else:
            print("Not recognized input for module Latent Formula Processor %s" % datos)

    return None

def is_global_ui():
    return False

def is_global_function():
    return False

def show():
    show_latent_processor_ui()

def show_latent_processor_ui():
    import gradio as gr
    with gr.Accordion("Latent formula processor module",open=False):
        pass


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
