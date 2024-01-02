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
    __name__='WildcardModule'
    print(args[0])
    #here a check of access of initializacion  if needed
    #return (["txt2img","hires"],"prompt_process")

    return {
        'tabs':["txt2img","hires"],
        'ui_position':"prompt_process",
        'func_processing':'prompt_process'}

def __call__(*args):
    #what to do when the module is called 
    #print("Processing wildcards Module")
    if args:
        datos=args[0]
        if type(datos)==dict:
            print("Dummy1-Wildcards")
            datos1=process(datos['prompt'])
        elif type(datos)==str:
            datos1=process(datos)
            #print("Dummy2")
        else:
            print("Not recognized input for module wildcards %s" % datos) 
    return datos1

def is_global_ui():
    return False

def is_global_function():
    return False

def show():
    import gradio
    gradio.Markdown("Wildcards Module Activated")
    #pass


def replace_wildcard(text, gen):
    import os,sys

    warned_about_files = {}
    wildcard_dir = os.getcwd()+"\Scripts"

    if " " in text or len(text) == 0:
        return text,False

    replacement_file = os.path.join(wildcard_dir, "wildcards", f"{text}.txt")
    if os.path.exists(replacement_file):
        with open(replacement_file, encoding="utf8") as f:
            changed_text=gen.choice(f.read().splitlines())
            if "__" in changed_text:
                changed_text, not_used = self.process(changed_text)
            return changed_text,True
    else:
        if replacement_file not in warned_about_files:
            print(f"File {replacement_file} not found for the __{text}__ wildcard.", file=sys.stderr)
            warned_about_files[replacement_file] = 1

    return text,False

def process(original_prompt):
    import random
    string_replaced=""
    new_prompt=""
    gen = random.Random()
    text_divisions=original_prompt.split("__")

    for chunk in text_divisions:
        text,changed=replace_wildcard(chunk, gen)
        if changed:
            string_replaced=string_replaced+"Wildcard:"+chunk+"-->"+text+","
            new_prompt=new_prompt+text
        else:
            new_prompt=new_prompt+text        
    
    return  new_prompt
    #return  new_prompt, string_replaced