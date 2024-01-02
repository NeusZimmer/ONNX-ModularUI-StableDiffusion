"""
Initialize the modules (__init__) to get the relevant info of what to do with them:

The loaded modules must return a tuple of 4 elements
1st: a list of tab names where to be shown if ui component,
2nd:  the area where to include the Ai and area to include the process to be done within the module
3rd and 4th: two funcions: one named "__call__" and another named "show"
if no function, create one with the return being the same as the imput or a pass function    
    3rd: show: will be user to show the UI component of the module in the specified TAB & AREA, 
            def show(*args):
                pass
    4th: Call will be used to process a dictionay of parameters or a specific parameter, depending on where is to be included
            def __call__(*args):
                return args
ATT: not implemented the area zones, fixed.
tabs names:["txt2img","hires"] --


"""
class Borg20:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state



class preProcess_modules(Borg20):
    all_modules=None
    _loaded=False


    def __init__(self):
        Borg20.__init__(self)
        if not self._loaded:
            self.__initclass__()

    def __str__(self): 
        import json
        return json.dumps(self.__dict__)

    def __initclass__(self):
        #self.all_modules=self._launch_preprocess_modules()
        self.all_modules=self._load_all_modules()
        self._loaded=True

    def check_available_modules(self,tab_name):
        available_modules=[]
        for module in self.all_modules:
            if tab_name in module['tabs']:
                available_modules.append(module)

        return available_modules #do not use a self var, it will provide always the functions for the last UI tab loaded



    def _load_all_modules(*args,**kwargs):
    #def _load_preprocess_modules(*args,**kwargs):
        from importlib import import_module

        #lista=['library_module','wildcards_module','styles_module','image_to_numpy_module','reload_hires_module']
        lista=['wildcards_module','styles_module','image_to_numpy_module']
        modules_data=[]

        for elemento in lista:
            my_modulo=import_module('modules.'+elemento, package="StylesModule")
            modules_info=my_modulo.__init__("External Module %s Loaded" % elemento )
            functions=(my_modulo.show,my_modulo.__call__,my_modulo.is_global_ui,my_modulo.is_global_function)
            modules_info.update({"show": functions[0]})
            modules_info.update({"call": functions[1]})
            modules_info.update({"is_global_ui": functions[2]})
            modules_info.update({"is_global_function": functions[3]})
            modules_data.append(modules_info)

        return modules_data  # A list of dicts, one for each module



if __name__ == "__main__":
    print("This is the module loader and is not intended to run as standalone")
    pass
else:
    __name__ = "ExternalModuleLoader"
