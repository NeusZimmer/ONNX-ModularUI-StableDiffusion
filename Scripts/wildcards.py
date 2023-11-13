# Wildcards 
#V3 Extended to provide info on status area and recursive wildcards (including one wildcard into a line of a wildcard file  for increased variability)
#example a prompt could be:"An incrincated drawing of __example__ " and the inside the example.txt wildcard file a line: __test__ could include the line "A __color__ __vehicle__ running trough __landscape__ "
#An extension modificated version of a script from https://github.com/jtkelm2/stable-diffusion-webui-1/blob/main/scripts/wildcards.py
#Idea originated , but modified, from module wildcards of AUTOMATIC111
#Allows you to use `__name__` syntax in your prompt to get a random line from a file named `name.txt` in the wildcards directory.



import os
import random
import sys

warned_about_files = {}
wildcard_dir = os.getcwd()+"\Scripts"
#print(wildcard_dir)


class WildcardsScript():
    def title(self):
        return "Simple wildcards class for OnnxDiffusersUI"

    def replace_wildcard(self, text, gen):
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

    def process(self, original_prompt):
        string_replaced=""
        new_prompt=""
        gen = random.Random()
        text_divisions=original_prompt.split("__")

        for chunk in text_divisions:
            text,changed=self.replace_wildcard(chunk, gen)
            if changed:
                string_replaced=string_replaced+"Wildcard:"+chunk+"-->"+text+","
                new_prompt=new_prompt+text
            else:
                new_prompt=new_prompt+text        
        
        return  new_prompt, string_replaced
