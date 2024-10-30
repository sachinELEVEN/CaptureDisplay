import os
import sys
import re

def get_resource_path(relative_path):
        #Paths change after creating an artifact using pyinstaller.
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        #This should be used in only dist, in dev mode return relative path
        if getattr(sys, 'frozen', False):  # PyInstaller sets this attribute
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")

        # globalSpace.append_to_logs("IN PY_INSTALLER",sys._MEIPASS)
        is_prod = True
        #since prod builds are done from ./build-pipeline that is our base path, we need to remove build-pipeline so that we can access other resources in normal fashion from other files during development like the assets folder
        base_path = modify_path(base_path,'build-pipeline')
        return os.path.join(base_path, relative_path)

def modify_path(path, part_to_remove):
    # Normalize the path to remove any redundant relative parts (like "../" or "./")
    normalized_path = os.path.normpath(path)
    
    # Remove the specified part using regex
    modified_path = re.sub(rf'/{re.escape(part_to_remove)}', '', normalized_path, count=1)
    
    return modified_path