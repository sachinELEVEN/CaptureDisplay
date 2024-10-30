import os
import sys

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
        #since prod builds are done from ./build-pipeline that is our base path, we need to go one level down from that so that we can access other resources in normal fashion from other files during development like the assets folder
        relative_path = '../' + relative_path  if is_prod else relative_path
        return os.path.join(base_path, relative_path)