import os
import shutil


def skip_files(dir, files):
    # return [f for f in files if os.path.isfile(os.path.join(dir, f)) and f != 'pedestron_train.json']   
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


src = 'source path'
des = 'destination path'

shutil.copytree(src, des, ignore=skip_files)
