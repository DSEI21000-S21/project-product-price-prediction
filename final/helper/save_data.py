import os
import numpy as np
from contextlib import redirect_stdout

def save_np_file(dir, filename, data):
    if dir !="":
        dir += '/'
        if not os.path.exists(dir):
            os.makedirs(dir)

    with open('%s%s'%(dir,filename), 'wb') as f:
        np.save(f, data)


def save_model_structure(dir, filename, model):
    if dir !="":
        dir += '/'
        if not os.path.exists(dir):
            os.makedirs(dir)
    print("Save Model: ", filename)
    with open('%s%s'%(dir,filename), "w+") as f:
        with redirect_stdout(f):
            model.summary()

def save_model(dir, filename, model):
    if dir !="":
        dir += '/'
        if not os.path.exists(dir):
            os.makedirs(dir)
    print("Save Model: ", filename)
    model.save('%s%s'%(dir,filename))
