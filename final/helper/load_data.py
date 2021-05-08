import numpy as np

def read_np_file(dir, filename):
    return np.load('%s/%s'%(dir,filename))

def read_model_struture(model_name):
    with open('%s/%s_structure.txt'%(model_name,model_name), 'r') as f:
        for line in f.readlines():
            print(line.strip())
