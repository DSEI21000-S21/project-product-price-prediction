import os
import numpy as np

def save_np_file(dir, filename, data):
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open('%s/%s'%(dir,filename), 'wb') as f:
        np.save(f, dir)
