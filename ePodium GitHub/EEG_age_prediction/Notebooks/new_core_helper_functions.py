# new core helper functions

import glob
import os

import pandas as pd
import numpy as np

import hashlib
import h5py
import warnings
import re

warnings.filterwarnings('ignore')

def hash_it_up_right_all(origin_folder1, file_extension):
    hash_list = []
    file_names = []
    files = '*' + file_extension
    non_suspects1 = glob.glob(os.path.join(origin_folder1, files))
    BUF_SIZE = 65536
    for file in non_suspects1:
        sha256 = hashlib.sha256()
        with open(file, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        result = sha256.hexdigest()
        hash_list.append(result)
        file_names.append(file)
        
    df = pd.DataFrame(hash_list, file_names)
    df.columns = ["hash"]
    df = df.reset_index() 
    df = df.rename(columns = {'index':'file_name'})
    
    return df