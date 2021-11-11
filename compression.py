#%% load modules ~10s
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import skimage.color as skm
import pickle
import bz2
import blosc
import lzma
import sys


def decomp(filename, kind):
    """Loads a compressed file from /Store, returns the file

    Parameters
    ----------
    f_name : string
        Filename of compressed file in /Store, 
        include file extension
    kind : int
        0 = bz2 \n
        1 = lzma
    """    
    path = sys.path[0] + '/Store/' + filename

    if kind == '0':
        with bz2.BZ2File(path, 'rb') as f:
            file = pickle.load(f)
    elif kind == '1': 
        with lzma.open(path, 'rb') as f:
            file = pickle.load(f)
    else:
        print("ERROR: type must be bz2(0) or lzma(1)")
    
    return file

def openpickle(filename):
    """Opens pickle file from /Store

    Parameters
    ----------
    filename : string
        The name of the file, with .pckl extension

    Returns
    -------
    [type]
        [description]
    """    
    path = sys.path[0] + '/Store/' + filename

    f = open(path, 'rb')
    file = pickle.load(f)
    f.close()

    return file

def comp(filename, data, type):
    """Compresses data and saves to file in /Store

    Parameters
    ----------
    filename : string
        filename of file to write to, ensure no file already exists
    data : any
        [description]
    type : [type]
        [description]
    """    

    if type == 0 or 'bz2':
        with bz2.BZ2File(filename, 'wb') as f:
            pickle.dump(data, f)
    elif type == 1 or 'lzma':
        with lzma.open(filename, "wb") as f:
            pickle.dump(data, f)  


# %% blosc

# tic1 = timer()

# pickled_data = pickle.dumps(video)  # returns data as a bytes object
# compressed_pickle = blosc.compress(pickled_data)

# with open("video1.dat", "wb") as f:
#     f.write(compressed_pickle)
# toc1 = timer()
# print(f'\n complete in {toc1-tic1: 0.1f}s')

