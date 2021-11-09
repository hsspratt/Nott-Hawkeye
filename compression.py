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

# open file
f_name = 'video1'

f = open(f_name+'.pckl', 'rb')
video = pickle.load(f)
f.close()

with lzma.open(''+'xz', 'rb') as f:
    video = pickle.load(f)

tic1 = timer()

# %% bz2

# save video array to compressed file 
with bz2.BZ2File(f_name, 'wb') as f:
    pickle.dump(video, f)

toc1 = timer()

print(f'\n complete in {toc1-tic1: 0.1f}s')

# %% lzma

tic1 = timer()
with lzma.open("lmza_test1.xz", "wb") as f:
    pickle.dump(video, f)
toc1 = timer()
print(f'\n complete in {toc1-tic1: 0.1f}s')

# %% blosc

tic1 = timer()

pickled_data = pickle.dumps(video)  # returns data as a bytes object
compressed_pickle = blosc.compress(pickled_data)

with open("video1.dat", "wb") as f:
    f.write(compressed_pickle)
toc1 = timer()
print(f'\n complete in {toc1-tic1: 0.1f}s')

