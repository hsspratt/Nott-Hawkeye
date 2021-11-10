# %% imports ~20s
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pickle
import bz2
import skimage.morphology as morph

f_name = "diff_thresh.pbz2"

with bz2.BZ2File(f_name, 'rb') as f:
    diff_thresh = pickle.load(f)

toc1 = timer()
print(f'completed import in {toc1-tic1: .1f}s')
# %%
str_el = morph.disk(8)
array_shape = np.shape(diff_thresh)

# for i in range 
im = diff_thresh[:,:,300]
opened = morph.closing(im, str_el)

plt.figure()
plt.imshow(im, cmap='gray') 

plt.figure()
plt.imshow(opened, cmap='gray') 

# %%
