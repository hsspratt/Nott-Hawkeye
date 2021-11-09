#%% load modules
from timeit import default_timer as timer

from numpy.lib.function_base import corrcoef
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pickle
import bz2

f_name = "diff_thresh.pbz2"

with bz2.BZ2File(f_name, 'rb') as f:
    diff_thresh = pickle.load(f)

toc1 = timer()
print(f'completed import in {toc1-tic1: .1f}s')
# %%

plt.figure()
plt.imshow(diff_thresh[:,:,300], cmap='gray'); plt.axis('off')

im1 = diff_thresh[:,:,300]

indices = np.where(im1==1)
no = np.count_nonzero(indices)
print(no)

coord = np.unravel_index(17, (6, 3))
print(coord)
index = np.ravel_multi_index(coord, (6,3))
print(index)
