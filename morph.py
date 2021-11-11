# %% imports ~20s
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pickle
import bz2
import skimage.morphology as morph
import functions

f_name = "diff_thresh.pbz2"

diff_thresh = functions.import_bz2(f_name[0:-5])

# with bz2.BZ2File(f_name, 'rb') as f:
#     diff_thresh = pickle.load(f)

toc1 = timer()
print(f'completed import in {toc1-tic1: .1f}s')
# %%  ~48s for closing ~29s for binary_closing (100 frames) ~1m for all frames

str_el = morph.disk(8)
array_shape = np.shape(diff_thresh)
nframes = array_shape[2]
closed = np.zeros(array_shape, dtype=np.int8)

print(f'Total number of frames: {nframes}')

tic2 = timer()
for i in range(0, nframes):
    im = np.int8(diff_thresh[:,:,i])
    closed[:,:,i] = morph.binary_closing(im, str_el).astype(np.int8)

    print(" ", end=f"\r frame: {i+1} ", flush=True)

toc2 = timer()
print(f'closed masks in {toc2-tic2: .1f}s')

# %% compare with unclosed frame

i = 240
plt.figure()
plt.imshow(np.uint8(diff_thresh[:,:,i]), cmap='gray') 

plt.figure()
plt.imshow(closed[:,:,i], cmap='gray') 

# %%

f_name_export = "closed.pbz2"

functions.export_bz2(f_name_export[0:-5], closed)

# with bz2.BZ2File(f_name_export, 'wb') as f:
#     pickle.dump(closed, f)


# %%
functions.video_play(closed)

# %%
cv.namedWindow('')
# cv.imshow('', closed[:,:,300])
# %%
