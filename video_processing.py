#%%
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pickle
import bz2
import lzma
import sys 
import skimage.morphology as morph
import functions
from PIL import Image


# %%


video = functions.video_read('IMG_0601.mp4')

# %%

plt.imshow(video[:,:,-1], cmap='gray')

# %%

nframes = np.shape(video)[2]
tic2 = timer()
video_size = np.hstack((np.shape(video[:,:,1])[0:2],nframes))
# video_size = np.shape(video)
diff = np.zeros(video_size)


# %%

background = video[:,:,0:1]

video_0 = np.repeat(background, nframes, 2)

diff = np.abs(video-video_0)


# %%
functions.video_play(diff)

# %%

diff_thresh = diff/np.max(diff)
diff_thresh = np.double(diff_thresh>0.7)

functions.video_play(diff_thresh)

# %%

plt.imshow(diff_thresh[:,:,100], cmap='gray')


# %% morphing

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
print(f'\n closed masks in {toc2-tic2: .1f}s')

#%%
plt.imshow(closed[:,:,100], cmap='gray')


# %%
# functions.video_play(closed)

# im = Image.fromarray(closed)
# im = np.float16(closed)
im = closed*255
im100 = closed[:,:,100]*255

# plt.imshow(im[:,:,100], cmap='gray')

cv.imshow('', im[:,:,100])
# %%

np.min(im)
# %%
