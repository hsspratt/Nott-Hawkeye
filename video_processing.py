#%%
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# import pickle
# import bz2
# import lzma
# import sys 
# from PIL import Image
# import skimage.morphology as morph
import functions
import importlib as imp
imp.reload(functions)

# %%

video = functions.video_read('IMG_0601.mp4')

plt.imshow(video[:,:,-1], cmap='gray'); plt.axis('off')

# %% difference image

diff = functions.difference(video, video[:,:,0:1])

plt.imshow(diff[:,:,-1], cmap='gray'); plt.axis('off')

# functions.video_play(diff)

# %% thresholded frames

diff_thresh = functions.threshold(diff, 0.7)

# functions.video_play(diff_thresh)

plt.imshow(diff_thresh[:,:,100], cmap='gray')

# %% morphing, closing

closed = functions.closing_disk(diff_thresh, 8)

plt.imshow(closed[:,:,105], cmap='gray'); plt.axis('off')

# functions.video_play(closed)

# %% centre points

centre_xy = functions.centre_points(closed)

i = 110
plt.figure()
plt.imshow(closed[:,:,i], cmap='gray'); plt.axis('off')
plt.scatter(centre_xy[0,i],centre_xy[1,i], color='r', s=10)


# %% visualisation

image_save = functions.visualise(closed, centre_xy)

# %% vis player

functions.vis_player(image_save)


# %%

angles = functions.import_bz2('angles')

# %%

i = 110
plt.figure()
plt.imshow(closed[:,:,i], cmap='gray'); plt.axis('off')
plt.scatter(centre_xy[0,i],centre_xy[1,i], color='r', s=10)

Aang_x = (centre_xy[0,:]-240)*angles[0,0]
Aang_y = (centre_xy[1,:]-240)*angles[0,1]
# %%

functions.export_bz2('A_angles_test', (Aang_x, Aang_y))

# %%
