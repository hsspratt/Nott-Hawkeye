#%%
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# import pickle
# import bz2
# import lzma
# import sys 
from PIL import Image, ImageFilter
# import skimage.morphology as morph
import functions
import importlib as imp


# %% 
# import the video 
imp.reload(functions)
video = functions.video_read('IMG_0601.mp4')


index = functions.min_frame(video)
plt.figure()
plot1 = plt.imshow(video[:,:,index], cmap='gray')
plt.axis('off')
plt.title(f'Video frame: {index}')
plt.show()

# %% gaussian blur
video_gauss = functions.gauss_blur(video[:,:,:], 4.2)

# sum_arr = np.sum(video_gauss, 2)
plt.imshow(video_gauss[:,:,1], cmap='gray')

# %% subtraction method
def ref_image(video):
    mean = np.mean(video, 2)
    background = np.expand_dims(mean, 2)
    return background

# background = video[:,:,0:1] # choose a reference image
background = ref_image(video) # meaning to create background

plt.imshow(background, cmap='gray')

diff = functions.difference(video, background)

index = functions.max_frame(diff)
plt.figure()
plt.imshow(diff[:,:,index], cmap='gray'); plt.axis('off')
plt.title(f'Absolute difference, frame: {index}')
plt.show()

# %% thresholded frames

diff_thresh = functions.threshold(diff, 0.7)

index = functions.max_frame(diff_thresh)
plt.imshow(diff_thresh[:,:,index], cmap='gray')
plt.title(f'Difference image, frame: {index}');

# %% morphing, closing

closed = functions.closing_disk(diff_thresh, 8)

plt.imshow(closed[:,:,index], cmap='gray'); plt.axis('off')



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
Aang_y = (centre_xy[1,:]-180)*angles[0,1]
# %%

functions.export_bz2('A_angles_test', (Aang_x, Aang_y))

# %%
