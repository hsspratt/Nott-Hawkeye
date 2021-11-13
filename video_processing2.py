#%%
from __future__ import all_feature_names
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

# %% takes approximatly half the length of the video to decompress

# A_video = functions.import_lzma('cameraA_test1')
# B_video = functions.import_lzma('cameraB_test1')
# %%

index = functions.max_frame(A_video)

# fig, ax = plt.subplots(1,2, figsize=(10,10))
# ax[0].imshow(A_video[:,:,index], cmap='gray')
# ax[0].axis('off')
# ax[0].set_title(f'Camera A, frame: {index}')
# ax[1].imshow(B_video[:,:,index], cmap='gray')
# ax[1].axis('off')
# ax[1].set_title(f'Camera B, frame: {index}')
# plt.show()

def compare(plot1, plot2, frame=np.nan, s_title=np.nan, figsize=(10,4)):

    fig, ax = plt.subplots(1,2, figsize=figsize)
    
    if not [s_title] == [np.nan]:
        fig.suptitle(s_title, fontsize=20)

    if not [frame] == [np.nan]:
        plot1 = plot1[:,:,frame]
        plot2 = plot2[:,:,frame]

    ax[0].imshow(plot1, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title(f'Camera A, frame: {index}')

    ax[1].imshow(plot2, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title(f'Camera B, frame: {index}')
    plt.show()

def frame(A, B, frame):
    pass

compare(A_video[:,:,index], B_video[:,:,index], 'Video')

# %%
imp.reload(functions)
A_background = functions.ref_image(A_video)




# %%
