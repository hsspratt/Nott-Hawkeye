# %%
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# import pickle
# import bz2
# import lzma
import sys 
from PIL import Image, ImageFilter
# import skimage.morphology as morph
import functions as f
import importlib as imp
import pandas
import skimage.color as skm

# %%

def pics2video(f_names):
    path = sys.path[0] + '/Physics Pics/'
    nframes = np.shape(f_names)[0]
    frame0 = skm.rgb2gray(cv.imread(path+f_names[0]))
    video_size = np.hstack((np.shape(frame0)[0:2],nframes))
    video = np.zeros(video_size)
    video[:,:,0] = frame0

    for i in range(1, nframes):
        #save next image to video frame
        video[:,:,i] = skm.rgb2gray(cv.imread(path+f_names[i]))

    return video


def add_fix(list, prefix='', suffix=''):

    result = [prefix + sub + suffix for sub in list]

    return result


suffix = '.jpg'
prefix = 'IMG_0'

A_names = [str(sub) for sub in range(794, 802+1)]
A_names = add_fix(A_names, prefix, suffix)

A_video = pics2video(A_names)

prefix = 'IMG_'
B_names = [str(sub) for sub in range(3014, 3022+1)]
B_names = add_fix(B_names, prefix, suffix)

B_video = pics2video(B_names)


f.compare_frames(A_video, B_video, 5)


# %%

blur = f.gauss_blur(A_video)

plt.imshow(blur[:,:,1], cmap='gray')

# %%

f.export_bz2('test_photos1', (A_video, B_video))

# %%
