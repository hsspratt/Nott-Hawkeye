# %% functions 
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# import skimage.color as skm
import pickle
import lzma
from PIL import Image, ImageFilter
import compression 
import sys

def play_video(video):

    tic3 = timer()
    nframes = np.shape(video)[2]
    for i in range(0,nframes-1):
        cv.imshow('video', video[:,:,i])
        cv.waitKey(10)

        print(" ", end=f"\r frame: {i+1} ", flush=True)

    toc3 = timer()

    print(f'plotted in {toc3-tic3: 0.1f}s, press any key to close window and continue')

    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)
    
    print('window closed')
    
    return

