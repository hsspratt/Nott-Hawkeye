# %% functions 
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import skimage.color as skm
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

def open_img(f_name):
    """Opens an image file from physics pics folder

    Parameters
    ----------
    f_name : string
        Name of the file

    Returns
    -------
    np.array
        numpy array of the picture
    """    
    path = sys.path[0] + '/Physics Pics/' + f_name
    img = np.array(Image.open(path))
    return img


# specify file to read from:
f_cap = 'IMG_0599.mp4'

# specify filename to save to 
f_name = 'video1'

def video_read(filename):
    
    # capture video into array
    cap = cv.VideoCapture(f_cap)
    fps = int(cap.get(5))
    nframes = int(cap.get(7))
    print(f'number of frames: {nframes}')

    if (cap.isOpened() == False):
        print('Error opening video')

    ret, frame = cap.read()

    n = 0
    video_size = np.hstack((np.shape(frame)[0:2],nframes))
    video = np.zeros(video_size)

    while (cap.isOpened()):
        # capture frame by frame
        ret, frame = cap.read()
        if ret == True:
            # save grayscale image
            video[:,:,n] = skm.rgb2gray(frame)

            n+=1
            print(" ", end=f"\r frame: {n} ", flush=True)

        else:
            break

    cap.release()

    plt.figure()
    plot1 = plt.imshow(video[:,:,nframes-50], cmap='gray')
    plt.axis('off')

    toc1 = timer()
    print(f'\n video read complete in {toc1-tic1: 0.1f}s')

    return video