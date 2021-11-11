#%% load modules ~1s
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import skimage.color as skm
import pickle
# import bz2
import lzma
import os.path

def export(f_name_full):
    with lzma.open(f_name_full, "wb") as f:
        pickle.dump(video, f)
        

# record video into array ~10s

def vid2array(f_cap):

    # capture video into array
    cap = cv.VideoCapture(f_cap)
    fps = int(cap.get(5))
    nframes = int(cap.get(7))
    print(f'number of frames: {nframes}')

    if (cap.isOpened() == False):
        print('Error opening video')

    ret, frame = cap.read()

    plt.figure()
    image1 = skm.rgb2gray(frame)
    plt.imshow(image1, cmap='gray')
    plt.axis('off')

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

    return video, nframes

# specify file to read from:
f_cap = 'IMG_0599.mp4'

# specify filename to save to 
f_name = 'video1'

video, nframes = vid2array(f_cap)

plt.figure()
plot1 = plt.imshow(video[:,:,nframes-50], cmap='gray')
plt.axis('off')

toc1 = timer()
print(f'\n complete in {toc1-tic1: 0.1f}s')

# %% export video to compressed .xy file ~145s

f_name_full = f_name+'.xz'

if os.path.isfile(f_name_full):
    print('This file already exists')
else:
    tic2 = timer()
    export(f_name_full)
    toc2 = timer()
    print(f'\n export complete in {toc2-tic2: 0.1f}s')



# %%

