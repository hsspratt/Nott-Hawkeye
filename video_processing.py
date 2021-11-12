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

# %% difference image

def difference(video, background):
    
    nframes = np.shape(video)[2]

    # create background array for all frames
    video_0 = np.repeat(background, nframes, 2)

    # create difference image for all frames
    diff = np.abs(video-video_0)

    return diff

diff = difference(video, video[:,:,0:1])

plt.imshow(diff[:,:,-1], cmap='gray')

# functions.video_play(diff)

# %% thresholded frames
def threshold(array, threshold_value):
    normalised = array/np.max(array)
    thresholded = np.double(normalised>0.7)    

    return thresholded



# diff_thresh = diff/np.max(diff)
# diff_thresh = np.double(diff_thresh>0.7)

diff_thresh = threshold(diff, 0.7)

# functions.video_play(diff_thresh)

plt.imshow(diff_thresh[:,:,100], cmap='gray')


# %% morphing, closing
def closing_disk(array, radius):
    str_el = morph.disk(radius)
    array_shape = np.shape(array)
    nframes = array_shape[2]
    closed = np.zeros(array_shape, dtype=np.int8)

    print(f'Total number of frames: {nframes}')

    tic2 = timer()
    for i in range(0, nframes):
        im = np.int8(array[:,:,i])
        closed[:,:,i] = morph.binary_closing(im, str_el).astype(np.uint8)

        print(" ", end=f"\r frame: {i+1} ", flush=True)

    toc2 = timer()
    print(f'\n closed masks in {toc2-tic2: .1f}s')

    closed = np.ones(array_shape)*closed

    return closed

closed = closing_disk(diff_thresh, 8)

plt.imshow(closed[:,:,105], cmap='gray')

# functions.video_play(closed)

# %% centre points

def centre_points(array):
    tic2 = timer()

    array_shape = np.shape(array)
    nframes = array_shape[2]
    x_av = np.ndarray.flatten(np.zeros((1,nframes)))
    y_av = np.ndarray.flatten(np.zeros((1,nframes)))

    for i in range(0, nframes):
        im = array[:,:,i]

        # only calculates position if an object within the frame
        if not np.sum(im) == 0:
            indices = np.where(im==1)

            l,coords = np.unravel_index(indices, np.shape(im))
            y = coords[0,:]
            x = coords[1,:]

            x_av[i] = np.around(np.mean(x))
            y_av[i] = np.around(np.mean(y))

        else:
            x_av[i] = np.nan
            y_av[i] = np.nan

    toc2 = timer()
    print(f'centre_points completed in {toc2-tic2: .1f}s')

    return np.array([x_av, y_av])

centre_xy = centre_points(closed)


#%%

i = 100
plt.figure()
plt.imshow(closed[:,:,i], cmap='gray'); plt.axis('off')
plt.scatter(centre_xy[0,i],centre_xy[1,i], color='r', s=10)


# %%

# add color channels to plot coloured shapes on top
color4 = np.repeat(np.expand_dims(closed,2), 3, 2)

nframes = np.shape(closed)[2]

image_save = np.repeat({},nframes,0)

tic = timer()
for i in range(0, nframes):
    im = closed[:,:,i]
    color = color4[:,:,:,i]
    
    if not np.isnan(centre_xy[0,i]):
        centre =  (int(centre_xy[0,i]),int(centre_xy[1,i]))
        image = cv.circle(color, centre, 5, (0,0,255), 2)
        image_save[i] = image
        cv.imshow('', image)
    else:
        cv.imshow('', im)
        image_save[i] = im
    cv.waitKey(10)

    print(" ", end=f"\r frame: {i+1} ", flush=True)
toc = timer()
print(f'\n video read complete in {toc-tic: 0.1f}s')
cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(10)

def visualise(video, centre_xy):
    # add color channels to plot coloured shapes on top
    color4 = np.repeat(np.expand_dims(video,2), 3, 2)

    nframes = np.shape(video)[2]

    image_save = np.repeat({},nframes,0)

    tic = timer()
    for i in range(0, nframes):
        im = video[:,:,i]
        color = color4[:,:,:,i]
        
        if not np.isnan(centre_xy[0,i]):
            centre =  (int(centre_xy[0,i]),int(centre_xy[1,i]))
            image = cv.circle(color, centre, 5, (0,0,255), 2)
            image_save[i] = image
            cv.imshow('', image)
        else:
            cv.imshow('', im)
            image_save[i] = im
        cv.waitKey(10)

        print(" ", end=f"\r frame: {i+1} ", flush=True)
    toc = timer()
    print(f'\n video read complete in {toc-tic: 0.1f}s')
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(10)

    return 


# %%
tic = timer()
for i in range(0, nframes):
    cv.imshow('', image_save[i])
    cv.waitKey(10)
toc = timer()
print(f'\n video read complete in {toc-tic: 0.1f}s')

cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(10)


# %%
