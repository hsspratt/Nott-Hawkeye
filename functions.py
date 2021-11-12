# %% functions 
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import skimage.color as skm
import pickle
import lzma
import bz2
from PIL import Image, ImageFilter
import sys
import os.path
import mpmath as mp
import skimage.morphology as morph


# video and image handling

def video_play(video):

    print('Playing video in separate window')
    tic3 = timer()
    nframes = np.shape(video)[2]
    for i in range(0,nframes):
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

def open_img(filename):
    """Opens an image file from Physics Pics folder. \n
    Physics Pics must be in the active folder with the python file.

    Parameters
    ----------
    filename : string
        Name of the file in /Physics Pics, include the file extension i.e. '.jpg'.

    Returns
    -------
    np.array
        NumPy array of the picture.
    """    
    
    path = sys.path[0] + '/Physics Pics/' + filename
    image = np.array(Image.open(path))
    return image

def video_read(filename_full):
    tic1 = timer()
    print('Loading video to variable...')
    path = sys.path[0] + '/Physics Pics/' + filename_full
    # capture video into array
    cap = cv.VideoCapture(path)
    fps = int(cap.get(5))
    nframes = int(cap.get(7))-1
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
    plt.title(f'Video frame: {nframes-50}')
    plt.show()

    toc1 = timer()
    print(f'\n video read complete in {toc1-tic1: 0.1f}s')

    return video


# compression imports and exports
def import_bz2(filename):    
    """Imports data from a bz2 compressed files in ~/Store folder. 
    The main Python file must be in folder containing the /Store folder.

    Parameters
    ----------
    filename : string
        Name of the file to import. filename does not need .pbz2 extension, as this will be added automatically.

    Returns
    -------
    any
        Returns the decompressed data stored in the file, in its uncompressed format.
    """    
    path = sys.path[0] + '/Store/' + filename + '.pbz2'
    if os.path.isfile(path):
        print(f'Importing {path}...')
        tic2 = timer()
        with bz2.BZ2File(path, 'rb') as f:
            file = pickle.load(f)
        toc2 = timer()
        print(f'\n import complete in {toc2-tic2: 0.1f}s')
    else:
        print('This file does not exist, check that filename has no extension or try a different filename.')

    return file

def import_lzma(filename):
    """Imports data from a lzma compressed files in ~/Store folder. 
    The main Python file must be in folder containing the /Store folder.

    Parameters
    ----------
    filename : string
        Name of the file to import. filename does not need .xz extension, as this will be added automatically.

    Returns
    -------
    any
        Returns the decompressed data stored in the file, in its uncompressed format.
    """    
    path = sys.path[0] + '/Store/' + filename + '.xz'
    if os.path.isfile(path):
        print(f'Importing {path}...')
        tic2 = timer()
        with lzma.open(path, 'rb') as f:
            file = pickle.load(f)
        toc2 = timer()
        print(f'\n import complete in {toc2-tic2: 0.1f}s')
    else:
        print('This file does not exist, check that filename has no extension or try a different filename.')

    return file

def export_bz2(filename, data):
    """Exports data, like NumPy arrays, to bz2 compressed files in ~/Store. Does not overwrite existing files. 
    The main Python file must be in folder containing the /Store folder.

    Parameters
    ----------
    filename : string
        Name of the file created. filename does not need .pbz2 extension as this will be added automatically.
    data : any
        Specify the variable you wish to write to a compressed file.
    """   
    path = sys.path[0] + '/Store/' + filename + '.pbz2'

    if os.path.isfile(path):
        print('This file already exists, choose a new filename or delete existing file.')
    else:
        print(f'Exporting to {path}...')
        tic = timer()
        with bz2.BZ2File(path, 'xb') as f:
            pickle.dump(data, f)
        toc = timer()
        print(f'completed export in {toc-tic: .1f}s')

def export_lzma(filename, data):
    """Exports data, like NumPy arrays, to lzma compressed files in ~/Store. Does not overwrite existing files. 
    The main Python file must be in folder containing the /Store folder.

    Parameters
    ----------
    filename : string
        Name of the file created. filename does not need .xz extension as this will be added automatically.
    data : any
        Specify the variable you wish to write to a compressed file.
    """    
    path = sys.path[0] + '/Store/' + filename + '.xz'
    if os.path.isfile(path):
        print('This file already exists, choose a new filename or delete existing file.')
    else:
        print(f'Exporting to {path}...')
        tic2 = timer()
        with lzma.open(path, "xb") as f:
            pickle.dump(data, f)
        toc2 = timer()
        print(f'\n export complete in {toc2-tic2: 0.1f}s')

# calibration functions

def calib_honing(img, x_calib, y_calib):
    img_size = np.shape(img)

    x_points = (240, x_calib, 240)
    y_points = (180, 180, y_calib)

    opacity = 0.8

    subplot, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,10))
    ax1.imshow(img);
    ax1.scatter(x_points, y_points, s=500, alpha=opacity, marker='x')
    ax1.axis([img_size[1]/2-40, img_size[1]/2+40, img_size[0]/2+40,img_size[0]/2-40])
    ax1.set_title('Centre point')
    ax2.imshow(img);
    ax2.scatter(x_points, y_points, s=500, alpha=opacity, marker='x')
    ax2.axis([0,80, 220,140])
    ax2.set_title('Centre left')
    ax3.imshow(img);
    ax3.scatter(x_points, y_points, s=500, alpha=opacity, marker='x')
    ax3.axis([200,280, 80,0])
    ax3.set_title('Top centre')
    subplot.tight_layout()
    plt.show(block=True)

    xy_calib = (x_calib, y_calib)

    return xy_calib

def calib_count(img, xy_calib):
    x_calib, y_calib = xy_calib

    img_size = np.shape(img)
    gray = skm.rgb2gray(img)
    thresh = gray>0.58

    x_points = (240, x_calib, 240)
    y_points = (180, 180, y_calib)

    opacity = 1

    fig = plt.figure(figsize=(10,10))
    # ax = plt.imshow(thresh, cmap='gray', alpha=0.5); 
    ax = plt.imshow(gray, cmap='gray'); 
    plt.scatter(x_points, y_points, s=200, alpha=opacity, marker='x', c='red')
    plt.title('Calibration image')
    plt.axis([0, img_size[1]/2+40, img_size[0]/2+40, 0])
    plt.show(block=True)

def calib_calc(xy_calib, z_dist):
    x_calib, y_calib = xy_calib
    # x_dist, y_dist, z_dist = dist
    x_dist, y_dist = (17, 13)

    x_points = (240, x_calib, 240)
    y_points = (180, 180, y_calib)

    theta_x = (mp.tan(x_dist/z_dist))
    x_pixels = x_points[0]-x_points[1]
    angle_pixel_x = float(theta_x/x_pixels)

    theta_y = (mp.tan(y_dist/z_dist))
    y_pixels = np.abs(y_points[0]-y_points[2])
    angle_pixel_y = float(theta_y/y_pixels)

    print(f'x-axis: {angle_pixel_x: 0.4f}rad/px' + f'   y-axis: {angle_pixel_y: 0.4f}rad/px')
    # print(f'For the y-axis: distance is {y_dist}cm; pixel distance is {y_pixels}px' + 
    #     f' and angle per pixel is {angle_pixel_y:0.4f}rad/px.')

    return angle_pixel_x, angle_pixel_y

def calib(image, z_dist, x_calib, y_calib):
    xy_calib = calib_honing(image, x_calib, y_calib)
    angle_pixel_xy = calib_calc(xy_calib, z_dist)

    return angle_pixel_xy


# image manipulation

def gauss_blur(image, radius):
    """Blurs an image using a gaussian blur filter with specified radius.

    Parameters
    ----------
    image : np.array
        NumPy array of an image in colour. 
    radius : float
        Specify the radius of the blur, the greater the radius, the greater the blur.

    Returns
    -------
    np.array
        Returns a NumPy array of the blurred image.
    """    
    im = Image.fromarray(np.uint8(image))
    blur = np.array(im.filter(ImageFilter.GaussianBlur(radius=radius)))

    return blur

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

def difference(video, background):
    
    nframes = np.shape(video)[2]

    # create background array for all frames
    video_0 = np.repeat(background, nframes, 2)

    # create difference image for all frames
    diff = np.abs(video-video_0)

    return diff

def threshold(array, threshold_value):
    normalised = array/np.max(array)
    thresholded = np.double(normalised>0.7)    

    return thresholded

# tracking

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

# visualising 

def visualise(video, centre_xy):
    tic = timer()
    # add color channels to plot coloured shapes on top
    color4 = np.repeat(np.expand_dims(video,2), 3, 2)

    nframes = np.shape(video)[2]
    visualised = np.repeat({},nframes,0)

    for i in range(0, nframes):
        im = video[:,:,i]
        color = color4[:,:,:,i]
        
        if not np.isnan(centre_xy[0,i]):
            centre =  (int(centre_xy[0,i]),int(centre_xy[1,i]))
            image = cv.circle(color, centre, 5, (0,0,255), 2)
            visualised[i] = image
        else:
            visualised[i] = im

        print(" ", end=f"\r frame: {i+1} ", flush=True)

    toc = timer()
    print(f'\n visualisation complete in {toc-tic: 0.1f}s')

    return visualised

def vis_player(visualised):
    tic = timer()
    nframes = np.shape(visualised)[0]
    for i in range(0, nframes):
        cv.imshow('', visualised[i])
        cv.waitKey(10)
    toc = timer()
    print(f'\n visualisation completed in {toc-tic: 0.1f}s')

    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(10)


# %% possible use for live images

def visualise_live(video, centre_xy):
    # add color channels to plot coloured shapes on top
    color4 = np.repeat(np.expand_dims(video,2), 3, 2)

    nframes = np.shape(video)[2]

    visualised = np.repeat({},nframes,0)

    tic = timer()
    for i in range(0, nframes):
        im = video[:,:,i]
        color = color4[:,:,:,i]
        
        if not np.isnan(centre_xy[0,i]):
            centre =  (int(centre_xy[0,i]),int(centre_xy[1,i]))
            image = cv.circle(color, centre, 5, (0,0,255), 2)
            visualised[i] = image
            cv.imshow('', image)
        else:
            cv.imshow('', im)
            visualised[i] = im
        cv.waitKey(10)

        print(" ", end=f"\r frame: {i+1} ", flush=True)
    toc = timer()
    print(f'\n video read complete in {toc-tic: 0.1f}s')
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(10)

    return visualised
