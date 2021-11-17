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
    """Opens an image file from /Physics Pics folder. \n
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
    """Reads a video from the specified file in ~/Physics Pics. Converts to grayscale to reduce filesize.

    Parameters
    ----------
    filename_full : string
        The file name within /Physics Pics, include the file extension, i.e. '.mp4'.

    Returns
    -------
    np.array
        A 3d array of each grayscale frame, with the frame number along axis 2.
    """    
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
    else:
        # set up blank video array
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
        print(f'Importing {path}')
        tic2 = timer()
        with bz2.BZ2File(path, 'rb') as f:
            file = pickle.load(f)
        toc2 = timer()
        print(f'Import complete in {toc2-tic2:0.1f}s \n')
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
        print(f'Importing {path}')
        tic2 = timer()
        with lzma.open(path, 'rb') as f:
            file = pickle.load(f)
        toc2 = timer()
        print(f'Import complete in {toc2-tic2:0.1f}s \n')
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
def calib_honing(img, x_calib, y_calib, zoom=15):
    img_size = np.shape(img)
    x_centre = int(img_size[1]/2)
    y_centre = int(img_size[0]/2)

    x_points = (x_centre, x_calib, x_centre)
    y_points = (y_centre, y_centre, y_calib)

    opacity = 0.8
    width = int((1/zoom)*img_size[1])

    subplot, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,10))
    ax1.imshow(img);
    ax1.scatter(x_points, y_points, s=500, alpha=opacity, marker='x')
    ax1.axis([x_points[0]-width, x_points[0]+width, y_points[0]+width,y_points[0]-width])
    ax1.set_title('Centre point')
    ax2.imshow(img);
    ax2.scatter(x_points, y_points, s=500, alpha=opacity, marker='x')
    ax2.axis([0, (width*2), y_points[1]+width, y_points[1]-width])
    ax2.set_title('Centre left')
    ax3.imshow(img);
    ax3.scatter(x_points, y_points, s=500, alpha=opacity, marker='x')
    ax3.axis([x_points[0]-width, x_points[0]+width, (width*2), 0])
    ax3.set_title('Top centre')
    subplot.tight_layout()
    plt.show(block=True)

    xy_calib = (x_calib, y_calib)

    return xy_calib

def calib_count(img, xy_calib):
    x_calib, y_calib = xy_calib

    img_size = np.shape(img)
    x_centre = int(img_size[1]/2)
    y_centre = int(img_size[0]/2)
    gray = skm.rgb2gray(img)
    thresh = gray>0.58

    x_points = (x_centre, x_calib, x_centre)
    y_points = (y_centre, y_centre, y_calib)

    opacity = 1

    fig = plt.figure(figsize=(10,10))
    # ax = plt.imshow(thresh, cmap='gray', alpha=0.5); 
    plt.imshow(gray, cmap='gray'); 
    plt.scatter(x_points, y_points, s=200, alpha=opacity, marker='x', c='red')
    plt.plot([x_centre, x_centre], [0, img_size[0]-1])
    plt.title('Calibration image')
    # plt.axis([0, img_size[1]/2+width, img_size[0]/2+width, 0])
    plt.show(block=True)

def calib_calc(xy_calib, z_dist):
    x_calib, y_calib = xy_calib
    # x_dist, y_dist, z_dist = dist
    x_dist, y_dist = (17, 13)

    x_points = (280, x_calib, 280)
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

def calib(image, z_dist, x_calib, y_calib, zoom=15):
    xy_calib = calib_honing(image, x_calib, y_calib, zoom)
    angle_pixel_xy = calib_calc(xy_calib, z_dist)

    return angle_pixel_xy

# image manipulation
def gauss_blur1(image, radius):
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

def gauss_blur(array, radius=4):
    """Blurs an image or video using a gaussian blur filter with specified radius.

    Parameters
    ----------
    array : np.array
        NumPy array of an image (2d) or video (3d). For a video, the frame number must be in axis 2.
    radius : float (optional, default: 4)
        Specify the radius of the blur, the greater the radius, the greater the blur.

    Returns
    -------
    np.array
        Returns a NumPy array of the blurred image.
    """    
    if np.ndim(array) == 2:
        im = Image.fromarray(np.uint8(array*255))
        blur = np.array(im.filter(ImageFilter.GaussianBlur(radius=radius)))
        return blur
    elif np.ndim(array) == 3:
        nframes =  np.shape(array)[2]
        blur3 = np.zeros(np.shape(array))
        for i in range(0, nframes):
            im = Image.fromarray(np.uint8(array[:,:,i]*255))
            blur = np.array(im.filter(ImageFilter.GaussianBlur(radius=radius)))
            blur3[:,:,i] = blur
        return blur3
    else:
        print('check that the array has 2 or 3 dimensions')

def ref_image(video):
    mean = np.mean(video, 2)
    background = np.expand_dims(mean, 2)
    return background

def difference(video, background=np.nan):
    """Calculates the difference between all frames of a video and a reference image(background).

    Parameters
    ----------
    video : 3d array
        A 3 dimensional array with frame number along the second axis.
    background : 2d array (optional, defaults to initial video frame)
        An array with matching size to a frame. Could be a clear frame from the video.

    Returns
    -------
    np.array 
        An array of the difference to the background for each frame, with the same shape as video.
    """    
    if np.ndim(video) == 3: 
        
        if [background.all] == [np.nan]:
            background = video[:,:,0]
        
        nframes = np.shape(video)[2]
        # create background array for all frames
        video_0 = np.repeat(background, nframes, 2)

        # create difference image for all frames
        diff = np.array(np.abs(video-video_0))
        return diff
    else:
        print('Error: check video dimensions')

def threshold(array, threshold_value):
    normalised = array/np.max(array)
    thresholded = np.double(normalised>threshold_value)    

    return thresholded

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
    normalise = video/np.max(video)
    color4 = np.repeat(np.expand_dims(normalise,2), 3, 2)
    # color4 = np.zeros(np.hstack((np.shape(video),3)))

    nframes = np.shape(video)[2]
    visualised = np.repeat({},nframes,0)

    for i in range(0, nframes):
        im = video[:,:,i]
        color = color4[:,:,:,i]
        print(np.shape(color4))
        
        if not np.isnan(centre_xy[0,i]): # check the object is in frame
            centre =  (int(centre_xy[0,i]),int(centre_xy[1,i]))
            image = cv.circle(color, centre, 5, (0,0,255), 2)
            visualised[i] = image
        else:
            visualised[i] = im
        plt.imshow(visualised[i])
        print(" ", end=f"\r frame: {i+1} ", flush=True)
        

    toc = timer()
    print(f'\n visualisation complete in {toc-tic: 0.1f}s')

    return visualised

def vis_player(visualised, wait=10):
    tic = timer()
    nframes = np.shape(visualised)[0]
    for i in range(0, nframes):
        cv.imshow('', visualised[i])
        cv.waitKey(wait)
    toc = timer()
    print(f'\n visualisation completed in {toc-tic: 0.1f}s')

    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(10)

def max_frame(array):
    """Finds frame with largest mean value from 3d array. \n
    Useful for automatically showing a frame of interest, as it should find the frame with the largest object.

    Parameters
    ----------
    array : np.array
        Array of image frames with 3 dimensions, where the third dimension is the frame number.

    Returns
    -------
    integer
        The index of the frame with the largest mean value.
    """    
    if np.ndim(array) == 3:
        sum = np.sum(array, axis=(0,1))
        index = int(np.where(sum==np.max(sum))[0])
    else:
        print('Array must have exactly 3 dimensions, with frame number as the third.')

    return index

def min_frame(array):
    """Finds frame with smallest mean value from 3d array. \n
    Useful for automatically showing a frame of interest, as it should find the frame with the largest object.

    Parameters
    ----------
    array : np.array
        Array of image frames with 3 dimensions, where the third dimension is the frame number.

    Returns
    -------
    integer
        The index of the frame with the smallest mean value.
    """    
    if np.ndim(array) == 3:
        sum = np.sum(array, axis=(0,1))
        index = int(np.where(sum==np.min(sum))[0])
    else:
        print('Array must have exactly 3 dimensions, with frame number as the third.')

    return index

def compare_frames(plot1, plot2, frame, s_title=np.nan, axis='off', figsize=(10,4)):

    plot1 = plot1[:,:,frame]
    plot2 = plot2[:,:,frame]
    
    fig, ax = plt.subplots(1,2, figsize=figsize)
    
    if not [s_title] == [np.nan]:
        fig.suptitle(s_title, fontsize=20)

    ax[0].imshow(plot1, cmap='gray')
    ax[0].axis(axis)
    ax[0].set_title(f'Camera A, frame: {frame}')

    ax[1].imshow(plot2, cmap='gray')
    ax[1].axis(axis)
    ax[1].set_title(f'Camera B, frame: {frame}')
    plt.show()

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
