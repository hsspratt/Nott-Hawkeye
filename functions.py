# %% functions 
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import skimage.color as skm
import pickle
import lzma
from PIL import Image, ImageFilter
# import compression 
import sys
import os.path
import mpmath as mp


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

def video_read(file_path):
    tic1 = timer()
    print('Loading video to variable...')
    # capture video into array
    cap = cv.VideoCapture(file_path)
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

def export_lzma(filename, data):
    """Exports data, like NumPy arrays, to lzma compressed files in ~/Store. Does not overwrite existing files. 
    Python file must be in folder containing /Store folder.

    Parameters
    ----------
    filename : string
        Name the file you wish to create, does not need .xz extension, this will be added automatically.
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
    ax = plt.imshow(thresh, cmap='gray', alpha=0.5); 
    plt.scatter(x_points, y_points, s=200, alpha=opacity, marker='x', c='red')
    plt.title('Calibration image')
    plt.axis([0, img_size[1]/2+40, img_size[0]/2+40, 0])
    plt.show(block=True)

def calib_calc(xy_calib, dist):
    x_calib, y_calib = xy_calib
    x_dist, y_dist, z_dist = dist

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


# %%
