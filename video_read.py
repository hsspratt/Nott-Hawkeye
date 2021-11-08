#%% load modules
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import skimage.color as skm
import pickle

# %%
# record video into array

# specify file to read from 
# f_cap = 'cars 10s 1.mp4'
f_cap = 'IMG_0599.mp4'

#specify filename
f_name = 'video1.pckl'

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

n = 0
video_size = np.hstack((np.shape(frame)[0:2],nframes))
video = np.zeros(video_size)

while (cap.isOpened()):
    # capture frame by frame
    ret, frame = cap.read()
    if ret == True:

        video[:,:,n] = skm.rgb2gray(frame)

        n+=1
        print(" ", end=f"\r frame: {n} ", flush=True)


    else:
        break

cap.release()

# %%

plt.figure()
plot1 = plt.imshow(video[:,:,nframes-100], cmap='gray')
plot1.axis = 'off'

f = open(f_name, 'xb')
pickle.dump(video, f)
f.close()

toc1 = timer()

print(f'\n complete in {toc1-tic1: 0.1f}s')

# %%
