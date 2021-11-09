#%% load modules
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# import skimage.color as skm
import pickle
import lzma

# choose file to read
f_name = 'video1.pckl'

with lzma.open(f_name, 'rb') as f:
    video = pickle.load(f)

nframes = np.shape(video)[2]

toc1 = timer()

print(f'completed import in {toc1-tic1: .1f}s')

# %%

tic2 = timer()

fig1 = plt.figure(figsize=(9,9))
plot1 = plt.imshow(video[:,:,300], cmap='gray'); 
plt.axis('off')

toc2 = timer()

print(f'completed plot in {toc2-tic2: .1f}s')


# %%
tic3 = timer()

for i in range(0,nframes-1):
    cv.imshow('video', video[:,:,i])
    cv.waitKey(10)

    print(" ", end=f"\r frame: {i+1} ", flush=True)

toc3 = timer()

print(f'plotted in {toc3-tic3: 0.1f}s, press any key to close window and continue')

# check that last frame was plotted
# cv.namedWindow('last video frame')
# cv.moveWindow('last video frame', 200,100) 
# cv.imshow('last video frame', video[:,:,i])

cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)

print('windows closed')


# %%

diff = video[:,:,2]-video[:,:,1]

plt.imshow(diff, cmap='gray')
# %%
