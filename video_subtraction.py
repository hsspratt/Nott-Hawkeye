#%% load modules
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# import skimage.color as skm
import pickle

f_name = 'video1.pckl'

f = open(f_name, 'rb')
video = pickle.load(f)
f.close()

nframes = np.shape(video)[2]

toc1 = timer()
print(f'completed import in {toc1-tic1: .1f}s')

# %%
tic2 = timer()

print(f'number of frames: {nframes}')

# video_size = np.hstack((np.shape(frame)[0:2],nframes))
video_size = np.shape(video)
diff = np.zeros(video_size)

for i in range(0, nframes-1):
    #  take difference image
    diff[:,:,i] = np.abs(video[:,:,i]-video[:,:,0])
    cv.imshow('video', diff[:,:,i])
    cv.waitKey(10)

    print(" ", end=f"\r frame: {i+1} ", flush=True)

toc2 = timer()
print(f'completed difference image in {toc2-tic2: .1f}s')

cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)

# %%
tic3 = timer()
#  normalise and threshold
diff1 = diff/np.max(diff)
diff1 = np.double(diff1>0.6)

for i in range(0, nframes-1):
    cv.imshow('video1', diff1[:,:,i])
    cv.waitKey(10)

    print(" ", end=f"\r frame: {i+1} ", flush=True)

toc3 = timer()
print(f'completed in {toc3-tic3: .1f}s')

cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)

# %%

plt.figure()
plt.imshow(diff1[:,:,300], cmap='gray'); plt.axis('off')

# %%
