#%% load modules
from timeit import default_timer as timer

from numpy.lib.function_base import corrcoef
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pickle
import bz2
import lzma

f_name = "lmza_test1.xz"

with lzma.open(f_name, 'rb') as f:
    video = pickle.load(f)

nframes = np.shape(video)[2]

toc1 = timer()
print(f'completed import in {toc1-tic1: .1f}s')

# %%
tic2 = timer()

print(f'number of frames: {nframes}')

# video_size = np.hstack((np.shape(frame)[0:2],nframes))
video_size = np.shape(video)
diff = np.zeros(video_size)

for i in range(0, nframes):
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
diff_thresh = diff/np.max(diff)
diff_thresh = np.double(diff_thresh>0.6)

for i in range(0, nframes):
    cv.imshow('video1', diff_thresh[:,:,i])
    cv.waitKey(10)

    print(" ", end=f"\r frame: {i+1} ", flush=True)

toc3 = timer()
print(f'completed in {toc3-tic3: .1f}s')

cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(10)

plt.figure()
plt.imshow(diff_thresh[:,:,300], cmap='gray'); plt.axis('off')

# %% ~28s
tic = timer()

with bz2.BZ2File('diff_thresh.pbz2', 'xb') as f:
    pickle.dump(diff_thresh, f)

toc = timer()
print(f'completed in {toc-tic: .1f}s')
# %%
