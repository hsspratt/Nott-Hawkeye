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
import sys 
import functions

f_name = "video1.xz"
path = sys.path[0] + '/Store/' + f_name

with lzma.open(path, 'rb') as f:
    video = pickle.load(f)

nframes = np.shape(video)[2]-1

toc1 = timer()
print(f'completed import in {toc1-tic1: .1f}s')

# %%
tic2 = timer()

print(f'number of frames: {nframes}')

video_size = np.hstack((np.shape(video[:,:,1])[0:2],nframes))
# video_size = np.shape(video)
diff = np.zeros(video_size)

for i in range(0, nframes):
    #  take difference image
    diff[:,:,i] = np.abs(video[:,:,i]-video[:,:,0])
    cv.imshow('video', diff[:,:,i])
    cv.waitKey(10)

    print(" ", end=f"\r frame: {i+1} ", flush=True)

toc2 = timer()
print(f'completed difference images in {toc2-tic2: .1f}s')

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

# %%

plt.figure()
plt.imshow(diff_thresh[:,:,-1], cmap='gray'); plt.axis('off')

# %% ~28s
tic = timer()
f_name_export = 'diff_thresh1.pbz2'
path = sys.path[0] + '/Store/' + f_name_export
with bz2.BZ2File(path, 'xb') as f:
    pickle.dump(diff_thresh, f)

toc = timer()
print(f'completed in {toc-tic: .1f}s')
# %%
import functions

functions.video_play(video)