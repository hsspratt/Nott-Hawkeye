#%% load modules
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import pickle
import bz2
import cv2 as cv
import sys
import lzma

def decomp(filename, type):
    """Loads a compressed file from /Store

    Parameters
    ----------
    f_name : string
        Filename of compressed file in /Store, include file ending
    type : int
        0 = bz2 \n
        1 = lzma
    """    
    path = sys.path[0] + '/Store/' + filename
    if type == 0 or 'bz2':
        with bz2.BZ2File(path, 'rb') as f:
            file = pickle.load(f)
    elif type == 1 or 'lzma': 
        with lzma.open(path, 'rb') as f:
            file = pickle.load(f)
    else:
        print("ERROR: type must be bz2(0) or lzma(1)")
    
    return file


f_name_load = "closed.pbz2"

diff_thresh = decomp(f_name_load, 0)

toc1 = timer()
print(f'completed import in {toc1-tic1: .1f}s')
# %%
tic2 = timer()

array_shape = np.shape(diff_thresh)
nframes = array_shape[2]
x_av = np.ndarray.flatten(np.zeros((1,nframes)))
y_av = np.ndarray.flatten(np.zeros((1,nframes)))

for i in range(0, nframes):
    im = diff_thresh[:,:,i]

    indices = np.where(im==1)

    l,coords = np.unravel_index(indices, np.shape(im))
    y = coords[0,:]
    x = coords[1,:]

    x_av[i] = np.around(np.mean(x))
    y_av[i] = np.around(np.mean(y))

toc2 = timer()
print(f'completed in {toc2-tic2: .1f}s')

# %%

i = 220
plt.figure()
plt.imshow(diff_thresh[:,:,i], cmap='gray'); plt.axis('off')
plt.scatter(x_av[i],y_av[i], color='r', s=10)



# cv.imshow('video', diff_thresh[:,:,i])

# cv.waitKey(0)
# cv.destroyAllWindows()
# cv.waitKey(1)

# %%

# index = np.ravel_multi_index((y_av[i],x_av[i]), np.shape(im))
# print(index)

# %%
plt.figure()
plt.plot(x_av, y_av)
plt.axis([0, array_shape[1], 0, array_shape[0]]);


# %%

tic3 = timer()

for i in range(0,nframes-1):
    cv.imshow('video', diff_thresh[:,:,i])
    cv.waitKey(10)

    print(" ", end=f"\r frame: {i+1} ", flush=True)

toc3 = timer()

print(f'plotted in {toc3-tic3: 0.1f}s, press any key to close window and continue')

cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)

print('window closed')

# %%
import functions

functions.play_video(diff_thresh)
# %%
