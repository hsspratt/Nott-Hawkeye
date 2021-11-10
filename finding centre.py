#%% load modules
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pickle
import bz2

f_name = "diff_thresh.pbz2"

with bz2.BZ2File(f_name, 'rb') as f:
    diff_thresh = pickle.load(f)

toc1 = timer()
print(f'completed import in {toc1-tic1: .1f}s')
# %%
tic2 = timer()

array_shape = np.shape(diff_thresh)
x_av = np.ndarray.flatten(np.zeros((1,array_shape[2])))
y_av = np.ndarray.flatten(np.zeros((1,array_shape[2])))

for i in range(0, array_shape[2]):
    im = diff_thresh[:,:,i]

    indices = np.where(im==1)

    l,coords = np.unravel_index(indices, np.shape(im))
    y = coords[0,:]
    x = coords[1,:]

    x_av[i] = np.around(np.mean(x))
    y_av[i] = np.around(np.mean(y))


no = np.count_nonzero(indices)
# print(no)

# %%

i = 391
plt.figure()
plt.imshow(diff_thresh[:,:,i], cmap='gray'); plt.axis('off')
plt.scatter(y_av[i],x_av[i], color='r', s=10)

toc2 = timer()
print(f'completed in {toc2-tic2: .1f}s')

# %%

index = np.ravel_multi_index((y_av[i],x_av[i]), np.shape(im))
print(index)

# %%
plt.figure()
plt.plot(x_av, y_av)
plt.axis([0, array_shape[1], 0, array_shape[0]]);


# %%
