#%% load and video ~15s 
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# import skimage.color as skm
import pickle
import lzma
from PIL import Image, ImageFilter

# %%

# choose file to read
f_name = 'video1.xz'

with lzma.open(f_name, 'rb') as f:
    video = pickle.load(f)

nframes = np.shape(video)[2]

toc1 = timer()

print(f'completed import in {toc1-tic1: .1f}s')

fig1 = plt.figure(figsize=(9,9))
plot1 = plt.imshow(video[:,:,300], cmap='gray'); 
plt.axis('off')


# %%
tic3 = timer()

for i in range(0,nframes-1):
    cv.imshow('video', video[:,:,i])
    cv.waitKey(10)

    print(" ", end=f"\r frame: {i+1} ", flush=True)

toc3 = timer()

print(f'plotted in {toc3-tic3: 0.1f}s, press any key to close window and continue')

cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)

print('window closed')


# %% gaussian blur

diff = np.abs(video[:,:,250]-video[:,:,1])
im = Image.fromarray(np.uint8(diff*255))
blur = im.filter(ImageFilter.GaussianBlur(radius=4))


plt.imshow(blur, cmap='gray')


# %%
