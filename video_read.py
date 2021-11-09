#%% load modules ~10s
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import skimage.color as skm
import pickle
import bz2

# %% record video into array 

# specify file to read from:
    # f_cap = 'cars 10s 1.mp4'
f_cap = 'IMG_0599.mp4'

# specify filename to save to 
f_name = 'video1.pbz2'

# capture video into array
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
        # save grayscale image
        video[:,:,n] = skm.rgb2gray(frame)

        n+=1
        print(" ", end=f"\r frame: {n} ", flush=True)


    else:
        break

cap.release()

plt.figure()
plot1 = plt.imshow(video[:,:,nframes-50], cmap='gray')
plot1.axis = 'off'

# %% 
tic1 = timer()

# save video array to compressed file 
with bz2.BZ2File(f_name, 'wb') as f:
    pickle.dump(video, f)

toc1 = timer()

print(f'\n complete in {toc1-tic1: 0.1f}s')

# %%
import lzma
tic1 = timer()
with lzma.open("lmza_test1.xz", "wb") as f:
    pickle.dump(video, f)
toc1 = timer()
print(f'\n complete in {toc1-tic1: 0.1f}s')

# %%
import blosc
tic1 = timer()
# with blosc.open("lmza_test1.xz", "wb") as f:
#     pickle.dump(video, f)

pickled_data = pickle.dumps(video)  # returns data as a bytes object
compressed_pickle = blosc.compress(pickled_data)

with open("video1.dat", "wb") as f:
    f.write(compressed_pickle)
toc1 = timer()
print(f'\n complete in {toc1-tic1: 0.1f}s')

# %%

