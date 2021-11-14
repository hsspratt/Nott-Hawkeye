#%%
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# import pickle
# import bz2
# import lzma
# import sys 
from PIL import Image, ImageFilter
# import skimage.morphology as morph
import functions as f
import importlib as imp

# %% takes approximatly half the length of the video to decompress

# A_video = f.import_lzma('cameraA_test1')
# B_video = f.import_lzma('cameraB_test1')

A_video_in, B_video_in = f.import_bz2('test_photos1')

frame = 4

imp.reload(f)

A_video = f.gauss_blur(A_video_in, 2)
B_video = f.gauss_blur(B_video_in, 2)

f.compare_frames(A_video, B_video, frame, s_title='Video frames')
# %%
# A_background = f.ref_image(A_video)
# B_background = f.ref_image(B_video)
A_background = A_video[:,:,0:1]
B_background = B_video[:,:,0:1]


f.compare_frames(A_background, B_background, 0, s_title='Reference image')

imp.reload(f)

A_diff = f.difference(A_video, A_background)
B_diff = f.difference(B_video, B_background)

f.compare_frames(A_diff, B_diff, frame, s_title='Difference')

nframes = (np.shape(A_video)[2], np.shape(A_video)[2])
print(nframes)

imp.reload(f)
thresh_val = 0.3
# A_thresh = f.threshold(A_diff[:,:,frame:frame+1], thresh_val)
A_thresh = f.threshold(A_diff, 0.15)
B_thresh = f.threshold(B_diff, 0.2)

# f.compare_frames(A_thresh, A_diff[:,:,frame:frame+1], 0)
f.compare_frames(A_thresh, B_thresh, frame, axis='on', s_title = 'Thresholded')

def plot(frame, title=np.nan):
    plt.imshow(frame, cmap='gray')

radius = 14
A_closed = f.closing_disk(A_thresh, radius)
B_closed = f.closing_disk(B_thresh, radius)

# %%

# # plot(A_closed)
# for i in range(0,9):
#     f.compare_frames(A_closed, B_closed, i)

# %%
A_xy = f.centre_points(A_closed)
B_xy = f.centre_points(B_closed)

i = -7
plt.figure()
plt.imshow(A_closed[:,:,i], cmap='gray'); plt.axis('off')
plt.scatter(A_xy[0,i],A_xy[1,i], color='r', s=20, marker='x')
# plt.axis([A_xy[0]-40, A_xy[0]+40, A_xy[1]-40, A_xy[1]+40])
plt.xlim([A_xy[0,i]-40, A_xy[0,i]+40])
plt.ylim([A_xy[1,i]-40, A_xy[1,i]+40])


# %%
imp.reload(f)
A_img = f.open_img('IMG_0795.jpg')

f.calib_honing(A_img, 100, 100)
f.calib_count(A_img, (100,100))

# %%

imp.reload(f)

image_save = f.visualise(B_video_in, B_xy)

f.vis_player(image_save, 1000)

# %% Angles

calib = f.import_bz2('angles')

Aang_x = (A_xy[0,:]-240)*calib[0,0]
Aang_y = (A_xy[1,:]-180)*calib[0,1]

Bang_x = (B_xy[0,:]-240)*calib[0,0]
Bang_y = (B_xy[1,:]-180)*calib[0,0]

angles = np.vstack((Aang_x, Aang_y, Bang_x, Bang_y))

plt.plot(np.repeat(np.expand_dims(range(0,9), 1), 4, 1), angles.T)


# %%

f.export_bz2('test_photos_angles', angles)

# %%
