#%% basic loading in of image

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys
import FindBall as FindBall

# path to Physic Pics folder

path_physics_pics = sys.path[0] + '/Physics Pics/'

f_image1 = 'photo 1.jpg'
f_image2 = 'photo 2.jpg'

f_A3_Backround = 'A3-Paper-Backround.jpeg'
f_A3_P1 = 'A3-P1.jpeg'

A3_Backround = cv.imread(f_A3_Backround) # cv.THRESH_BINARY
A3_P1 = cv.imread(f_A3_P1) # cv.THRESH_BINARY

gray_A3_Backround = cv.cvtColor(A3_Backround,cv.COLOR_BGR2GRAY)
gray_A3_P1 = cv.cvtColor(A3_P1,cv.COLOR_BGR2GRAY)

img2 = cv.imread(f_image2,0)

image1 = plt.imread(f_image1, 'gray')
image2 = plt.imread(f_image2)

plt.figure('1')
plt.imshow(image1); plt.axis("off")

plt.figure('2')
plt.imshow(image2); plt.axis("off")

img1 = cv.imread(f_image1,0)
img2 = cv.imread(f_image2,0)

cv.imshow(f_image1,img1)
cv.imshow(f_image2,img2)

plt.imshow(img1)

cv.waitKey(10)
cv.destroyAllWindows()

# double1 = np.double(image1)

# %% image subtraction - plt
img_subtraction_plt = image2-image1
plt.imshow(img_subtraction_plt)

# %% image subtraction - cv2
import skimage.filters as filters

img_subtraction_cv2 = abs(img1-img2)
#cv.imshow('Image Subtraction CV2', img_subtraction_cv2)

img_subtraction_A3 = abs(A3_P1-A3_Backround)
#cv.imshow('Image Subtraction A3 CV2', img_subtraction_A3)

# th1 = cv.adaptiveThreshold(cv.imread(f_A3_P1,0),255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv.THRESH_BINARY,11,2)

# th2 = cv.adaptiveThreshold(cv.imread(f_A3_Backround,0),255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv.THRESH_BINARY,11,2)

# img_subtraction_A3_threshold = abs(th2-th1)
#cv.imshow('Image Subtraction A3 theshold CV2', img_subtraction_A3)

smooth_backround = cv.GaussianBlur(gray_A3_Backround, (95,95), 0)
smooth_p1 = cv.GaussianBlur(gray_A3_P1, (95,95), 0)

# divide gray by morphology image
division_Backround = cv.divide(gray_A3_Backround, smooth_backround, scale=255)
division_p1 = cv.divide(gray_A3_P1, smooth_p1, scale=255)

# sharpen using unsharp masking
sharp_backround = filters.unsharp_mask(division_Backround, radius=1.5, amount=1.5, multichannel=False, preserve_range=False)
sharp_backround = (255*sharp_backround).clip(0,255).astype(np.uint8)

sharp_p1 = filters.unsharp_mask(division_p1, radius=1.5, amount=1.5, multichannel=False, preserve_range=False)
sharp_p1 = (255*sharp_p1).clip(0,255).astype(np.uint8)

thresh = cv.threshold(sharp_backround, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1] #cv.THRESH_BINARY + cv.THRESH_OTSU
thresh1 = cv.threshold(sharp_p1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1] #cv.THRESH_BINARY + cv.THRESH_OTSU

plt.imshow(abs(thresh1),cmap='gray')
plt.imshow(abs(thresh),cmap='gray')


plt.imshow(abs(thresh-thresh1),cmap='gray')
plt.savefig('img_subtraction_A3.png', dpi = 1000)

plt.imshow(A3_P1-A3_Backround,cmap='gray')


# %%
keep = np.double(img1>130)
# img1()

plt.imshow(keep); plt.axis('off')


# %%

