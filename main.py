#%% basic loading in of image

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

f_image1 = 'photo 1.jpg'
f_image2 = 'photo 2.jpg'

f_A3_Backround = 'A3-Paper-Backround.jpeg'
f_A3_P1 = 'A3-P1.jpeg'

A3_Backround = cv.imread(f_A3_Backround,0)
A3_P1 = cv.imread(f_A3_P1,0)

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

cv.waitKey(10)
cv.destroyAllWindows()

# double1 = np.double(image1)

# %% image subtraction - plt
img_subtraction_plt = image2-image1
plt.imshow(img_subtraction_plt)

# %% image subtraction - cv2

img_subtraction_cv2 = abs(img1-img2)
cv.imshow('Image Subtraction CV2', img_subtraction_cv2)

img_subtraction_A3 = abs(A3_P1-A3_Backround)
cv.imshow('Image Subtraction A3 CV2', img_subtraction_A3)

th1 = cv.adaptiveThreshold(A3_P1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

th2 = cv.adaptiveThreshold(A3_Backround,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

img_subtraction_A3_threshold = abs(th1-th2)
cv.imshow('Image Subtraction A3 theshold CV2', img_subtraction_A3)



cv.waitKey(100)
cv.destroyAllWindows()

# %%
keep = np.double(img1>130)
# img1()

plt.imshow(keep); plt.axis('off')


# %%

