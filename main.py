#%%

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

f_image1 = 'photo 1.jpg'
f_image2 = 'photo 2.jpg'
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

# %%
img1 = image2-image1
plt.imshow(img1)

# %%
keep = np.double(img1>130)
# img1()

plt.imshow(keep); plt.axis('off')


# %%
