#%%

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image1 = plt.imread('photo 1.jpg', 'gray')
image2 = plt.imread('photo 2.jpg')

plt.figure('1')
plt.imshow(image1); plt.axis("off")

plt.figure('2')
plt.imshow(image2); plt.axis("off")

# double1 = np.double(image1)

# %%
img1 = image2-image1
plt.imshow(img1)

# %%
keep = np.double(img1>130)
# img1()

plt.imshow(keep); plt.axis('off')


# %%
