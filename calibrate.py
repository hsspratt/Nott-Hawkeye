# %%
from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
import math
import skimage.color as skm
import functions

# %%

f_name = 'IMG_2987.jpg'

img = functions.open_img(f_name)
img_size = np.shape(img)

x_points = (240, 6, 240)
y_points = (180, 180, 5)

opacity = 0.8

fig = plt.figure(figsize=(10,10))
ax = plt.imshow(img);
plt.scatter(x_points, y_points, s=200, alpha=opacity, marker='x')
plt.title('Full calibration image')
plt.show(block=True)


subplot, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,10))
ax1.imshow(img);
ax1.scatter(x_points, y_points, s=500, alpha=opacity, marker='x')
ax1.axis([img_size[1]/2-40, img_size[1]/2+40, img_size[0]/2+40,img_size[0]/2-40])
ax1.set_title('Centre point')
ax2.imshow(img);
ax2.scatter(x_points, y_points, s=500, alpha=opacity, marker='x')
ax2.axis([0,80, 220,140])
ax2.set_title('Centre left')
ax3.imshow(img);
ax3.scatter(x_points, y_points, s=500, alpha=opacity, marker='x')
ax3.axis([200,280, 80,0])
ax3.set_title('Top centre')
subplot.tight_layout()
plt.show(block=True)


# if __name__ == "__main__":
#     main()

# %%
z_dist = 30.5
x_dist = 17
y_dist = 13

theta_x = (math.tan(x_dist/z_dist))
x_pixels = x_points[0]-x_points[1]
angle_pixel_x = theta_x/x_pixels

theta_y = (math.tan(y_dist/z_dist))
y_pixels = np.abs(y_points[0]-y_points[2])
angle_pixel_y = theta_y/y_pixels

print(f'x-axis: {angle_pixel_x: 0.4f}rad/px' + f'   y-axis: {angle_pixel_y: 0.4f}rad/px')

print(f'For the y-axis: distance is {y_dist}cm; pixel distance is {y_pixels}px' + 
    f' and angle per pixel is {angle_pixel_y:0.4f}rad/px.')



# %%
gray = skm.rgb2gray(img)

# %%

histogram, bin_edges = np.histogram(gray, bins=100, range=(0, 1))

plt.plot(bin_edges[0:-1], histogram)

    
# ang_pix_x, ang_pix_y = functions.calib_calc(17, 15, 30.5)



# %%

# %%
