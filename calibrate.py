# %%
from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
import math

def open_img(f_name):
    """Opens an image file from physics pics folder

    Parameters
    ----------
    f_name : string
        Name of the file

    Returns
    -------
    np.array
        numpy array of the picture
    """    
    path = sys.path[0] + '/Physics Pics/' + f_name
    img = np.array(Image.open(path))
    return img


f_name = 'IMG_2987.jpg'

img = open_img(f_name)

x_points = (240, 4)
y_points = (180, 180)

plt.imshow(img);
plt.scatter(x_points, y_points, alpha=0.8, marker='x')
plt.show(block=True)

# if __name__ == "__main__":
#     main()

# %%
z_dist = 30.5
x_dist = 17

theta_x = (math.tan(x_dist/z_dist))
x_pixels = x_points[0]-x_points[1]
angle_pixel_x = theta_x/x_pixels

# x_coord = r*sin(angle_pixel_x*x_pixels)

print(angle_pixel_x)


# %%

