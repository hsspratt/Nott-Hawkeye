# %%
from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np


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

def main():
    f_name = 'IMG_2987.jpg'

    img = open_img(f_name)

    x = (240, 4)
    y = (180, 180)

    plt.imshow(img);
    plt.scatter(x, y, alpha=0.8, marker='x')
    plt.show(block=True)

if __name__ == "__main__":
    main()

# %%
z_dist = 30.5
x_dist = 17

import math

theta_x = (math.tan(x_dist/z_dist))
x_pixels = 240-4
angle_pixel_x = theta_x/x_pixels

# x_coord = r*sin(angle_pixel_x*x_pixels)

print(angle_pixel_x)



# %%

x = np.linspace(0,30.5, 100)
y_A_xy = math.tan(theta_x)*x
n = 50

plt.plot(-y_A_xy,x)
plt.scatter(-y_A_xy[n], x[n], color='red', marker='x')
plt.xlim(-20, 20);
# %%
import mpmath as mp

theta_B_xy = 0.1
y_B_xy = mp.cot(0.01)*(x-100)+100

plt.figure()
plt.plot(-y_B_xy,x)
plt.scatter(-y_B_xy[n], x[n], color='red', marker='x')
plt.xlim(-300, 300);
# %%
plt.figure()
plt.plot(-y_A_xy,x)
plt.plot(-y_B_xy,x)
# plt.scatter(-y_B_xy[n], x[n], color='red', marker='x')
# %%
