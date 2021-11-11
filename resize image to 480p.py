# %%
from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np

f_name = 'IMG_2987.jpg'
path = sys.path[0] + '/Physics Pics/' + f_name
img = np.array(Image.open(path))
plt.imshow(img);
plt.scatter(480/2, 360/2, alpha=0.8, marker='x')

# %%

basewidth = 480

wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), Image.ANTIALIAS)
img.save('resized_image.jpg')


# %%
