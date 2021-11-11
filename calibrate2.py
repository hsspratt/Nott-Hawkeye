# %%
import matplotlib.pyplot as plt
import numpy as np
import functions

# %% calibrate camera A

A_img = functions.open_img('IMG_2987.jpg')

A_ang_pix = functions.calib(A_img, 30.5, 6, 5)
print('Angle per pixel for camera A')

# %% calibrate camera B

B_img = functions.open_img('IMG_0742.jpg')

B_ang_pix = functions.calib(B_img, 30, 12, 5)
print('Angle per pixel for camera B')



# %%  export angles to file
ang_pix = np.stack([A_ang_pix, B_ang_pix],1)

filename = 'angles'
functions.export_bz2(filename, ang_pix)

