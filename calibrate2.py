# %%
import matplotlib.pyplot as plt
import numpy as np
import functions
import importlib as imp

# %% calibrate camera A
imp.reload(functions)

A_img = functions.open_img('IMG_2987_new.jpg')

A_ang_pix = functions.calib(A_img, 30.5, 7, 7, zoom=30)
print('Angle per pixel for camera A')

# %% calibrate camera B

B_img = functions.open_img('IMG_0742_new.jpg')

B_ang_pix = functions.calib(B_img, 30, 16, 6, zoom=30)
print('Angle per pixel for camera B')



# %%  export angles to file
ang_pix = np.stack([A_ang_pix, B_ang_pix],1)

filename = 'angles2'
functions.export_bz2(filename, ang_pix)


# %%
print(np.degrees(B_ang_pix[0]*B_img.shape[1]))


# %%
