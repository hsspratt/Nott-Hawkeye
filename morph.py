# %% imports
from timeit import default_timer as timer
tic1 = timer()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pickle
import bz2

f_name = "diff_thresh.pbz2"

with bz2.BZ2File(f_name, 'rb') as f:
    diff_thresh = pickle.load(f)


# %%
