#%% load modules
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import skimage.color as skm
import pickle

f = open('video.pckl', 'rb')
video = pickle.load(f)
f.close()

