#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# cap = cv.VideoCapture('/Physics Pics/IMG_0600.mov')

cap = cv.VideoCapture('cars 10s 1.mp4')

if (cap.isOpened() == False):
    print('Error opening video')

# ret, frame = cap.read()
# plt.imshow(frame)

# image1 = cv.imread("A3-P1.jpeg")
# plt.imshow(image1)


while (cap.isOpened()):
    # capture frame by frame
    ret, frame = cap.read()
    if ret == True:
        # display frame
        # plt.figure()
        # plt.imshow(frame)
        cv.imshow('Frames', frame)
        
    # Press Q on keyboard to  exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

    else:
        break

cap.release()



# %%
