#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import skimage.color as skm

# cap = cv.VideoCapture('/Physics Pics/IMG_0600.mov')
#%%
cap = cv.VideoCapture('cars 10s 1.mp4')
fps = int(cap.get(5))
nframes = int(cap.get(7))

if (cap.isOpened() == False):
    print('Error opening video')

ret, frame = cap.read()
image1 = skm.rgb2gray(frame)
plt.imshow(image1, cmap='gray')

# image1 = cv.imread("A3-P1.jpeg")
# plt.imshow(image1)
#%%

n = 0
video_size = np.hstack((np.shape(frame)[0:2],nframes))
video = np.zeros(video_size)

#%%
while (cap.isOpened()):
    # capture frame by frame
    ret, frame = cap.read()
    if ret == True:
        # display frame
        # plt.figure()
        # plt.imshow(frame)
        # cv.imshow('Frames', frame)
        video[:,:,n] = skm.rgb2gray(frame)

        n+=1
        print('frame: '+str(n))

        
    # Press Q on keyboard to  exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

    else:
        break

cap.release()

cv.waitKey(0)
cv.destroyAllWindows()


# %%




# %%
