
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

cap = cv2.VideoCapture('/Physics Pics/IMG_0600.mov')
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    cv2.imshow('window-name', frame)
    cv2.imwrite("frame%d.jpg" % count, frame)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() # destroy all opened windows