# %% Test Calibration

import numpy as np
import cv2
import glob
import os
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('/Users/harold/Documents/Academia/Nottingham Uni/Year 4/Image Processing/Coding Project/Nott-Hawkeye/Calibration GG/*.jpeg')

for i in range(len(images)):
    print(os.path.basename(images[i]))

for fdir in images:
    fname = os.path.basename(fdir)
    print('Finding Chessboard Corners for ', fname)
    img = cv2.imread(fdir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,7), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.destroyWindow('img')
        cv2.waitKey(5)
cv2.destroyAllWindows()

# %%

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# %%

img = cv2.imread('/Users/harold/Documents/Academia/Nottingham Uni/Year 4/Image Processing/Coding Project/Nott-Hawkeye/Calibration GG/Model/IMG_0719.jpeg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# %%

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult_test.png', dst)

# %%

# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# crop the image
x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
dst = dst[x:x+w, y:y+h]
cv2.imwrite('calibresult_test_1.png', dst)




# %%
