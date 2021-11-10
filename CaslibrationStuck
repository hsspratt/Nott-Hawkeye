# %%

import cv2 as cv
import numpy as np
import os
import glob

fname = '/Users/harold/Documents/Academia/Nottingham Uni/Year 4/Image Processing/Coding Project/Nott-Hawkeye/Calibration Pictures/IMG_0673.jpeg'
end_name = os.path.basename(fname)

files = glob.glob('/Users/harold/Documents/Academia/Nottingham Uni/Year 4/Image Processing/Coding Project/Nott-Hawkeye/New Calibration/*jpeg')

# %%

def CalibrateChessBoard_Image(files, PatternSize):
    for fname in files:
        end_name = os.path.basename(fname)
        print('Chessboard is being found in', end_name)
        
        img = cv.imread(fname)

        if img is None:
            print(fname, ' failed to be loaded correctly')
            return None

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((7*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,7), None)

        if ret == 'False':
            print('The Chessboard has failed to be identified in ', end_name)


        # If found, add object points, image points (after refining them)
        if ret == True:
            cv.startWindowThread()

            print('The chessboard was identified in ', end_name)
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,7), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(0)
            print('Figure ', end_name, ' shutdown')
            cv.destroyAllWindows()
            cv.destroyWindow('img')
            #for i in range (1,5):
            cv.waitKey(1)
        else:
            print('The Chessboard has failed to be identified')

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(np.float32(objpoints), np.float32(imgpoints), gray.shape[::-1], None, None)

    return objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs


# fname = '/Users/harold/Documents/Academia/Nottingham Uni/Year 4/Image Processing/Coding Project/Nott-Hawkeye/Calibration Pictures/IMG_0673.jpeg'
# %%

objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs = CalibrateChessBoard_Image(files, (7,7))
# %%

img = cv.imread('/Users/harold/Documents/Academia/Nottingham Uni/Year 4/Image Processing/Coding Project/Nott-Hawkeye/Calibration.jpeg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)


# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult1.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
# %%
