# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import argparse as ap
from nms import nms
from config import *

DEBUG_VISUALIZE = True

def sliding_window(image, window_size, step_size):
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def find(im, step_size, threshold,full_image,x_start,y_start,x_end,y_end,show_slide = False):
 
    # Read the image
    min_wdw_sz = (50, 50)
    
    visualize_det = False

    # Load the classifier
    clf = joblib.load(model_path)

    # List to store the detections
    detections = []
    # The current scale of the image
    im_scaled = im
    
    # This list contains detections at the current scale
    cd = []
    # final = (0,0,0,0,0)
    for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
        if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
            continue
        # Calculate the HOG features
        fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        pred = clf.predict(fd)
        if pred == 1 and clf.decision_function(fd) > threshold:
            detections.append((x,y,clf.decision_function(fd),step_size[0],step_size[1]))
            cd.append(detections[-1])
        
        if show_slide:    
            clone1 = full_image.copy()
            for (x, y, _, _, _)  in cd:
                cv2.rectangle(clone1, (x_start + x, y_start + y), (x_start +  x + 50, y_start + y + 50), (0, 0, 0), thickness=2)
            cv2.rectangle(clone1, (x_start,y_start ), (x_end, y_end), (0, 122, 122), thickness=2)
            cv2.rectangle(clone1, (x_start + x, y_start + y), (x_start + x + 50, y_start + y + 50), (255, 255, 255), thickness=2)        
            cv2.imshow("Sliding Window in Progress", clone1)
            cv2.waitKey(1)

    # return (final[0], final[1])        
    # Display the results before performing NMS
    clone = im.copy()
    final = (0,0,0,0,0)
    for (x_tl, y_tl, confidence, w, h) in detections:
        if confidence > final[2]:
            final = (x_tl,y_tl,confidence,w,h)
    # cv2.rectangle(im, (final[0], final[1]), (final[0]+50, final[1]+50), (0, 0, 0), thickness=2)
    if not(final[0] == 0 and final[1] == 0):
        cv2.rectangle(full_image, (x_start + final[0],y_start + final[1]), (x_start + final[0]+50, y_start + final[1]+50), (255, 255, 255), thickness=2)
    cv2.rectangle(full_image, (x_start,y_start ), (x_end, y_end), (0, 122, 122), thickness=2)
    if DEBUG_VISUALIZE:
        cv2.imshow("Searching window", full_image)
        cv2.waitKey(1)
    
    return (final[0],final[1])