"""
Author: YWiyogo
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

# importing internal classes and methods
from helper_methods import visualize_img, color_grad_thresh
from calibration import Calibration
from perspective_transform import PerspectiveTransform
from thresholding import Thresholding
from lanefinder import LaneFinder


# cv2.imshow("Test image", img)
# cv2.waitKey()

# Compute the camera calibration matrix and distortion coefficients given
# a set of chessboard images. We need to perform this once -> using OOP
# The dimension of the chessboard is 6x9
# Even though the chessboard can be 2D, the object points has to be
# 3 dimensional due to the camera matrix for the funt cv2.undistort
# cv2.error: (-210) objectPoints should contain vector of vectors of points of
# type Point3f in function collectCalibrationData


# calibrate_camera()

def main():
    """Main function"""
    # Compute the camera calibration matrix and distortion coefficients given
    #  Apply a distortion correction to raw images.
    calibrator = Calibration()
    transformer = PerspectiveTransform()
    # print("Cam matrix:\n", calibrator.mtx)
    # open a video file / test image with straight line
    rgb_img = mpimg.imread('test_images/straight_lines1.jpg')

    thres_img = color_grad_thresh(rgb_img)

    warped_img = transformer.warp(thres_img)
    transformer.visualize_transform(thres_img)

    lanefinder = LaneFinder()
    lanefinder.calc_histogram(warped_img)

#  Use color transforms, gradients, etc., to create a thresholded binary image.


#  Apply a perspective transform to rectify binary image ("birds-eye view").


#  Detect lane pixels and fit to find the lane boundary.


#  Determine the curvature of the lane and vehicle position with respect to center.


#  Warp the detected lane boundaries back onto the original image.


#  Output visual display of the lane boundaries and numerical estimation of
# lane curvature and vehicle position.

if __name__ == '__main__':
    main()