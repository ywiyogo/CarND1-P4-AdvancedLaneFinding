"""
Author: YWiyogo
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# importing internal classes and methods
from helper_methods import sobel_color_threshold, draw_result
from calibration import Calibration
from perspective_transform import PerspectiveTransform
from thresholding import Thresholding
from lanefinder import LaneFinder
from curvature import Curvature
from line import Line

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
    curvature = Curvature()
    # print("Cam matrix:\n", calibrator.mtx)
    # open a video file / test image with straight line
    rgb_img = mpimg.imread('test_images/straight_lines1.jpg')
    undist_img = calibrator.undistort_img(rgb_img)
    thres_img = sobel_color_threshold(undist_img)

    transformer.search_start_point(thres_img)
    transformer.do_perpectivetransform()
    warped_img = transformer.warp(thres_img)
    transformer.visualize_transform(thres_img)

    lline = Line()
    rline = Line()
    lanefinder = LaneFinder()
    leftx, rightx = lanefinder.calc_histogram(warped_img)

    curvature.calc_radcurvature(undist_img, leftx, rightx)
    print("Radius curvature: %f, %f" % (curvature.left_curverad,
                                        curvature.right_curverad))

    x_m_per_pix = 3.7 / 700  # meters per pixel in x dimension
    centerroad = (lanefinder.leftx_base + lanefinder.rightx_base) / 2
    centercam = undist_img.shape[1] / 2
    veh_pos = (centerroad - centercam)  * x_m_per_pix
    print("vehicle pos: {}".format(veh_pos))
    if veh_pos > 0:
        pos_text = "Vehicle is {} m left from center".format(abs(veh_pos))
    else:
        pos_text = "Vehicle is {} m right from center".format(abs(veh_pos))

    draw_result(undist_img, warped_img, transformer.Minv, curvature.ploty,
                curvature.left_fitx, curvature.right_fitx, pos_text)
#  Use color transforms, gradients, etc., to create a thresholded binary image.


#  Apply a perspective transform to rectify binary image ("birds-eye view").


#  Detect lane pixels and fit to find the lane boundary.


#  Determine the curvature of the lane and vehicle position with respect to center.


#  Warp the detected lane boundaries back onto the original image.


#  Output visual display of the lane boundaries and numerical estimation of
# lane curvature and vehicle position.

if __name__ == '__main__':
    main()
