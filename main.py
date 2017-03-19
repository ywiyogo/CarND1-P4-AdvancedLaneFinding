"""
Author: YWiyogo
Desc: P4, Adcanced Lane Lines detection
"""
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
# importing internal classes and methods
from helper_methods import sobel_color_threshold
from calibration import Calibration
from perspective_transform import PerspectiveTransform
from lanefinder import LaneFinder

# Compute the camera calibration matrix and distortion coefficients given
# a set of chessboard images. We need to perform this once -> using OOP

DEBUG = 0
calibrator = None
transformer = None
lanefinder = None


def process_image(rgb_img):
    """
    NOTE: The output you return should be a color image (3 channel) for
    processing video below
    """
    global calibrator
    global transformer
    global lanefinder
    # Compute the camera calibration matrix and distortion coefficients given
    if calibrator is None:
        calibrator = Calibration()
    undist_img = calibrator.undistort_img(rgb_img)

    #  Use color transforms, etc.,to create a thresholded binary image.
    thres_img = sobel_color_threshold(undist_img)
    if transformer is None:
        transformer = PerspectiveTransform()

    #  Apply a perspective transform to rectify binary image ("birds-eye view").
    transformer.do_perpectivetransform()
    warped_img = transformer.warp(thres_img)
    if DEBUG:
        transformer.visualize_transform(thres_img)

    if lanefinder is None:
        lanefinder = LaneFinder()
    # Perform the lane detection
    leftx, rightx = lanefinder.detect_lanes(warped_img)

    lanefinder.calc_radcurvature()
    lanefinder.calc_vehicle_position(undist_img)
    # Output visual display of the lane boundaries and numerical estimation of
    return lanefinder.draw_result(undist_img, warped_img, transformer.Minv)


def main():
    """Main function"""
    # open a video file / test image with straight line
    # rgb_img = mpimg.imread("./test_images/test4.jpg")
    # process_image(rgb_img)
    write_output = "project_video_result.mp4"
    vidfilename = "project_video.mp4"
    clip = VideoFileClip(vidfilename)
    # NOTE: this below function expects color images!!
    proc_clip = clip.fl_image(process_image)
    proc_clip.write_videofile(write_output, audio=False)


if __name__ == '__main__':
    main()
