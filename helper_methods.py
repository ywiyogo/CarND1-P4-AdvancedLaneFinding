"""
Author: YWiyogo
Description: Helper function for P3
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Analyzing the training datasets

DEBUG = 0


def visualize_img(imglist, titlelist, cmaplist):
    """Visualize list of image"""
    if DEBUG:
        rows = int(len(imglist) / 2) + (len(imglist) % 2 > 0)
        f, axarr = plt.subplots(rows, 2, figsize=(10, 8))
        f.tight_layout()
        i = 0
        j = 0
        for idx, img in enumerate(imglist):
            if rows < 2:
                axis = axarr[i]
                i = i + 1
            else:
                axis = axarr[i, j]
                if j < axarr.shape[1] - 1:
                    j = j + 1
                else:
                    i = i + 1
                    j = 0
            axis.set_title(titlelist[idx])
            if cmaplist[idx] == 1:
                axis.imshow(img, cmap="gray")
            else:
                axis.imshow(img)
        plt.show()


def s_channel_thresholding(img, thresh_min=170, thresh_max=255):
    """Convert image to HLS and return the S channel and its thresholding"""
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    binary_s = np.zeros_like(s_channel)
    binary_s[(s_channel > thresh_min) & (s_channel <= thresh_max)] = 1
    return s_channel, binary_s


def sobel_color_threshold(img, orient='x', thresh_min=25, thresh_max=100):
    """Perform binary thresholding based on Sobel operator and S channel"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value

    kernelsize = 3
    if orient == 'x':
        sobel_gray = np.absolute(
            cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernelsize))
    if orient == 'y':
        sobel_gray = np.absolute(
            cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernelsize))
    abs_sobel_gray = np.absolute(sobel_gray)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel_gray / np.max(abs_sobel_gray))
    # Note: It's not entirely necessary to convert to 8-bit (range from 0 to
    # 255) but in practice, it can be useful in the event that you've written
    # a function to apply a particular threshold, and you want it to work the
    # same on input images of different scales, like jpg vs. png.

    # Create a copy and apply the binary threshold
    binary_sobel = np.zeros_like(scaled_sobel)
    binary_sobel[(scaled_sobel >= thresh_min) &
                 (scaled_sobel <= thresh_max)] = 1
    # call S channel
    s_channel, binary_s = s_channel_thresholding(img, 190, 255)

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image.
    # It might be beneficial to replace this channel with something else.
    combine_binary = np.dstack(
        (np.zeros_like(binary_s), binary_s, binary_sobel))
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    visualize_img((img, hls, s_channel, binary_sobel, binary_s, combine_binary),
                  ("RGB", "hls", "s_channel", "binary_sobel",
                   "binary_s", "combine_binary"),
                  (0, 0, 0, 1, 1, 1))
    return combine_binary
