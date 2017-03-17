"""
Author: YWiyogo
Description: Helper function for P3
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Analyzing the training datasets


def show_histogram(data, title="Histogram of the datasets"):
    """
    Plotting histogram
    """
    fig_hist = plt.figure(figsize=(15, 8))
    ax = fig_hist.add_subplot(111)
    ax.hist(data, rwidth=0.8, align="mid", zorder=3)
    ax.yaxis.grid(True, linestyle='--', zorder=0)
    ax.set_ylabel('Occurrences')
    ax.set_xlabel('Steering angle')
    ax.set_title(title)
    plt.show()


def get_relative_path(abs_path):
    """
    Get the relative path of the image training data
    """
    filename = abs_path.split("/")[-1]
    return "../p3_training_data/IMG/" + filename


def visualize_img(img1, img2, title1="Image 1", title2="Image 2", cmap=""):
    """Visualize two image side-by-side"""
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    f.tight_layout()
    ax1.set_title(title1)
    ax2.set_title(title2)
    if cmap == "gray":
        ax1.imshow(img1, cmap="gray")
        ax2.imshow(img2, cmap="gray")
        print("visualize as gray")
    else:
        ax1.imshow(img1)
        ax2.imshow(img2)
    plt.show()


def visualize_img(imglist, titlelist, cmaplist):
    """Visualize list of image"""
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


def sobel_color_threshold(img, orient='x', thresh_min=15, thresh_max=100):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
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

    s_thresh = (170, 255)
    binary_s = np.zeros_like(s_channel)
    binary_s[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    combine_binary = np.dstack(
        (np.zeros_like(binary_s), binary_s, binary_sobel))

    visualize_img((img, hls, scaled_sobel, binary_sobel, binary_s, combine_binary),
                  ("RGB", "hls", "scaled_sobel", "binary_sobel",
                   "binary_s", "combine_binary"),
                  (0, 0, 0, 1, 1, 1))
    return combine_binary


def draw_result(undist_img, warped_img, Minv, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(warped_img).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    #pts = np.array(pts, dtype=np.int32)
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective
    # matrix (Minv)
    unwarp = cv2.warpPerspective(
        color_warp, Minv, (undist_img.shape[1], undist_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, unwarp, 0.3, 0)
    plt.imshow(result)
    plt.show()
