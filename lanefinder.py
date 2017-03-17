"""
Author: Y.Wiyogo
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper_methods import visualize_img
from matplotlib.patches import Rectangle

DEBUG = 0
class LaneFinder:
    """Lane Finder Class"""

    def __init__(self):
        """Constructor"""
        self.left_fitx = 0
        self.right_fitx = 0
        self.leftx_base = 0
        self.rightx_base = 0

    def calc_histogram(self, binwarped_img):
        """Calculate histogram of a binary warped image"""
        half_y = int(binwarped_img.shape[0] / 2)
        print("binwarped_img shape: ", binwarped_img.shape)
        # binary warped image has 3 channels
        histogram = np.sum(binwarped_img[half_y:, :, 1:], axis=0)
        plt.plot(histogram)
        # add the distribution of both channels
        histogram = histogram[:, 0] + histogram[:, 1]
        plt.plot(histogram)
        plt.show()
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binwarped_img, binwarped_img, binwarped_img)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint], axis=0)
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        print("Histogram shape: ", histogram.shape)
        print("midpoint: ", midpoint)
        print("Left base: ", leftx_base)
        print("Right base: ", rightx_base)
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binwarped_img.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binwarped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binwarped_img.shape[0] - (window + 1) * window_height
            win_y_high = binwarped_img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            # cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2)
            # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            if DEBUG:
                print("Out img: ", out_img.shape)
                print("Window xleftlow %d %d %d %d" % (win_xleft_low, win_y_low,
                      win_xleft_high, win_y_high))
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.add_patch(Rectangle((win_xleft_low, win_y_high), 2 * margin, window_height, fill=False, alpha=1, color="red"))
                ax.add_patch(Rectangle((win_xright_low, win_y_high), 2 * margin, window_height, fill=False, alpha=1, color="red"))
                ax.imshow(binwarped_img)
                plt.show()
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) &
                              (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) &
                               (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binwarped_img.shape[0] - 1, binwarped_img.shape[0] )
        self.left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        self.right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(binwarped_img)
        plt.plot(self.left_fitx, ploty, color='yellow')
        plt.plot(self.right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

        return self.left_fitx, self.right_fitx
