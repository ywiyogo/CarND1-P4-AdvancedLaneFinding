"""
Author: YWiyogo
"""
import numpy as np
import matplotlib.pyplot as plt


class Curvature:
    """Curvature Class"""

    def __init__(self):
        self.ploty = 0
        self.left_fitx = 0
        self.right_fitx = 0
        self.left_curverad = 0
        self.right_curverad = 0
        self.veh_pos = 0

    def calc_radcurvature(self, undist_img, leftx, rightx):
        # Generate some fake data to represent lane-line pixels
        # to cover same y-range as image
        self.ploty = np.linspace(0, 719, num=720)
        quadratic_coeff = 3e-4  # arbitrary quadratic coefficient
        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        #leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
        #                               for y in self.ploty])
        #rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
        #                                 for y in self.ploty])

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Fit a second order polynomial to pixel positions in each fake lane
        # line
        left_fit = np.polyfit(self.ploty, leftx, 2)
        right_fit = np.polyfit(self.ploty, rightx, 2)

        self.left_fitx = left_fit[0] * self.ploty**2 + \
            left_fit[1] * self.ploty + left_fit[2]

        self.right_fitx = right_fit[0] * self.ploty**2 + \
            right_fit[1] * self.ploty + right_fit[2]

        # Plot up the fake data
        mark_size = 3
        plt.plot(leftx, self.ploty, 'o', color='red', markersize=mark_size)
        plt.plot(rightx, self.ploty, 'o', color='blue', markersize=mark_size)
        plt.xlim(0, undist_img.shape[1])
        plt.ylim(0, undist_img.shape[0])
        plt.plot(self.left_fitx, self.ploty, color='green', linewidth=3)
        plt.plot(self.right_fitx, self.ploty, color='green', linewidth=3)
        plt.gca().invert_yaxis()  # to visualize as we do the images
        plt.title("Curvature")
        plt.show()

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the
        # image
        y_eval = np.max(self.ploty)

        # R_curve = (1 + (2Ay+B)^2 )^1.5 / |2A|
        self.left_curverad = (
            (1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
        self.right_curverad = (
            (1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])
        print(self.left_curverad, self.right_curverad)
        # Example values: 1926.74 1908.48

        # Define conversions in x and y from pixels space to meters
        y_m_per_pix = 30 / 720  # meters per pixel in y dimension
        x_m_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty * y_m_per_pix, leftx * x_m_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty * y_m_per_pix, rightx * y_m_per_pix, 2)
        print("left_fit_cr: {}, right_fit_cr: {}".format(left_fit_cr,
                                                         right_fit_cr))
        # Calculate the new radii of curvature
        self.left_curverad = ((1 +
                               (2 * left_fit_cr[0] * y_eval * y_m_per_pix +
                                left_fit_cr[1]) **
                               2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        self.right_curverad = ((1 +
                                (2 * right_fit_cr[0] * y_eval * y_m_per_pix +
                                 right_fit_cr[1]) **
                                2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        print(self.left_curverad, 'm', self.right_curverad, 'm')
        # Example values: 632.1 m    626.2 m
        #print("leftlane_x: {}, rightlane_x: {}".format(self.leftlane_x,
        #                                               self.rightlane_x))


        return self.left_curverad, self.right_curverad
