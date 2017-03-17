"""
Author: YWiyogo
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


class PerspectiveTransform:
    """Class of perspective transform"""

    def __init__(self):
        """Constructor"""
        # test_images/straight_lines1.jpg
        self.src = np.float32([[278, 670],
                               [597, 450],
                               [685, 450],
                               [1030, 670]])

        self.dst = np.float32([[280, 710],
                               [280, 100],
                               [1030, 100],
                               [1030, 710]])

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = np.linalg.inv(self.M)

    def warp(self, img):
        """Warping image"""
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]),
                                   flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        """Warping image"""
        return cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0]),
                                   flags=cv2.INTER_LINEAR)

    def visualize_transform(self, src_img):
        """Visualize the transformation line"""
        dst_img = cv2.warpPerspective(src_img, self.M, (src_img.shape[1],
                                                        src_img.shape[0]),
                                      flags=cv2.INTER_LINEAR)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        f.tight_layout()
        ax1.set_title("source")
        ax2.set_title("destination")
        ax1.plot(self.src[:, 0], self.src[:, 1], "r")
        ax2.plot(self.dst[:, 0], self.dst[:, 1], "r")
        ax1.imshow(src_img)
        ax2.imshow(dst_img)
        plt.show()
