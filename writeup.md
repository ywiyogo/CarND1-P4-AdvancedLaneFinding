# Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_images/straight_lines1.jpg "Test image"
[image2]: ./examples/undistorted_comparison.png "Undistorted"
[image3]: ./examples/img_compare_straighttest.png "Threshold Images Comparisons"
[image4]: ./examples/perspectivetrans_straight.png "Warp Example"
[image5]: ./examples/perspectivetrans_threshol_straight.png "Warp Example"
[image6]: ./examples/histogram_straight.png "Histogram"
[image7]: ./examples/laneline_fitting_straight.png "Line fitting and sliding windows"
[image8]: ./examples/laneline_fitting.png "Line fitting curve"
[image9]: ./examples/endresult_straight.png "End result"
[image10]: ./test_images/test4.jpg "Test4"
[image11]: ./examples/wrong_detect_test4_0.png "Wrong sliding windows"
[image12]: ./examples/wrong_detect_test4_1.png "Wrong fitting line"
[image13]: ./examples/correct_detect_test4.png "Correction"
[image14]: ./examples/endresult_test4.png "End result test4"
[video1]: ./project_video_result_outliers.mp4 "Video with the outliers"
[video2]: ./project_video_result.mp4 "End result video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
## Camera Calibration

### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Since the camera calibration have to be done only once, I implemented the code as a class in *calibration.py*. The camera matrix, the distortion coeficients, the rotation and the translation matrix are saved as *calibration_data.p*. The *calibration.py* contains the class constructor and the `undistort_img()` function.

For this project, Udacity provides 9x6 chessboard images. So, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `opoints` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. The function`cv2.findChessboardCorners()` returns `true` if it finds corners. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The function detects 20 from 23 chessboard images.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

## Pipeline (single images)
For the first basic implementation, I use the test image in case of the straight lanes (*straight_lines1.jpg*), as shown below:

![alt text][image1]

### 1. Provide an example of a distortion-corrected image.
For a single image, firstly I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]
### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image. For the color threshold, the HLS image conversion is applied to get the S channel. S channel shows more robustness to detect the lane lines. The thresholding steps are implemented in the function `sobel_color_threshold()` in a helper file (`helper_methods.py`). I use the Sobel operation for the x derivative. Below are the steps and the image comparison of each steps:

![alt text][image3]

The *combine_binary* image is an image combination of the image *binary_sobel* and *binary_s*

### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform can be found in `perspective_transform.py`. The file includes a class implementation and several functions. Like the calibration, the perspective transformation needs to be initialized once. 

The main functions that can be called from the main function are  `do_perpectivetransform()` and `ẁarp()`.
The `do_perpectivetransform()` performs the perspective transformation from the manually defined points on the source image and on the warp image.
The `warp(img)` function takes as inputs an image (`img`). It used the source and destination points which are declared as the member variables.  I chose the hardcode the source and destination points in the following way:

```
    def __init__(self):
        """Constructor"""
        # Calibrate the trapezoid first with test_images/straight_lines1.jpg
        # The trapezoid shall not to long, since it will affects the detection
        # for the curve
        self.src = np.float32([[278, 670],    # 1
                               [588, 455],    # 2
                               [693, 455],    # 3
                               [1030, 670]])  # 4

        self.dst = np.float32([[280, 710],
                               [280, 20],
                               [1030, 20],
                               [1030, 710]])
        self.M = 0
        self.Minv = 0
```
The trapezoid lines shall be done in case of the straight road. As a helper function, I implemented the function `search_start_point()` in order to get the lower point of the trapezoid. The order of the trapezoid points in the source code is represented in this figure:

![alt text][image4]

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. The drawing is implemented in the function `visualize_transform(src_img)`. An example of the transformation with the binary threshold image can be seen below:

![alt text][image5]

### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For this step, I write the code in `linefinder.py` and implemented as a class LaneFinder. The class contains:
1. Constructor and its member variables for the line properties
2. `detect_lanes(binwarped_img)` to calculate the histogram and continued by performing sliding windows to detect the left and right lanes
3. `calc_radcurvature()` to calculate the radius curvature
4. `calc_vehicle_position(undist_img)` in order to calculate the vehicle position in respect to the center lane
5. `draw_result(undist_img, warped_img, m_inv)` to draw the result on an image or a frame of a video.

The function `detect_lanes(binwarped_img)` contains the steps to identified lane-line pixeled:
1. Calculate the histogram and the x-axis with the most bins for left and right lines

![alt text][image6]
2. From the x-axis base, I applied the sliding windows method, which splits the vertical region into several windows. If I find that the found pixel in the binary threshold image higher than the minimum pixels, I recenter the next window on their mean position.
3. Based on the sliding windows we can call the `numpy.polyfit()` to get the coeficients of the second order polynomial. Using this math formula `Ax^2 + By + C`, I can produce the x points in respect to the y-axis.

| Straight lanes| Curve lanes | 
| :----: | :----: | 
|![alt text][image7] |![alt text][image8]|


### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

As described above, the calculation of the curvature radius is implemented in `calc_radcurvature()`. The implementation is based on the code implementation in section 35 of the Advance Line Finding lecture. I define two variables to map the value from pixel image to the real world unit:

    self.y_m_per_pix = 30 / 720 
    self.x_m_per_pix = 3.7 / 700

A new coeficient values for the real world can be calculated by this function call:

    np.polyfit(self.plot_y * self.y_m_per_pix, self.llane_x * self.x_m_per_pix, 2)

The formula for the radius can be seen in the section 35 of the lecture.

For the vehicle position, I need the x values from the both lanes at the last y-axis (720-1 = 719). Then, from these values I can calculate the center of the lane. The center of the camera is the half of the image width. The function `calc_vehicle_position(undist_img)` implements this calculation:

    centerroad = (self.llane_x[y_pos] + self.rlane_x[y_pos]) / 2
    centercam = undist_img.shape[1] / 2
    self.veh_pos = round((centerroad - centercam) * self.x_m_per_pix, 2)


### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

`draw_result(undist_img, warped_img, m_inv)` in `linefinder.py` implements the drawing function for the end result.  Here is an example of my result on a test image:

![alt text][image8]

---

## Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

In order to get the video frames, I use the module `VideoFileClip` from `moviepy.editor` which has been used in the first project. These codes are implemented in the main function:

    write_output = "project_video_result.mp4"
    clip = VideoFileClip("project_video.mp4")
    # NOTE: this below function expects color images!!
    proc_clip = clip.fl_image(process_image)
    proc_clip.write_videofile(write_output, audio=False)


Here's a [link to my video result](./project_video_result.mp4)

---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

After I ran my first pipeline, I observe several outliers in my [first video](./project_video_result_outliers.mp4). In the video *project_videp.mp4*, the outliers happen because of two reasons:

1. the shadows of the trees.
2. wide distance of the dashed lines
3. Bright color of the road

I can verified the outliers using the test image *test4.jpg*.

![alt text][image10]

The result of the sliding windows and the line fitting are like these figures:

1. Wrong sliding windows 

![alt text][image11] 

2. The resulting fitting line
 
![alt text][image12]


The first effort to solve this issue is to tuning the threshold values of the binary images. Then, based on the visualization of the warped image and the windows, we can enlarge the width size of the sliding windows (e.g. 130). The below figure shows the correction:

![alt text][image13]

Seconds, due to the noise on a binary threshold image, the starting sliding window can be wrong. For instance, the right line starts from the middle of the image (x-axis: 640) since the histogram in the middle area is higher due to the missing dashed line (see my [first video](./project_video_result_outliers.mp4)).  To deal with this issue, I compared the difference between the detected base x-axis (from histogram) with the previous detection. If the difference is more than 80 to 100 pixels, I can use the previous value. This approach is very significant since it tell us the start points of the lines. 

![alt text][image14]

Third, in order to cope with the outliers in the video frames, I modified `detect_lanes(binwarped_img)`, so that I can detect an outlier or bad line fitting. During the sliding windows method, I count the bad windows that contain less pixel than the minimum thershold pixel. After the loop, I check if the nu,ber of the detected windows is higher than the minimum threshold (e.g. 60% detection rate of the total windows for a line, 6 of 9). If there are not enough windows with minimal pixel for one line, I use the previous calculation, which is saved as the member variable.

    if detected_win_count[0] >= min_detected_win:
        self.left_detected = True
        self.llane_x = (self.llane_coeffis[0] * self.plot_y**2 +
                        self.llane_coeffis[1] * self.plot_y +
                        self.llane_coeffis[2])
    else:
        self.left_detected = False
        print("Warning left detected windows is too low: {} < {}".format(
            detected_win_count[0], min_detected_win))



