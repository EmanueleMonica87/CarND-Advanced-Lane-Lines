## Advanced Lane Finding

### Emanuele Monica

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

[image1]: ./examples/Ema_chessboard.jpg "Calibration"
[image2]: ./examples/Ema_unwarp.jpg "Road Undistorted"
[image3]: ./examples/Ema_unwarp2.jpg "Road Unwarped"
[image4]: ./examples/Ema_binary.jpg "Thresholding"
[image5]: ./examples/Ema_color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/Ema_example_output.jpg "Output"
[image7]: ./examples/Ema_example_output2.jpg "Output Wrong"
[video1]: ./result.mp4 "Video"
[video2]: ./challenge_result.mp4 "Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

This is the Writeup file. The code I wrote is a mixture of sources. I would like to thank for the contribution JustinHeaton (https://github.com/JustinHeaton/Advanced-Lane-Finding) from whom I took many parts of code (especially from gradients onward), rearranging and modifying some of them to better understand the logic behind and possible improvements. I also took help from the Udacity course code.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines #7 through #27 of the third code cell of the IPython notebook file called `P2_final.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I used the distortion matrix found previously with the chessboard to undistort and then unwarp the images of the road, like the one below (please notice the car hood differences for the first pair):

![alt text][image2]
![alt text][image3]


I created a function called `undistort()` and another one called `warped()` to be able to repeat this operation quickly. Undistort takes in an image and 2 parameters, show and write. By properly setting these parameters to True or False, I can command if the output of the single block will be plotted or its output saved. This helped me a lot during debugging so I used the same approach many times in the other functions.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used HLS and CIE Lab color conversion from cv2 package to extract the l and b channels respectively. I found this choice to be fortunate. In the presence of mixed yellow and white lanes, the combined behavior of these threshold performed very good. Before this, I used Sobel approach with a previous file, then scrapped it because I believed this was more performant.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The undi`undistort()` and `warped()` functions can be found at 5th code cell of the IPython notebook `P2_final.ipynb`.
The `warped()` function takes as inputs an image (`img`), and the source (`src`) and destination point (`dst`) are hardcode in the following manner:

```python
# Define the 4 corners of the trapezoid, source
    top_left =    [580,460]               #these values get a better performance for the warping but
    bottom_left =  [203,720]             #in the pipeline they totally distort one of the lane output
    bottom_right = [1127,720]
    top_right=      [705,460]
    
    #top_left =    [490,482]
    #top_right =     [810,482]
    #bottom_right = [1250,720]
    #bottom_left=   [40,720]
    
    #trapezoid_src = [top_right,top_left,bottom_right,bottom_left]
    x_values_src = [top_left[0],bottom_left[0],bottom_right[0],top_right[0],top_left[0]]
    y_values_src = [top_left[1],bottom_left[1],bottom_right[1],top_right[1],top_left[1]]
    
    # Define the 4 corners of the trapezoid, warped
    top_left_dst =        [320,0]         #these values get a better performance for the warping but
    bottom_left_dst =     [320,720]       #in the pipeline they totally distort one of the lane output  
    bottom_right_dst =    [960,720]
    top_right_dst=        [960,0]
    
    #top_left_dst =    [0,0]
    #top_right_dst =     [1280,0]
    #bottom_right_dst = [1250,720]
    #bottom_left_dst=   [40,720]
    
    x_values_dst = [top_left_dst[0],bottom_left_dst[0],bottom_right_dst[0],top_right_dst[0],top_left_dst[0]]
    y_values_dst = [top_left_dst[1],bottom_left_dst[1],bottom_right_dst[1],top_right_dst[1],top_left_dst[1]]

    # Define the trapezoids, source and destination
    src = np.float32([[top_right],[top_left],[bottom_right],[bottom_left]])
    dst = np.float32([[top_right_dst],[top_left_dst],[bottom_right_dst],[bottom_left_dst]])

```


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear approximately parallel in the warped image.

Note: due to time constraint, I was not able to optimize the warping function, however the code reports commented another set of points, which would not perform the perfect unwarping, however in the overall pipeline, they perfomed better. Seems that the point set somehow influence the polynomial function. Please see below where this warping fails.


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used hystogram and sliding windows to identify the points to connect and use a second order polynomial, like explained in the course lesson. I used a function called `get_intercepts()` to extrapolate the lines until the end of the warped image:

```python
    def get_intercepts(self, polynomial):
        bottom = polynomial[0]*720**2 + polynomial[1]*720 + polynomial[2]
        top = polynomial[0]*0**2 + polynomial[1]*0 + polynomial[2]
        return bottom, top
```
```python
    # Calculate intercepts to extend the polynomial to the top and bottom of warped image
    leftx_int, left_top = Left.get_intercepts(left_fit)

```
And I used the following code to average the lines:

```python
    # Average intercepts across n frames
    Left.x_int.append(leftx_int)
    Left.top.append(left_top)
    leftx_int = np.mean(Left.x_int)
    left_top = np.mean(Left.top)
    Left.lastx_int = leftx_int
    Left.last_top = left_top

```
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature in lines `fill_lanes()` function of `P2_final.ipynb`. This is already in meters with the conversion values below. Also, in the same block the position of the vehicle is calculated from the values of the polynomial.

```python

left_fitx = left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]
rightx_int = right_fit[0]*720**2 + right_fit[1]*720 + right_fit[2]

# Measure Radius of Curvature for each lane line
    ym_per_pix = 30./720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*np.max(lefty) + left_fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*np.max(lefty) + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])
    
    
    # Calculate the position of the vehicle
    center = abs(640 - ((rightx_int+leftx_int)/2))


```
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image:

![alt text][image6]

Here is the example in where the optimal warping points fail. If you want to try, uncomment the other points and comment the used ones to get a different warping which seems less performant in the straight lines but actually perform better on this image.

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

I also tried the pipeline with the challenge video with the following results:

Here's a [link to my video result](./challenge_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have noticed wobbled lines in the video, and definitely the warping function needs to be further investigating (the optimal warping points get the final image to fail). If I have time, I would like to come back to this project later to implement some of the features I learn new everyday, like the combination of `start_time = time.clock()` and `print(time.clock()-start_time, "seconds"`, which I used to test and improved the speed of the processing. I would also try implementing some of the Sobel functions to get a more robust detection of the lines and a reality checks on different right/left curvatures and rapidly changing curvature values. I would also put a threshold of curvature above which the output would print "straight".