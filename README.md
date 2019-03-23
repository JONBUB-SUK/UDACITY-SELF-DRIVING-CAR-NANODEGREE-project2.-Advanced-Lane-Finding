# UDACITY-SELF-DRIVING-CAR-NANODEGREE-project2.-Advanced-Lane-Finding


[//]: # (Image References)

[image1-1]: ./images/1.1_Percaptron,Lambda.JPG "RESULT1"
[image1-2]: ./images/2.LeNet.JPG "RESULT2"
[image1-3]: ./images/3.Center,Left,Right_images.JPG "RESULT3"
[image1-4]: ./images/4.Cropping.JPG "RESULT4"
[image1-5]: ./images/5.NVIDIA_architecture.JPG "RESULT5"

[image2-1]: ./images/NVIDIA_CNN_architecture.png "NVIDIA"

[image3-1]: ./images/동영상_스크릿샷.png "RESULT VIDEO"

# Introduction

The object of the project is finding lanes at driving car videos

There are some conditions to pass the project

1. Have to calibrate camera

2. Calculate curvature of lane at each frame

3. Calculate distance from center at each frame

4. Make lane class to use before detection information

5. Draw lines on found lanes and fill color inside of lines

6. Found lanes are similar to original lanes


# Background Learning

### Camera calibration

- Camera calibration using cv2 library

- Perspective transform

### Gradients and color spaces

- Gradient threshold

- Various color spaces

### Advanced Computer Vision

- Finding lanes using histogram

- Finding lanes using sliding window

- Finding lanes using prior detection

- Measuring curvature



# Approach

### 1. Image calibration

Use `cv2.findChessboardCorners` & `cv2.drawChessboardCorners` & `cv2.calibrateCamera` to calibrate camera

As a rusult I can get `mtx`, `dist`

Using `mtx`, `dist`, `cv2.undistort` calibrate test image




# Results


![alt text][image3-1]




# Conclusion & Discussion

### 1. data set










