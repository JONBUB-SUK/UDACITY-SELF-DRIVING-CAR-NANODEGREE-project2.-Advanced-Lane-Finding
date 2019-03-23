# UDACITY-SELF-DRIVING-CAR-NANODEGREE-project2.-Advanced-Lane-Finding


[//]: # (Image References)

[image1-1]: ./images/1.1_Percaptron,Lambda.JPG "RESULT1"


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

#### 1. Define cam_cal function : find mtx, dist using chessboard images

```
def cam_cal(cam_cal_img_gray):
    
    nx = 9
    ny = 5

    ret, corners = cv2.findChessboardCorners(cam_cal_img_gray, (nx,ny), None)

    if ret == True:
        cv2.drawChessboardCorners(cam_cal_img, (nx,ny), corners, ret)
        #plt.imshow(cam_cal_img)

    objpoints = []
    imgpoints = []

    objp = np.zeros((9*5, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:5].T.reshape(-1,2)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, cam_cal_img_gray.shape[::-1], None, None)
    
    return ret, mtx, dist, rvecs, tvecs
```


#### 2. Define undistort function : undistort camera pictured image using cv2.undistort

```
def undistort_img(img_gray, mtx, dist):
    
    undistorted_img = cv2.undistort(img_gray, mtx, dist, None, mtx)
    
    return undistorted_img
```

(이미지 : before calibration)

(이미지 : after calibration)

(이미지 : before calibration test)

(이미지 : after calibration test)


### 2. Threshold image




# Results


![alt text][image3-1]




# Conclusion & Discussion

### 1. data set










