This repository contains my bachelor’s thesis project: a Python-based system for estimating the camera position and orientation in 3D space from a single 2D image of a football pitch. The main output is a transformation/homography matrix that allows accurate projection of 2D image points onto the real-world pitch, enabling overlays, AR graphics, and analytics on the playing field.

The approach combines classical computer-vision techniques — feature detection and matching, homography estimation, and pose estimation (solvePnP) — with practical engineering: preprocessing, calibration routines, and visualization tools. Implemented using OpenCV, NumPy and Matplotlib, the repo includes example images.

Key features

Automatic estimation of homography and camera pose from single images.

Visualization of projected overlays on the pitch for validation.


Tech stack / libraries

Python 3.x

OpenCV (cv2)

NumPy, SciPy

Matplotlib 
