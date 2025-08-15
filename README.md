Automated Circular Object Detection Project
====
This project is my attempt to develop an automated algorithm for locating the center of a circular metallic object and computing the rotation angle of its notch, based on some images provided for a programming test. I explored different methods like Hough Transform and YOLO, and documented the process.
Required Packages
To run the code, you need to install these packages:

OpenCV: pip install opencv-python
NumPy: pip install numpy
Ultralytics (for YOLO): pip install ultralytics

How to Run
----
Clone the repository and run the scripts. Each script supports command line input with --input for path (default is '/kaggle/input/houghtest/'). For example:

python hough_rough.py --input /path/to/image.jpg
Or run without args to use default path.

The scripts will print the center and angle for each image.

Exploration Summary
----
I started with a basic Hough method, then added preprocessing, and tried YOLO for better accuracy. The basic Hough was consistent but angles were wrong; YOLO gave better results but slight deviations.

TO DO LIST
----
Hybrid method is under discovering.
