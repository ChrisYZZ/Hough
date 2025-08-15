import cv2
import numpy as np
import math
import os

# I'm defining the dataset path at the top so I can easily change it during testing
dataset_path = '/kaggle/input/houghtest/'  # Change this to your image path or directory

# Function to process a single image automatically - this is my basic attempt
def detect_center_and_angle(image_path):
    # Load image - hoping it reads correctly
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    # Preprocess: grayscale and blur - helps reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect circles using Hough Transform - trying with these params
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=50, maxRadius=200)
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        # Assume the largest circle is the target - sorting by radius
        x, y, r = sorted(circles, key=lambda c: c[2], reverse=True)[0]
        center = (x, y)
    else:
        return None, None
    
    # Edge detection for notch - using Canny to find boundaries
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours and identify notch - simplified: find the point with max deviation
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    notch_x, notch_y = None, None
    max_dist = 0
    for cnt in contours:
        for point in cnt:
            px, py = point[0]
            dist = math.sqrt((px - x)**2 + (py - y)**2) - r  # Deviation from radius
            if abs(dist) > max_dist:  # Assume notch has larger deviation
                max_dist = abs(dist)
                notch_x, notch_y = px, py
    
    if notch_x is not None:
        # Calculate angle: from center to notch, relative to positive x-axis
        dx = notch_x - x
        dy = notch_y - y
        angle = math.degrees(math.atan2(dy, dx)) 
        if angle < 0:
            angle += 360  # Normalize to 0-360 degrees

    
    return center, angle

# Main execution: Process directory or single file
if __name__ == "__main__":
    if os.path.isdir(dataset_path):
        for filename in os.listdir(dataset_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(dataset_path, filename)
                center, angle = detect_center_and_angle(image_path)
                print(f"Image: {filename}")
                print(f"Center: {center}")
                print(f"Rotation Angle: {angle} degrees" if angle else "Not detected")
    elif os.path.isfile(dataset_path):
        center, angle = detect_center_and_angle(dataset_path)
        print(f"Image: {os.path.basename(dataset_path)}")
        print(f"Center: {center}")
        print(f"Rotation Angle: {angle} degrees" if angle else "Not detected")
    else:
        print("Invalid dataset path")