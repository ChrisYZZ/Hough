import cv2
import numpy as np
import math
import os

# Default path for convenience
dataset_path = '/kaggle/input/houghtest/'  # Change this to your image path or directory

# Independent preprocessing function - building this to make Hough more robust
def preprocess_image(img):
    # Convert to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Median blur 
    median_blurred = cv2.medianBlur(gray, 5)
    # Gaussian blur 
    blurred = cv2.GaussianBlur(median_blurred, (5, 5), 0)
    # Adaptive thresholding 
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Morphological closing 
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closed

# Main detection function - enhanced version after basic one
def detect_center_and_angle(image_path):
    # Load image - same as before
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    # Apply preprocessing - this should help with noise
    preprocessed = preprocess_image(img)
    
    # Detect circles - tuning params based on tests
    circles = cv2.HoughCircles(preprocessed, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=50, maxRadius=200)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Select largest circle - assuming it's the main object
        x, y, r = sorted(circles, key=lambda c: c[2], reverse=True)[0]
        center = (x, y)
    else:
        return None, None  # If no circle, stop here
    
    # Edge detection - on preprocessed for better results
    edges = cv2.Canny(preprocessed, 50, 150)
    
    # Find notch - approximating contours to reduce noise
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    notch_x, notch_y = None, None
    max_deviation = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), closed=True)
        for point in approx:
            px, py = point[0]
            dist = math.sqrt((px - x)**2 + (py - y)**2) - r  # Deviation check
            if abs(dist) > max_deviation and abs(dist) > 5:  # Ignoring small noise
                max_deviation = abs(dist)
                notch_x, notch_y = px, py
    
    if notch_x is not None:
        # Angle calc - normalizing as usual
        dx = notch_x - x
        dy = notch_y - y
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
    else:
        angle = None
    
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