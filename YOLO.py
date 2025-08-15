import cv2
import numpy as np
import math
import os
from ultralytics import YOLO

# Default path for easy testing
dataset_path = '/kaggle/input/houghtest/'  # Change this to your image path or directory

# Independent preprocessing function - trying to make it robust step by step
def preprocess_image(img):
    # Convert to grayscale to simplify - this seems basic but important
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Median blur to remove salt-pepper noise and reflections - added this after seeing reflections in images
    median_blurred = cv2.medianBlur(gray, 5)
    # Gaussian blur for general smoothing - helps with overall noise
    blurred = cv2.GaussianBlur(median_blurred, (5, 5), 0)
    # Adaptive thresholding to handle varying lighting/shadows - this was key for uneven light
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Morphological closing to fill small gaps/noise in edges - trying to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Convert back to RGB for YOLO input - YOLO needs 3 channels
    preprocessed = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
    return preprocessed

# Main detection function using YOLO - my first try with DL
def detect_center_and_angle(image_path):
    # Load image - checking if it loads properly
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    # Apply preprocessing - this step improved detection a lot
    preprocessed = preprocess_image(img)
    
    # Load pre-trained YOLO model - using yolo11n for speed
    model = YOLO("yolo11n.pt")  # Hoping this recognizes the circle as 'clock'
    
    # Run inference - detecting the object
    results = model(preprocessed)  # Returns detections
    
    # Assume first detection is the circle object - simplifying for now
    if len(results) > 0 and len(results[0].boxes) > 0:
        box = results[0].boxes[0]  # Get bounding box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Box coordinates
        center = ((x1 + x2) / 2, (y1 + y2) / 2)  # Center as midpoint
        
        # For notch: Use edge analysis on cropped box - trying a simple way to find the notch
        crop = preprocessed[int(y1):int(y2), int(x1):int(x2)]
        edges = cv2.Canny(crop, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            # Find notch as min x in contour - assuming upper-left notch for my images
            notch_idx = np.argmin(cnt[:, 0, 0])  # Simplified: leftmost point
            notch_x_rel, notch_y_rel = cnt[notch_idx][0]
            notch_x = x1 + notch_x_rel
            notch_y = y1 + notch_y_rel
            
            # Calculate angle - from center to notch
            dx = notch_x - center[0]
            dy = notch_y - center[1]
            angle = math.degrees(math.atan2(dy, dx))
            if angle < 0:
                angle += 360
        else:
            angle = None
    else:
        center, angle = None, None
    
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