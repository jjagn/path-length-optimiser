import cv2
import numpy as np


max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)

def detect_circles(image_path, 
                   min_radius=10, 
                   max_radius=100, 
                   color_lower=None, 
                   color_upper=None,
                   min_circularity=0.8):
    """
    Detect circles in an image with multiple filtering criteria.
    
    Parameters:
    - image_path: Path to the input image
    - min_radius: Minimum radius of circles to detect
    - max_radius: Maximum radius of circles to detect
    - color_lower: Lower HSV color bounds for color filtering (optional)
    - color_upper: Upper HSV color bounds for color filtering (optional)
    - min_circularity: Minimum circularity threshold (1.0 is a perfect circle)
    
    Returns:
    - List of detected circles (x, y, radius)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale for circle detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.waitKey(0)
    
    # Optional color filtering
    if color_lower is not None and color_upper is not None:
        # Convert to HSV for color filtering
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.imshow('hsv', hsv)
        color_mask = cv2.inRange(hsv, color_lower, color_upper)
        cv2.imshow('color_mask', color_mask)
        gray = cv2.bitwise_and(gray, gray, mask=color_mask)
        cv2.imshow('gray', gray)
        cv2.waitKey(0)
    
    scale = 0.6
    width = int(gray.shape[1] * scale)
    height = int(gray.shape[0] * scale)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    lower = cv2.resize(blurred, (width, height))

    # print("blurred image")
    # cv2.imshow('blurred image', blurred)
    # cv2.waitKey(0)
    #
    # print("shrunk image")
    # cv2.imshow('shrunk', lower)
    # cv2.waitKey(0)
    
    # Detect circles using Hough Circle Transform
    print("Detecting circles...")
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20,
        param1=50,   # Higher threshold for edge detection
        param2=30,   # Accumulator threshold for circle detection
        minRadius=min_radius, 
        maxRadius=max_radius
    )
    
    # If no circles found, return empty list
    if circles is None:
        return []
    
    # Convert circles to integer coordinates
    circles = np.uint16(np.around(circles))
    
    # Filter circles based on circularity
    filtered_circles = []
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(image, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(image, center, radius, (255, 0, 255), 3)
        filtered_circles.append(i)
    return filtered_circles

def visualize_circles(image_path, circles):
    """
    Draw detected circles on the image for visualization.
    
    Parameters:
    - image_path: Path to the input image
    - circles: List of detected circles (x, y, radius)
    
    Returns:
    - Image with circles drawn
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Draw detected circles
    for (x, y, r) in circles:
        # Outer circle
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        # Center point
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
    
    return image

# Example usage
if __name__ == "__main__":
    # 99, 31, 11
    # threshold
    image_path = "./download.png"
    high_H = 7
    low_H = 0
    high_S = 255
    low_S = 74
    high_V = 161
    low_V = 29
    cv2.namedWindow(window_detection_name)
    cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
    cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
    cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
    cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
    cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
    cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

    frame = cv2.imread(image_path)
    while True:
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        
        cv2.imshow(window_detection_name, frame_threshold)
        
        key = cv2.waitKey(30)
        if key == ord('q') or key == 27:
            break
    desired_red = [100, 30, 10]
    red_lower = np.array([low_H, low_S, low_V])
    red_upper = np.array([high_H, high_S, high_V])
    
    # Detect circles
    detected_circles = detect_circles(
        image_path, 
        min_radius=10, 
        max_radius=30, 
        color_lower=red_lower, 
        color_upper=red_upper,
        min_circularity=0.5
    )

    # Print detected circles
    print(f"Detected {len(detected_circles)} circles:")
    for (x, y, r) in detected_circles:
        print(f"Circle at ({x}, {y}) with radius {r}")
    
    # Visualize circles
    result_image = visualize_circles(image_path, detected_circles)
    cv2.imshow("Detected Circles", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
