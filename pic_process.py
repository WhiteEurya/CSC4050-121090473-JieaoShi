import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

def get_bounding_box(contour):
    x, y, width, height = cv2.boundingRect(contour)
    return (x, y + height, x + width, y)  # Left-bottom and right-top coordinates

def classify_contour(contour):
    _, _, width, height = cv2.boundingRect(contour)
    if max(width, height) <= 5 : return -1
    return 2 if max(width, height) >= 100 else 1

def get_polyline(contour, cls):
    n = 4 if cls == 2 else 2
    p1, p2 = get_extreme_points(contour)
    division_points = [((p2 - p1) * i / (n + 1)) + p1 for i in range(1, n + 1)]
    return flatten_and_pad(division_points)

def get_extreme_points(contour):
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    min_point = min(approx, key=lambda x: x[0][1])
    max_point = max(approx, key=lambda x: x[0][1])
    return min_point[0], max_point[0]

def flatten_and_pad(points):
    flat_points = [coord for point in points for coord in point]
    return flat_points

def points_too_close(points, threshold=5):
    for i in range(1, len(points)):
        if np.linalg.norm(np.array(points[i]) - np.array(points[i-1])) < threshold:
            return True
    return False

def replace_zeros_and_check_proximity(polyline):
    proximity_threshold = 10  
    
    if points_too_close(polyline, proximity_threshold):
        return None

    return [max(val, 1) for val in polyline[:-1]] + [polyline[-1]]

def process_img(img):
    _, binary_image = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox = []
    cls_label = []
    polyline = []
    mask = []
    value = []

    for contour in contours:
        cls = classify_contour(contour)
        if cls == -1 : continue
        bbox.append(get_bounding_box(contour))
        cls_label.append(cls)
        single_polyline = get_polyline(contour, cls)
        processed_polyline = replace_zeros_and_check_proximity(single_polyline)
        if processed_polyline is None:
            continue 
        polyline.append(processed_polyline + [0]) 
        mask.append([True] * len(processed_polyline) + [True])
        value.append([1 / (cls * (cls + 1))] * len(processed_polyline) + [0])
    
    # Find the length of longest polyline array
    try :
        max_length = max(len(pl) for pl in polyline)
    except :
        max_length = 0
    
    # Pad all polyline, mask, and value arrays to be of the same length
    for i in range(len(polyline)):
        additional_padding = [0] * (max_length - len(polyline[i]))
        polyline[i].extend(additional_padding)
        mask[i].extend([False] * len(additional_padding))
        value[i].extend(additional_padding)
    
    idx = [0] * len(cls_label)
    
    # print(polyline, mask, value, cls_label, idx, bbox)
    
    return polyline, mask, value, cls_label, idx, bbox


if __name__ == "__main__" :
    # Load an image from file or capture from camera
    image_path = 'dataset/train_demo/masks/3_1679459118803000.png'  # replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image loaded successfully
    if image is None:
        print("Error: Could not load image.")
        exit()

    # Process the image to extract lines and bounding boxes
    output_array, X, X_value, line_cls, idx, lines_corners = process_img(image)