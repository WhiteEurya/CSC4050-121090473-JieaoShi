import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

# 函数：在线段上等距放置点
def place_points_on_line(p1, p2, num_points):
    return np.linspace(p1, p2, num=num_points, endpoint=False)

# 函数：计算两线段之间的最小距离
def min_distance_between_lines(line1, line2):
    min_dist = float('inf')
    for point1 in line1:
        for point2 in line2:
            dist = np.linalg.norm(point1 - point2)
            min_dist = min(min_dist, dist)
    return min_dist

# 函数：删除相近的线段
def remove_close_lines(lines, min_distance):
    keep = [True] * len(lines)  # 初始化所有线段都保留
    for i in range(len(lines)):
        if not keep[i]:
            continue
        for j in range(i+1, len(lines)):
            if keep[j] and min_distance_between_lines(lines[i], lines[j]) < min_distance:
                keep[j] = False  # 删除与线段i相近的线段j
    return [line for k, line in enumerate(lines) if keep[k]]

def find_extreme_points_of_line(line_points):
    # 转换成 NumPy 数组以便于处理
    line_points_np = np.array(line_points)
    # 最左下角的点（最小的x和最大的y，因为图像坐标系y是向下的）
    bottom_left = line_points_np[np.lexsort((line_points_np[:,1], line_points_np[:,0]))][0]
    # 最右上角的点（最大的x和最小的y）
    top_right = line_points_np[np.lexsort((-line_points_np[:,1], line_points_np[:,0]))][-1]
    return np.array([bottom_left, top_right])

def process_img(image):
    # 应用高斯模糊以减少噪声
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # 使用Canny边缘检测
    edges = cv2.Canny(blurred_image, 100, 200)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 存储每个轮廓的等分点的二维数组
    contours_line_points = []

    # 找到最大的线段点数和最长线段的长度
    max_points = 0
    longest_line_length = 0
    for contour in contours:
        points = contour.squeeze()
        if points.ndim == 1 or len(points) < 2:
            continue
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        distance = np.linalg.norm(np.array(leftmost) - np.array(rightmost))
        num_points = max(int(distance // 30), 2)  # 使用20作为点间的基础距离
        max_points = max(max_points, num_points)
        longest_line_length = max(longest_line_length, distance)

    # 计算最长线段的点间最小距离
    min_distance_longest_line = longest_line_length / (max_points - 1) if max_points > 1 else 0

    # 对每个轮廓进行操作，并填充到相同长度
    for contour in contours:
        points = contour.squeeze()
        if points.ndim == 1 or len(points) < 2:
            continue
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        distance = np.linalg.norm(np.array(leftmost) - np.array(rightmost))
        # 确保点间距不小于最大线段的最小距离
        num_points = max(int(distance // max(min_distance_longest_line, 20)), 2)
        line_points = place_points_on_line(np.array(leftmost), np.array(rightmost), num_points)
        contours_line_points.append(line_points)

    # 删除相近的线段
    min_dist_threshold = 5
    filtered_line_points = remove_close_lines(contours_line_points, min_dist_threshold)

    output_array = []
    X = []
    X_value = []
    line_cls = []  # 这个数组将保存每条线段的分类

    max_points = max(len(line_points) for line_points in filtered_line_points)

    for line_points in filtered_line_points:
        # 计算实际点的数量
        actual_points_count = len(line_points)

        # 创建一个临时数组用于存储整数输出和布尔值
        temp_output = np.zeros((max_points, 2), dtype=int)
        temp_X = np.zeros((max_points, 2), dtype=bool)
        temp_X_value = np.zeros((max_points) * 2)  # 初始化X_value数组

        # 填充实际的点，并设置布尔数组的对应位置为True
        for i, point in enumerate(line_points):
            temp_output[i] = point
            temp_X[i] = [True, True]

        # 将临时数组转换为一维，然后添加到最终数组中
        temp_output = temp_output.flatten()
        
        temp_X = temp_X.flatten()
        

        for j in range(len(temp_X)):
            if temp_X[j]:
                temp_output[j] = 1 if temp_output[j] == 0 else temp_output[j]
                
        temp_output = np.append(temp_output, 0)  # 在数组末尾添加一个0
        temp_X = np.append(temp_X, False) 
        temp_X[actual_points_count * 2] = True
    
        # 线段分类
        threshold_length = 15
        cls = 2 if actual_points_count > threshold_length else 1
        line_cls.append(cls)

        # 计算X_value
        # 为每个非填充值的点赋值为1/N，填充值为0
        value = 1.0 / actual_points_count if actual_points_count > 0 else 0
        temp_X_value[:actual_points_count * 2] = value

        output_array.append(temp_output)
        X.append(temp_X)
        temp_X_value = np.append(temp_X_value, 0.00)
        X_value.append(temp_X_value)
    
    idx = np.zeros(len(line_cls), dtype=int)
    
     # 在这里初始化一个新的列表来存储线段的角点
    lines_corners = []

    for line_points in filtered_line_points:
        
        # 对于每个过滤后的线段，找到它的左下角和右上角的点
        corners = find_extreme_points_of_line(line_points)
        # 将浮点数坐标转换为整数，并扁平化后添加到列表中
        corners_int = np.round(corners).astype(int).flatten()
        lines_corners.append(corners_int)
        
        # Create a copy of the image to draw on
        # Create copies of the image to draw on
    image_with_points = image.copy()
    image_with_boxes = image.copy()

    # Draw each point on the lines on one image copy
    for line_points in output_array:
        for i in range(0, len(line_points), 2):
            # Check if we should draw this point (X array contains True at this point's index)
            if i // 2 < len(X) and X[i // 2][i % 2]:
                x, y = line_points[i], line_points[i+1]
                # Draw the point on the image
                cv2.circle(image_with_points, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    # Draw bounding boxes around lines in purple on another image copy
    for line in lines_corners:
        # Unpack the line's corner points
        x1, y1, x2, y2 = line
        # Calculate the top left and bottom right points for the rectangle
        top_left = (x1, y1)
        bottom_right = (x2, y2)
        # Draw the rectangle on the image
        cv2.rectangle(image_with_boxes, top_left, bottom_right, (255, 0, 255), 2)

    # Display the images
    cv2.imshow('Points', image_with_points)
    cv2.imshow('Bounding Boxes', image_with_boxes)
    cv2.waitKey(0)  # Wait for a key press to close the windows
    cv2.destroyAllWindows()  # Close all OpenCV windows

    return output_array, X, X_value, line_cls, idx, lines_corners

if __name__ == "__main__" :
    # Load an image from file or capture from camera
    image_path = 'dataset/train_demo/masks/1_1678933258433000.png'  # replace with your image path
    image = cv2.imread(image_path)

    # Check if image loaded successfully
    if image is None:
        print("Error: Could not load image.")
        exit()

    # Process the image to extract lines and bounding boxes
    output_array, X, X_value, line_cls, idx, lines_corners = process_img(image)