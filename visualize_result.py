import pickle
import matplotlib.pyplot as plt
import torch

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def draw_polylines(polylines):
    plt.figure(figsize=(10, 10))
    for polyline in polylines:
        # 如果张量在GPU上，先转移到CPU
        if polyline.is_cuda:
            polyline = polyline.cpu()
            
        # 如果长度是奇数，忽略最后一个元素
        if len(polyline) % 2 != 0:
            polyline = polyline[:-1]
        
        # 将polyline张量转换为NumPy数组
        polyline = polyline.numpy()
        
        # 将扁平的polyline分割为(x, y)坐标
        x_coords = polyline[0::2]  # 取出x坐标，即偶数索引位置的值
        y_coords = polyline[1::2]  # 取出y坐标，即奇数索引位置的值
        
        # 绘制线条和点
        plt.plot(x_coords, y_coords, marker='o')  # 绘制整个polyline
    
    plt.gca().set_aspect('equal', adjustable='box')  # 设置等比例的坐标轴
    plt.show()


file_path = 'partial_test_results.pkl'  # 替换为你的 .pkl 文件路径
loaded_data = load_pickle(file_path)

print(loaded_data)

polyline_tensor = loaded_data[0][1][0]["lines"]

# Create a new figure
plt.figure(figsize=(10, 10))

# Plot each polyline
for polyline in polyline_tensor:
    plt.plot(polyline[:, 0], polyline[:, 1], marker='o')  # Adjust marker styles as desired

# Set the limits of the plot
plt.xlim(0, 1)
plt.ylim(0, 1)

# Optionally set grid, labels and title
plt.grid(True)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Visualization of Polylines')

# Show the plot
plt.show()

# draw_polylines(polyline_tensor)

torch.set_printoptions(threshold=10000)
# print(loaded_data)

# 现在 loaded_data 包含了 pickle 文件中的数据