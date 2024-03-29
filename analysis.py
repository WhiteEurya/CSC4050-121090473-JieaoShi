import matplotlib.pyplot as plt
import re
import os

# 确保有一个用于保存图像的目录
output_dir = 'run_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打开文件，并按行读取
with open('training_logs.txt', 'r') as file:
    lines = file.readlines()

# 初始化存储数据的字典
run_data = {}
current_run = None

# 解析文件，提取数据
for line in lines:
    run_match = re.match(r'Run (\d+)', line)
    if run_match:
        current_run = int(run_match.group(1))
        run_data[current_run] = {'iterations': [], 'losses': []}
    else:
        iter_match = re.match(r'iteration (\d+), the loss is (\d+\.\d+)', line)
        if iter_match and current_run is not None:
            iteration, loss = map(float, iter_match.groups())
            run_data[current_run]['iterations'].append(iteration)
            run_data[current_run]['losses'].append(loss)

# 为每个运行绘制图像并保存
for run_number, data in run_data.items():
    plt.figure(figsize=(10, 5))
    plt.plot(data['iterations'], data['losses'], label=f'Run {run_number}')
    plt.title(f'Loss Curve for Run {run_number}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    # 保存图像到指定目录
    plt.savefig(f'{output_dir}/run_{run_number}.png')
    plt.close()  # 关闭图形以释放内存