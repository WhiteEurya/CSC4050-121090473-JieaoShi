import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch
import cv2
import numpy as np

class VectorMapDataset(Dataset):
    def __init__(self, input_img_dir, gt_img_dir, transform=None):
        self.input_img_dir = input_img_dir
        self.gt_img_dir = gt_img_dir
        self.transform = transform

        # 假设文件名在两个目录中是对应的
        self.filenames = [f for f in os.listdir(input_img_dir) 
                          if os.path.isfile(os.path.join(input_img_dir, f))]

    def __len__(self):
        return len(self.filenames)

    def process_gt_image(self, gt_img):
        # 识别图中的所有白线，寻找最小外接矩形
        gt_img_cv = np.array(gt_img)[:, :, ::-1]
        gray = cv2.cvtColor(gt_img_cv, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        LENGTH_THRESHOLD = 50  # 设置长度阈值

        class_labels = []
        batch_idx = []  # 可以为空，但是为了匹配det字典结构，我们将其初始化为空列表
        bboxes = []

        lines_bs_idx = []
        bbox_flats = []
        polylines = []
        polyline_masks = []
        polyline_weights = []

        for contour in contours:
            # 计算最小外接矩形并分类线条
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            length = cv2.arcLength(contour, True)
            line_cls = 2 if length > LENGTH_THRESHOLD else 1

            # 构建det字典的内容
            class_labels.append(line_cls)
            x_min, y_min = np.min(box, axis=0)
            x_max, y_max = np.max(box, axis=0)
            bboxes.append([x_min, y_min, x_max, y_max])

            # 构建gen字典的内容
            lines_bs_idx.append(0)
            bbox_flat = [x_min, y_min, x_max, y_max]
            bbox_flats.append(bbox_flat)
            polyline = bbox_flat + [0]
            polylines.append(polyline)
            polyline_mask = [True] * len(polyline)
            polyline_masks.append(polyline_mask)
            polyline_weight = [1.0 / len(polyline)] * len(polyline)
            polyline_weights.append(polyline_weight)

        # 转换为张量
        det = {
            'class_label': torch.tensor(class_labels, dtype=torch.long),
            'batch_idx': torch.tensor(batch_idx, dtype=torch.long),  # 如果需要，此处应为相应的batch索引
            'bbox': torch.tensor(bboxes, dtype=torch.float32)
        }

        gen = {
            'lines_bs_idx': torch.tensor(lines_bs_idx, dtype=torch.long),
            'bbox_flat': torch.tensor(bbox_flats, dtype=torch.int32),
            'polylines': torch.tensor(polylines, dtype=torch.int32),
            'polyline_masks': torch.tensor(polyline_masks, dtype=torch.bool),
            'polyline_weights': torch.tensor(polyline_weights, dtype=torch.float32)
        }

        return det, gen

    def __getitem__(self, idx):
        # 定位到当前索引的文件名
        filename = self.filenames[idx]

        # 加载鸟瞰图和修正后的图像
        input_img_path = os.path.join(self.input_img_dir, filename)
        gt_img_path = os.path.join(self.gt_img_dir, filename)

        input_img = Image.open(input_img_path).convert('RGB')
        gt_img = Image.open(gt_img_path).convert('RGB')

        # 应用变换（如果有）
        if self.transform is not None:
            processed_input_img = self.transform(input_img)
            processed_gt_img = self.transform(gt_img)
        
        det, gen = self.process_gt_image(gt_img)
        gt_data = {
            'processed_gt_img': processed_gt_img,
            'det': det,
            'gen': gen
        }

        return processed_input_img, gt_data

def build_dataset(cfg) :
    input_img_dir = cfg["input_img_dir"]
    gt_img_dir = cfg["gt_img_dir"]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = VectorMapDataset(
        input_img_dir=input_img_dir,
        gt_img_dir=gt_img_dir,
        transform=transform
    )

    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


