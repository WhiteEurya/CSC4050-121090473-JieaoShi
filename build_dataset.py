import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch
import cv2
import numpy as np
from pic_process import process_img
from torch.utils.data._utils.collate import default_collate

class VectorMapDataset(Dataset):
    def __init__(self, input_img_dir, gt_img_dir, transform=None):
        self.input_img_dir = input_img_dir
        self.gt_img_dir = gt_img_dir
        self.transform = transform

        self.filenames = [f for f in os.listdir(input_img_dir) 
                          if os.path.isfile(os.path.join(input_img_dir, f))]

    def __len__(self):
        return len(self.filenames)

    def process_gt_image(self, gt_img):
        original_width, original_height = gt_img.size
        target_size = 200.0
        scale_factor = target_size / max(original_width, original_height)
    
        gt_img = np.array(gt_img)    
    
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        gt_img_cv = cv2.resize(gt_img, (new_width, new_height))
        
        polylines, polyline_masks, polyline_weights, class_labels, lines_idxs, bboxes = process_img(gt_img_cv)

        # gray = cv2.cvtColor(gt_img_cv, cv2.COLOR_RGB2GRAY)
        # _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # kernel = np.ones((3,3), np.uint8)
        # dilated = cv2.dilate(binary, kernel, iterations=2)  
        # contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # LENGTH_THRESHOLD = 100 
        
        # batch_idx = []  
        # bboxes = []
        # bbox_flats = []

        # if len(gt_img_cv.shape) != 3 or gt_img_cv.shape[2] != 3:
        #     if len(gt_img_cv.shape) == 2:
        #         gt_img_cv = cv2.cvtColor(gt_img_cv, cv2.COLOR_GRAY2BGR)
        #     elif gt_img_cv.shape[2] == 4:
        #         gt_img_cv = gt_img_cv[:, :, :3]

        # if not gt_img_cv.flags['C_CONTIGUOUS']:
        #     gt_img_cv = np.ascontiguousarray(gt_img_cv)


        # for contour in contours:
        #     rect = cv2.minAreaRect(contour)
        #     box = cv2.boxPoints(rect)
        #     box = np.int0(box)
            
        #     length = cv2.arcLength(contour, True)
        #     line_cls = 2 if length > LENGTH_THRESHOLD else 1

        #     class_labels.append(line_cls)
        #     x_min, y_min = np.min(box, axis=0).clip(min=0)
        #     x_max, y_max = np.max(box, axis=0).clip(min=0)
        #     bboxes.append([x_min, y_min, x_max, y_max])
            
        #     # color = (0, 255, 0) if line_cls == 1 else (0, 0, 255)  # 类别1使用绿色，类别2使用红色
        #     # thickness = 1
        #     # start_point = (int(x_min), int(y_min))
        #     # end_point = (int(x_max), int(y_max))
        #     # gt_img_cv = cv2.rectangle(gt_img_cv, start_point, end_point, color, thickness)

        #     bbox_flat = [x_min, y_min, x_max, y_max]
        #     bbox_flats.append(bbox_flat)

        # gt_img_cv = cv2.cvtColor(gt_img_cv, cv2.COLOR_BGR2RGB)

        # plt.figure(figsize=(10, 10))
        # plt.imshow(gt_img_cv)
        # plt.title('Visualization of Fitting Results')
        # plt.axis('off')
        # plt.savefig("1.png")
        # raise(rua)

        det = {
            'class_label': torch.tensor(class_labels, dtype=torch.long),
            # 'batch_idx': torch.tensor(batch_idx, dtype=torch.long), 
            'bbox': torch.tensor(bboxes, dtype=torch.float32)
        }

        gen = {
            'lines_cls': torch.tensor(class_labels, dtype=torch.long),
            'lines_bs_idx' : torch.tensor(lines_idxs, dtype=torch.long),
            'bbox_flat': torch.tensor(bboxes, dtype=torch.int32),
            'polylines': torch.tensor(polylines, dtype=torch.int32),
            'polyline_masks': torch.tensor(polyline_masks, dtype=torch.bool),
            'polyline_weights': torch.tensor(polyline_weights, dtype=torch.float32)
        }

        # print("Shapes of tensors in 'det':")
        # for key in det:
        #     print(f"{key}: {det[key].shape}")

        # # 打印gen字典中所有键的形状
        # print("\nShapes of tensors in 'gen':")
        # for key in gen:
        #     print(f"{key}: {gen[key].shape}")
        
        return det, gen

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        input_img_path = os.path.join(self.input_img_dir, filename)
        gt_img_path = os.path.join(self.gt_img_dir, filename)

        input_img = Image.open(input_img_path).convert('RGB')
        gt_img = Image.open(gt_img_path).convert('L')
        

        if self.transform is not None:
            processed_input_img = self.transform(input_img)
            processed_gt_img = self.transform(gt_img)
        
        det, gen = self.process_gt_image(gt_img)
        
        
        gt_data = {
            "path" : input_img_path,
            'processed_gt_img': processed_gt_img,
            'det': det,
            'gen': gen
        }
        
        return processed_input_img, gt_data


def custom_collate_fn(batch):
    # 处理 imgs
    imgs = [item[0] for item in batch]  # 提取所有图像
    imgs = default_collate(imgs)  # 使用默认方法堆叠图像

    # 处理 gts
    # 这里我们假设所有样本的 gt 都有相同的键
    det = {key: [] for key in batch[0][1]["det"].keys()}
    gen = {key: [] for key in batch[0][1]["gen"].keys()}
    
    for _, gt in batch:
        for key, value in gt["det"].items():
            det[key].append(value)

    gt_c = dict(det = det, gen = gt["gen"], path = gt["path"])

    return imgs, gt_c

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    return dataloader


if __name__ == "__main__" :
    print("debuging...")
    cfg = dict(
        input_img_dir = 'dataset/train_demo',
        gt_img_dir = 'dataset/train_demo/masks',
    )
    build_dataset(cfg)