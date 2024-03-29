import torch
import torch.nn as nn
import torch.optim as optim
from encoder.model_utils import weights_init
from itertools import chain
import os

from plugin.models.heads.dghead import DGHead
from build_dataset import build_dataset
from encoder.encoder import CycleGanEncoder
import pickle


cfg_path = "config/vectormapnet.py"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(cfg) :
    
    encoder = CycleGanEncoder(cfg["encoder_cfg"])
    encoder.apply(weights_init)
    
    model = DGHead(
    det_net_cfg = cfg["model"]["head_cfg"]["det_net_cfg"],
    gen_net_cfg = cfg["model"]["head_cfg"]["gen_net_cfg"],
    max_num_vertices = 50,
    top_p_gen_model = 0.9,
    sync_cls_avg_factor = True,
    augmentation = False,
    augmentation_kwargs = {'p': 0.3, 'scale': 0.01, 'bbox_type': 'xyxy'},
    )
    return encoder, model

def read_config(cfg_path) :
    # 定义一个空的字典来存放配置变量
    cfg = {}
    with open(cfg_path, "r", encoding = "utf-8") as f:
        exec(f.read(), cfg)
    return cfg
    
    
def debug(model) :
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient of {name}: {param.grad}")
        else:
            print(f"Gradient of {name} is None")
    
from tqdm import tqdm
import torch.optim as optim

import torch
from tqdm import tqdm
import os
import pprint
from matplotlib import pyplot as plt
from torchviz import make_dot

def test(model, dataloader, encoder, checkpoint_path_encoder, checkpoint_path, log_filename="test_logs.txt",  result_filename="partial_test_results.pkl"):
    
    model.to(device)
    encoder.to(device)
    partial_results = {}
    
    if os.path.isfile(checkpoint_path):
        checkpoint_encoder = torch.load(checkpoint_path_encoder)
        checkpoint = torch.load(checkpoint_path)
        head_parameters = {}
        for key in checkpoint["state_dict"]:
            if key.startswith('head.'):
                new_key = key.split('head.', 1)[1]
                if not new_key.startswith('augmentation') :
                    head_parameters[new_key] = checkpoint["state_dict"][key]
        encoder.load_state_dict(checkpoint_encoder['encoder_state_dict'])
        model.load_state_dict(head_parameters)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    model.eval()
    
    with torch.no_grad(), open(log_filename, "w") as log_file:
        for i, (img, gt) in tqdm(enumerate(dataloader), total=len(dataloader)):
            img = img.to(device)
            for k, tensor_list in gt["det"].items():
                gt["det"][k] = [tensor.to(device) for tensor in tensor_list]
            gt["gen"] = {k: v.to(device) for k, v in gt["gen"].items()}
            
            embedding = encoder.forward(img)
            
            input_data = {
                'bev_embeddings': embedding
            }
            
            # 仅执行前向传播以获取预测结果
            out, res = model.inference(context=input_data, batch=gt)
            
            if i == 0 or (i % 100) == 0:
                partial_results[i] = out, res
            
            # print(res)    
            
            with open(result_filename, "wb") as result_file:
                pickle.dump(partial_results, result_file)
                
            raise()
            

cfg = read_config(cfg_path)
dataloader = build_dataset(cfg)
encoder, model = build_model(cfg)

run_number = 1
checkpoint_path_encoder = os.path.join("checkpoints", f"model_checkpoint_run_{run_number}.pth")
checkpoint_path = os.path.join("checkpoints", f"vectormapnet.pth")
test(model, dataloader, encoder, checkpoint_path_encoder, checkpoint_path)
    
    
