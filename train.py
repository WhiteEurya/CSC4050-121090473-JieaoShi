import torch
import torch.nn as nn
import torch.optim as optim
from encoder.model_utils import weights_init
from itertools import chain
import os

from plugin.models.heads.dghead import DGHead
from build_dataset import build_dataset
from encoder.encoder import CycleGanEncoder


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

def train(model, dataloader, encoder, log_filename="training_logs.txt", checkpoint_dir="checkpoints", load_ckpt = 0):
    
    model.to(device)
    encoder.to(device)
    
    max_runs = 1000
    optimizer = optim.Adam(chain(encoder.parameters(), model.parameters()), lr=1e-3)
    model.train()
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    if load_ckpt != 0 :
        checkpoint_path = os.path.join("checkpoints", f"model_checkpoint_run_{load_ckpt}.pth")
        
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    with open(log_filename, "w") as log_file:
        for runs in range(max_runs):
            log_file.write(f"\n\n=================================\n\nRun {runs}\n")
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
            for i, (img, gt) in progress_bar:
                img = img.to(device)
                for k, tensor_list in gt["det"].items():
                    gt["det"][k] = [tensor.to(device) for tensor in tensor_list]
                gt["gen"] = {k: v.to(device) for k, v in gt["gen"].items()}
                
                embedding = encoder.forward(img)
                input_data = {
                    'bev_embeddings': embedding
                }
                
                try :
                    out, loss_dict = model.forward_train(context=input_data, batch=gt, optimizer=optimizer)
                    total_loss = 0
                    for k in loss_dict:
                        total_loss += loss_dict[k]

                    progress_bar.set_description(f"Run: {runs} Iteration: {i} Loss: {total_loss.item():.4f}")

                    log_file.write(f"iteration {i}, the loss is {total_loss.item()}\n")
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                except Exception as e:
                    path = gt["path"]
                    log_file.write(f"iteration {i}, error occurs at {path}\n")
                    continue
            # Check if the current run is at a run interval and save the model
            if (runs + 1) % 1 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_run_{runs}.pth')
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'run': runs,
                    'loss': total_loss.item()
                }, checkpoint_path)


def train_debug(model, dataloader, encoder):
    max_runs = 3
    optimizer = optim.Adam(chain(encoder.parameters(), model.parameters()), lr=1e-3)
    model.train()
    parameters_info = []
    
    for runs in range(max_runs):
        print(f"\n\n=================================\n\nRun {runs}")
        
        # progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Run {runs}")
        
        for i, path, (img, gt) in enumerate(dataloader):
            # gt_squeezed = {key: value.squeeze(0) if value.dim() > 1 else value for key, value in gt['det'].items()} 
            # gt['det'] = gt_squeezed
            
            print(gt)
            
            embedding = encoder(img) 
            
            input_data = {
                'bev_embeddings': embedding
            }
            
            out, loss_dict = model.forward_train(context=input_data, batch=gt, optimizer=optimizer)
            
            total_loss = 0
            for k in loss_dict:
                total_loss += loss_dict[k]
            
            print("loss = ", total_loss)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            for k, v in model.named_parameters():
                if v.grad is not None:
                    parameters_info.append("{0}:{1}".format(k, torch.max(v.grad)))
                else:
                    parameters_info.append("{0}:{1}".format(k, v.grad))

            # debug(model)
        # for items in parameters_info :
        #     print(items, end = "\n")
        # print("\n\n\n\n")
    
cfg = read_config(cfg_path)
dataloader = build_dataset(cfg)
encoder, model = build_model(cfg)
train(model, dataloader, encoder)