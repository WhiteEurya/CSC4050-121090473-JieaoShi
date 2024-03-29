from plugin.models.heads import PolylineGenerator
import torch
import numpy as np
from tools.get_keypoint import get_key_bbox
import re
import matplotlib.pyplot as plt

ckpt_path = './work_dirs/vectormapnet/vectormapnet.pth'
data_path = "/mnt/c/Users/xiaoy/Desktop/001.png"
tokens = ["001"]

in_channels = 128
decoder_config = {
    'layer_config': {
        'd_model': 256, 
        'nhead': 8, 
        'dim_feedforward': 512, 
        'dropout': 0.2, 
        'norm_first': True, 
        're_zero': True
    }, 
    'num_layers': 6
}

def debug(gen_net_params) :
    for name, param in gen_net_params.items():
        print(f"Param name: {name}")
        print(f"Param shape: {param.size()}")
        print(f"Param values: {param}")
        print()

## Build the model
def build_model(ckpt_path) :
    state_dict = torch.load(ckpt_path)["state_dict"]
    gen_net_params = {k.replace('head.gen_net.', ''): v for k, v in state_dict.items() if k.startswith('head.gen_net')}
    
    # debug(gen_net_params)
    
    gen_net_model = PolylineGenerator(in_channels = in_channels, encoder_config = None, decoder_config = decoder_config, canvas_size=(200, 100), decoder_cross_attention=False, num_classes=3)
    gen_net_model.load_state_dict(gen_net_params)
    return gen_net_model


def read_data(data_path) :
    return get_key_bbox(data_path)

def trans_into_polyline(model, preds, tokens) :
    range_size = model.canvas_size.cpu().numpy()
    coord_dim = model.coord_dim
    
    ret_list = []
    ret_dict_single = {}
    
    print(preds)
    
    for batch_idx in range(len(tokens)):

        # for gen results.
        batch2seq = np.nonzero(
            preds['lines_bs_idx'].cpu().numpy() == batch_idx)[0]

        bbox_res = {
                'bboxes': preds['bbox'][batch_idx].detach().cpu().numpy(),
                'token': tokens[batch_idx],
                'labels': preds['labels'][batch_idx].detach().cpu().numpy(),
            }
        ret_dict_single.update(bbox_res)

        ret_dict_single.update({
                'nline': len(batch2seq),
                'lines': []
        })

        for i in batch2seq:

            pre = preds['polylines'][i].detach().cpu().numpy()
            pre_msk = preds['polyline_masks'][i].detach().cpu().numpy()
            valid_idx = np.nonzero(pre_msk)[0][:-1]

            # From [200,1] to [199,0] to (1,0)
            line = (pre[valid_idx].reshape(-1, coord_dim) - 1) / (range_size-1)

            ret_dict_single['lines'].append(line)

        ret_list.append(ret_dict_single)

    return ret_list

def generate_polyline(data, model, context) :
    model.training = False
    result = model.forward(data, context = context)
    return result

def get_context() :
    with open('logs.txt', 'r') as file:
        content = file.read()

    numbers = re.findall(r"[-+]?\d*\.?\d+e?[-+]?\d*", content)
    numbers = [float(num) for num in numbers]

    expected_size = 1 * 128 * 50 * 25

    if len(numbers) == expected_size + 1:
        numbers = numbers[:-1]
    elif len(numbers) == expected_size:
        pass
    # else:
        # raise ValueError("The number of numeric values in the file does not match the expected tensor size.")
    tensor = torch.tensor(numbers).view(1, 128, 50, 25)

    # 创建字典
    result_dict = {"bev_embeddings": tensor}
        
    return result_dict

def visiualize_result(result) :
    fig, ax = plt.subplots()
    for line in result["lines"] :
        x_values = line[:, 0]  # 所有点的 x 坐标
        y_values = line[:, 1]  # 所有点的 y 坐标
        ax.plot(x_values, y_values)  # 绘制线条
    plt.savefig("./generate_result.png")
    plt.show()

if __name__ == "__main__" :
    model = build_model(ckpt_path)
    data = read_data(data_path)
    context = get_context()
    preds = generate_polyline(data, model, context)
    ret_list = trans_into_polyline(model, preds | data, tokens)
    print(ret_list)
    for single_result in ret_list :
        visiualize_result(single_result)
    

