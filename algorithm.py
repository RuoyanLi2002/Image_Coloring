import time
import math
from tqdm import tqdm
from collections import Counter
import numpy as np
import torch
import pyjuice as juice
from functools import partial

from forward_backward_algo import forward, backward
from utils import grey_index_to_rgb_index

def sample_pixel(pc, ns, args, pc_args, flow, grey_index, color, candidate_color):
    red_index, green_index, blue_index = grey_index_to_rgb_index(grey_index, args)
    
    param_array = pc_args["param_array"]
    if color == "red":
        color_val = param_array[red_index, :, :]
    elif color == "green":
        color_val = param_array[green_index, :, :]
    else:
        color_val = param_array[blue_index, :, :] # (64, 256)
    color_val = torch.from_numpy(color_val)

    grey_start, grey_end = pc_args["grey_idx_to_prod_se_array"][grey_index]
    aaa = pc_args["grey_idx_to_prod_se_array"][grey_index]

    flow_val = flow[grey_start : grey_end, ]
    flow_val = flow_val.unsqueeze(-1) # [64, 1]
 
    result = flow_val * color_val[:, candidate_color]
    column_sums = result.sum(dim=0)
            
    max_column_index = torch.multinomial(column_sums, 1).item()
    
    return candidate_color[max_column_index]

def compute_argmax_pixel(pc, ns, args, pc_args, flow, grey_index, color, candidate_color):
    red_index, green_index, blue_index = grey_index_to_rgb_index(grey_index, args)
    
    param_array = pc_args["param_array"]
    if color == "red":
        color_val = param_array[red_index, :, :]
    elif color == "green":
        color_val = param_array[green_index, :, :]
    else:
        color_val = param_array[blue_index, :, :] # (64, 256)
    color_val = torch.from_numpy(color_val)

    grey_start, grey_end = pc_args["grey_idx_to_prod_se_array"][grey_index]
    aaa = pc_args["grey_idx_to_prod_se_array"][grey_index]

    flow_val = flow[grey_start : grey_end, ]
    flow_val = flow_val.unsqueeze(-1) # [64, 1]
 
    result = flow_val * color_val[:, candidate_color]
    column_sums = result.sum(dim=0)
            
    max_column_index = torch.argmax(column_sums).item()
    
    return candidate_color[max_column_index]

def color_patch_algo(pc, ns, args, pc_args, Grey_patch, grey_prob):
    start_time = time.perf_counter()
    grey_lls, backward_node_mars = forward(pc, ns, grey_prob, args)
    end_time = time.perf_counter()
    print(f"Time taken by forward: {(end_time - start_time):.6f} seconds")

    print(f"grey_lls: {grey_lls}")
    
    start_time = time.perf_counter()
    flow = backward(pc, backward_node_mars)
    end_time = time.perf_counter()
    print(f"Time taken by backward: {(end_time - start_time):.6f} seconds")
    flow = flow.cpu()

    color_tensor = torch.zeros(args.data_shape)
    h = args.data_shape[1]
    w = args.data_shape[2]

    arr_valid_ycocg = pc_args["arr_valid_ycocg"]
    arr_num_valid_ycocg = pc_args["arr_num_valid_ycocg"]
    
    progress_bar = tqdm(total=args.data_shape[0]*args.data_shape[1]*args.data_shape[2], desc="Compute Color")
    for y in range(h):
        for z in range(w):
            grey_index = y * w + z
            grey = Grey_patch.flatten()[grey_index]
            valid_ycocg = arr_valid_ycocg[grey]
            valid_ycocg = valid_ycocg[:, :int(arr_num_valid_ycocg[grey])]
            valid_ycocg = valid_ycocg.astype(np.int32) 
            valid_ycocg = torch.from_numpy(valid_ycocg) # [3, 25543]

            for c in range(args.data_shape[0]):
                candidate_color = torch.unique(valid_ycocg[c, :])
                
                color = ["red", "green", "blue"][c]
                if args.argmax:
                    sample_color = compute_argmax_pixel(pc, ns, args, pc_args, flow, grey_index, color, candidate_color)
                else:
                    sample_color = sample_pixel(pc, ns, args, pc_args, flow, grey_index, color, candidate_color)

                color_tensor[c, y, z] = sample_color
                valid_ycocg = valid_ycocg[:, torch.abs(valid_ycocg[c, :] - color_tensor[c, y, z]) < 10**(-2)]

                progress_bar.update(1)

    return color_tensor