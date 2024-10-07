import time
import math
import random
import argparse
import numpy as np
import torch
import pyjuice as juice

from img_utils import *
from utils import *
from algorithm import *

def get_args():
    parser = argparse.ArgumentParser(description="Script for model inference")

    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--data_shape', type=int, nargs=3, default=(3, 16, 16), help="Shape of the data")
    parser.add_argument('--num_latents', type=int, default=64, help="Number of latent variables")
    parser.add_argument('--split_intervals', type=int, nargs=3, default=(3, 2, 2), help="Split intervals")
    parser.add_argument('--max_prod_block_conns', type=int, default=8, help="Max product block connections")
    parser.add_argument('--load_model', type=bool, default=True, help="Whether to load a model checkpoint")
    parser.add_argument('--model_ckpt', type=str, default="model.jpc", help="Model checkpoint file")
    parser.add_argument('--save_img', type=bool, default=True, help="Whether to save images")
    parser.add_argument('--save_dir', type=str, default="results", help="Save figure directory")
    parser.add_argument('--random_patch', type=bool, default=False, help="Whether to use random patches")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of samples to generate")
    parser.add_argument('--cache_dir', type=str, default="cache", help="Cache directory")
    parser.add_argument('--compute_color_ll', type=bool, default=True, help="Whether to compute color likelihood")
    parser.add_argument('--argmax', type=bool, default=True, help="Whether to take argmax pixel")

    args = parser.parse_args()

    args.data_shape = tuple(args.data_shape)
    args.split_intervals = tuple(args.split_intervals)

    return args

def color_img(pc, ns, img, img_index, args, pc_args):
    img = np.array(img)
    img = img.reshape(3, 32, 32)
    img = torch.tensor(img)
    original_shape = (img.shape[0], img.shape[1], img.shape[2])

    img = img.float() / 127.5 - 1
    patched_img = Patchify(img)

    colored_patches = []
    for i in range(patched_img.shape[0]):
        patch = patched_img[i, :, :, :]
        pred_patch = color_patch(patch, pc, ns, args, pc_args)

        colored_patches.append(pred_patch)

    colored_patches = torch.stack(colored_patches)
    colored_img = Depatchify(colored_patches, original_shape)

    pred_img = Dequantize(colored_img)
    pred_img = YCoCg2RGB(pred_img)
    pred_img = Quantize(pred_img)

    save_imgs(f"{args.save_dir}/Image{img_index}", img, colored_img, pred_img)

    if img_index == 4:
        exit()

def color_patch(patch, pc, ns, args, pc_args):
    YCoCg_patch = RGB2YCoCg(patch)
    YCoCg_patch = Quantize(YCoCg_patch)
    Grey_patch = YCoCg2Grey(YCoCg_patch)
    Grey_patch = Flatten(Grey_patch)

    if args.compute_color_ll:
        x = Flatten(YCoCg_patch)
        x = x.unsqueeze(0)
        x = x.repeat(4, 1)
        x = x.to(args.device)
        color_lls = pc(x)
        print(f"color_lls: {color_lls}")
    
    start_time = time.perf_counter()
    # grey_prob = compute_grey_prob(args, auto_index = None, color_pred = None, 
    #                                 pc_num_elements = pc.num_elements, grey_arr = Grey_patch.numpy(),
    #                                 param_arr = pc_args["param_array"], grey_idx_to_prod_se_arr = pc_args["grey_idx_to_prod_se_array"],
    #                                 arr_valid_ycocg = pc_args["arr_valid_ycocg"], arr_num_valid_ycocg = pc_args["arr_num_valid_ycocg"])

    grey_prob = compute_grey_prob_numba(args.data_shape, pc.num_elements, grey_arr = Grey_patch.numpy(),
                                    param_arr = pc_args["param_array"], grey_idx_to_prod_se_arr = pc_args["grey_idx_to_prod_se_array"],
                                    arr_valid_ycocg = pc_args["arr_valid_ycocg"], arr_num_valid_ycocg = pc_args["arr_num_valid_ycocg"])

                                    
    end_time = time.perf_counter()
    print(f"Time taken by grey_prob: {(end_time - start_time):.6f} seconds")
    '''
        3.143323
        1.538325
    '''

    pred_patch = color_patch_algo(pc, ns, args, pc_args, Grey_patch, grey_prob)

    return pred_patch


def main():
    args = get_args()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if args.load_model:
        print(f"Loading model from {args.model_ckpt}")
        ns = juice.load(args.model_ckpt)
        pc = juice.TensorCircuit(ns)
        pc.print_statistics()
        pc.to(device)

    param_array, grey_idx_to_prod_se_array, color_idx_to_input_se_array = create_circuit_info(ns, pc, args)
    arr_valid_ycocg, arr_num_valid_ycocg = create_grey_info(args)

    pc_args = {"param_array": param_array, "grey_idx_to_prod_se_array": grey_idx_to_prod_se_array, "color_idx_to_input_se_array": color_idx_to_input_se_array,
                "arr_valid_ycocg": arr_valid_ycocg, "arr_num_valid_ycocg": arr_num_valid_ycocg}

    data = np.load('/space/liruoyan/ImageNet/imagenet32/val/val_data.npz')
    images = data['data']

    for img_index, img in enumerate(images):
        color_img(pc, ns, img, img_index, args, pc_args)

main()