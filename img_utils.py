import os
import random
from PIL import Image
import numpy as np
import torch


def Patchify(x, patch_size = 16):
    patch_size = (patch_size, patch_size)
    H, W = x.size(1), x.size(2)

    num_patches_h = H // patch_size[0]
    num_patches_w = W // patch_size[1]

    patches = []

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            Hs, Ws = i * patch_size[0], j * patch_size[1]
            He, We = Hs + patch_size[0], Ws + patch_size[1]
            patches.append(x[:, Hs:He, Ws:We])

    patches = torch.stack(patches)

    return patches

def Patchify_random(x, patch_size = 16):
    patch_size = (patch_size, patch_size)
    H, W = x.size(1), x.size(2)

    Hs, Ws = random.randint(0, H - patch_size[0]), random.randint(0, W - patch_size[1])
    He, We = Hs + patch_size[0], Ws + patch_size[1]
    
    return x[:, Hs:He, Ws:We]

def Depatchify(patches, original_shape, patch_size=16):
    patch_size = (patch_size, patch_size)
    C, H, W = original_shape

    num_patches_h = H // patch_size[0]
    num_patches_w = W // patch_size[1]

    image = torch.zeros(C, H, W)

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            Hs, Ws = i * patch_size[0], j * patch_size[1]
            He, We = Hs + patch_size[0], Ws + patch_size[1]
            patch_index = i * num_patches_w + j
            image[:, Hs:He, Ws:We] = patches[patch_index]
    
    return image

def RGB2YCoCg(x, rgb_range = (-1, 1)):
    R, G, B = x[0,:,:], x[1,:,:], x[2,:,:]

    R = (R - rgb_range[0]) / (rgb_range[1] - rgb_range[0])
    G = (G - rgb_range[0]) / (rgb_range[1] - rgb_range[0])
    B = (B - rgb_range[0]) / (rgb_range[1] - rgb_range[0])

    Co  = R - B
    tmp = B + Co/2
    Cg  = G - tmp
    Y   = tmp + Cg/2

    # Make the range of Y to be [-1, 1]
    Y = Y * 2 - 1

    return torch.stack((Y, Co, Cg), dim = 0) # [-1, 1]

def YCoCg2RGB(ycocg, rgb_range=(-1, 1)):
    Y, Co, Cg = ycocg[0, :, :], ycocg[1, :, :], ycocg[2, :, :] # [-1, 1]

    Y = (Y + 1) / 2

    tmp = Y - Cg / 2
    G = Cg + tmp
    B = tmp - Co / 2
    R = B + Co

    R = R * (rgb_range[1] - rgb_range[0]) + rgb_range[0]  # Scale back to original rgb_range
    G = G * (rgb_range[1] - rgb_range[0]) + rgb_range[0]  # Scale back to original rgb_range
    B = B * (rgb_range[1] - rgb_range[0]) + rgb_range[0]  # Scale back to original rgb_range

    return torch.stack((R, G, B), dim=0) # [-1, 1]

def Quantize(x, num_levels = 256, input_range = (-1, 1)):
    x = (x - input_range[0]) / (input_range[1] - input_range[0])

    return torch.floor(x * num_levels).long().clip(0, num_levels - 1)

def Dequantize(ycocg, num_levels=256):
    Y, Co, Cg = ycocg[0, :, :], ycocg[1, :, :], ycocg[2, :, :]

    Y = Y.float() / (num_levels - 1)  # Scale Y back to [0, 1]
    Co = Co.float() / (num_levels - 1)  # Scale Co back to [0, 1]
    Cg = Cg.float() / (num_levels - 1)  # Scale Cg back to [0, 1]

    Y = Y * 2 - 1  # Normalize Y to [-1, 1]
    Co = Co * 2 - 1  # Normalize Co to [-1, 1]
    Cg = Cg * 2 - 1  # Normalize Cg to [-1, 1]

    return torch.stack((Y, Co, Cg), dim=0)

def Flatten(x):
    return x.reshape(-1)

def YCoCg2Grey(x, num_levels = 256):
    Y, Co, Cg = x[0,:,:], x[1,:,:], x[2,:,:]

    return torch.round(Y).long().clip(0, num_levels - 1)

def save_img(directory, file_name, img):
    img = img.numpy()
    img = img.astype(np.uint8)

    if file_name != "Grey_Image.png":
        img = np.transpose(img, (1, 2, 0))
    
    image = Image.fromarray(img)
    image.save(f"{directory}/{file_name}")


def save_imgs(directory, original_img, pred_YCoCg_img, pred_img):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Original_Image
    save_img(directory, "Original_Image.png", Quantize(original_img))

    # YCoCg_Image
    YCoCg_img = RGB2YCoCg(original_img)
    YCoCg_img = Quantize(YCoCg_img)
    save_img(directory, "YCoCg_Image.png", YCoCg_img)

    # Pred_YCoCg_Image
    save_img(directory, "Pred_YCoCg_Image.png", pred_YCoCg_img)

    # Pred_Image
    save_img(directory, "Pred_Image.png", pred_img)

    # Grey_Image
    grey = YCoCg2Grey(Quantize(RGB2YCoCg(original_img)))
    save_img(directory, "Grey_Image.png", grey)