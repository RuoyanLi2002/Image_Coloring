#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python3 main.py --seed 42 --data_shape 3 16 16 --num_latents 64 --split_intervals 3 2 2 --max_prod_block_conns 8 --load_model True --model_ckpt "model.jpc" \
                --save_img True --random_patch False --num_samples 10 --cache_dir "cache" --argmax False --save_dir "Sample"
