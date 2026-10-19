#!/bin/bash
ROOT_FOLDER_PATH="${ROOT_FOLDER_PATH:-./data}"
OUTPUT_FOLDER_PATH="${OUTPUT_FOLDER_PATH:-./output}"
# 设置 GPU 设备
GPU_DEVICE="${GPU_DEVICE:-0}"





#--scale_lr= \

python trian-10.py \
  --mask_channels 7 \
  --dataset LNCTVSeg \
  --input_img_channels 1 \
  --image_size 96 \
  --data_path  "$ROOT_FOLDER_PATH" \
  --dim 16 \
  --epochs 100 \
  --batch_size 1 \
  --crop_sample 2 \
  --mode train \
  --learning_rate 5e-5 \
  --cache_rate 0.1 \
  --num_workers 0 \
  --gpu "$GPU_DEVICE" \
  --output_dir "$OUTPUT_FOLDER_PATH" \
  --timesteps 1000 \
  --lstm_layers 4 \
  --fft_hidden_size 4 \
  --hidden_size 8 \
  --sampling_timesteps 10
