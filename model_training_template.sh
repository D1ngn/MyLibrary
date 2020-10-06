#!/bin/bash

MODEL_TYPE="U_Net"
CHECKPOINT_PATH="./ckpt/ckpt_10000.pth"

CUDA_VISIBLE_DEVICE=0 /home/ubuntu/anaconda3/envs/pytorch_latest_p36/bin/python3 training.py \
--sample_rate=16000 \
--fft_size=512 \
--hop_size=160 \
--mel_bins=64 \
--fmin=50 \
--fmax=8000 \
--model_type=${MODEL_TYPE} \
--pretrained_checkpoint_path=${CHECKPOINT_PATH} \
