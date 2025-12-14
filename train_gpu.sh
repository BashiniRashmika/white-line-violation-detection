#!/bin/bash
# Bash script to train lane segmentation model on GPU
# Usage: ./train_gpu.sh [epochs] [batch_size]

EPOCHS=${1:-100}
BATCH=${2:-4}

echo "========================================"
echo "Training Lane Segmentation Model on GPU"
echo "========================================"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH"
echo "Device: CUDA"
echo "========================================"
echo

python train_lane_model_gpu.py \
    --model yolov8n-seg.pt \
    --data data/jpj_dataset/data.yaml \
    --epochs $EPOCHS \
    --imgsz 640 \
    --batch $BATCH \
    --device cuda

