@echo off
REM Batch script to train lane segmentation model on GPU
REM Usage: train_gpu.bat [epochs] [batch_size]

set EPOCHS=100
set BATCH=4

if not "%1"=="" set EPOCHS=%1
if not "%2"=="" set BATCH=%2

echo ========================================
echo Training Lane Segmentation Model on GPU
echo ========================================
echo Epochs: %EPOCHS%
echo Batch Size: %BATCH%
echo Device: CUDA
echo ========================================
echo.

python train_lane_model_gpu.py --model yolov8n-seg.pt --data data/jpj_dataset/data.yaml --epochs %EPOCHS% --imgsz 640 --batch %BATCH% --device cuda

pause

