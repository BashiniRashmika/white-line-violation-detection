"""
Training script for lane segmentation model using GPU
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

def train_lane_model(
    model_path: str = "yolov8n-seg.pt",
    data_yaml: str = "data/jpj_dataset/data.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 4,  # Reduced default for 4GB GPU
    device: str = "cuda",
    project: str = "runs/segment",
    name: str = "train"
):
    """
    Train YOLOv8 segmentation model on lane detection dataset using GPU.
    
    Args:
        model_path: Path to pretrained model weights
        data_yaml: Path to dataset YAML file
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to use ('cuda', 'cpu', or '0' for specific GPU)
        project: Project directory name
        name: Experiment name
    """
    # Check GPU availability
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU")
        device = "cpu"
    elif device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Warn if batch size might be too large for GPU memory
        if gpu_memory <= 4 and batch > 8:
            print(f"Warning: Batch size {batch} might be too large for {gpu_memory:.2f}GB GPU")
            print("Recommendation: Use batch size 4-8 for 4GB GPU")
        
        # Set memory allocation config
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Validate paths
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")
    
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    model = YOLO(model_path)
    
    # Train the model with memory optimizations
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=50,  # Early stopping patience
        save=True,  # Save checkpoints
        save_period=10,  # Save checkpoint every 10 epochs
        val=True,  # Validate during training
        plots=True,  # Generate plots
        verbose=True,
        amp=True,  # Automatic Mixed Precision - reduces memory usage
        cache=False,  # Don't cache images in RAM (saves memory)
        workers=4  # Reduce dataloader workers for less memory usage
    )
    
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved to: {model.trainer.best}")
    print(f"Last model saved to: {model.trainer.last}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 segmentation model for lane detection on GPU',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n-seg.pt',
        help='Path to pretrained model (default: yolov8n-seg.pt)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/jpj_dataset/data.yaml',
        help='Path to data YAML file (default: data/jpj_dataset/data.yaml)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training (default: 640)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=4,
        help='Batch size (default: 4 for 4GB GPU, use 8-16 for larger GPUs)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', '0', '1'],
        help='Device to use: cuda, cpu, or GPU ID (default: cuda)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/segment',
        help='Project directory (default: runs/segment)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='train',
        help='Experiment name (default: train)'
    )
    
    args = parser.parse_args()
    
    try:
        train_lane_model(
            model_path=args.model,
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name
        )
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

