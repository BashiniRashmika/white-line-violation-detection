# White-Line Violation Detection System

<div align="center">

**A Computer Vision System for Detecting Vehicles Crossing White Lines**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Model Training](#model-training)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [License](#license)

---

## 🎯 Overview

This project implements a **Proof-of-Concept (PoC)** system for detecting white-line violations in traffic scenarios. The system combines:

- **YOLOv8 Segmentation** for detecting and segmenting road lanes (solid and double white lines)
- **YOLOv8 Object Detection** for detecting vehicles (cars, motorcycles, buses, trucks)
- **Spatial Analysis** to identify when vehicles cross or overlap with white lines

The system processes images and videos, providing visual feedback with bounding boxes, lane masks, and violation alerts.

---

## ✨ Features

### Core Capabilities

- ✅ **Dual Model Architecture**: Separate models for lane segmentation and vehicle detection
- ✅ **Modular Design**: Reusable functions for easy integration and customization
- ✅ **Multi-Format Support**: Process images (JPG, PNG) and videos (MP4, AVI, MOV, MKV)
- ✅ **Advanced Violation Logic**: 
  - Full bounding box overlap detection
  - Bottom-edge overlap check (focuses on wheel area)
  - Proximity-based detection for subtle violations
- ✅ **GPU Acceleration**: Supports both CPU and CUDA for faster processing
- ✅ **Visual Output**: Rich visualization with color-coded boxes, masks, and statistics
- ✅ **CLI & API**: Easy-to-use command-line interface and programmatic API

### Detection Capabilities

**Lane Types:**
- Solid white lines (Class 5)
- Double white lines (Class 2)

**Vehicle Types:**
- Cars
- Motorcycles
- Buses
- Trucks

---

## 💻 System Requirements

### Minimum Requirements

- **OS**: Windows 10+, Linux, or macOS
- **Python**: 3.10 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space

### Optional (Recommended)

- **GPU**: NVIDIA GPU with CUDA support (for faster processing)
- **CUDA**: Version 11.8 or higher
- **cuDNN**: Compatible version for your CUDA installation

---

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd white-line-violation-poc
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv white-line
.\white-line\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv white-line
source white-line/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Models (Optional)

The system will automatically download pretrained models on first use. Alternatively, you can manually download:

```bash
# Pretrained models will be auto-downloaded by Ultralytics
# Custom trained model should be placed at:
# runs/segment/train3/weights/best.pt
```

---

## 🚀 Quick Start

### Process a Single Image

```bash
python white_line_violation.py \
    --input test/test5.png \
    --output outputs/images/result.jpg \
    --lane-model runs/segment/train3/weights/best.pt
```

### Process a Video

```bash
python white_line_violation.py \
    --input video.mp4 \
    --output outputs/videos/result.mp4 \
    --video \
    --lane-model runs/segment/train3/weights/best.pt
```

---

## 📖 Usage

### Command Line Interface

#### Basic Usage

**Process Image:**
```bash
python white_line_violation.py -i input.jpg -o output.jpg
```

**Process Video:**
```bash
python white_line_violation.py -i video.mp4 -o output.mp4 --video
```

#### Advanced Options

```bash
python white_line_violation.py \
    --input test.jpg \
    --output result.jpg \
    --lane-model runs/segment/train3/weights/best.pt \
    --vehicle-model yolov8n.pt \
    --conf 0.5 \
    --lane-conf 0.15 \
    --overlap 0.3 \
    --bottom-overlap 0.05 \
    --device cuda
```

#### Command-Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--input` | `-i` | Path to input image or video | **Required** |
| `--output` | `-o` | Path to output file | `output.jpg` |
| `--lane-model` | | Path to lane segmentation model | `yolov8n-seg.pt` |
| `--vehicle-model` | | Path to vehicle detection model | `yolov8n.pt` |
| `--conf` | | General confidence threshold | `0.25` |
| `--lane-conf` | | Lane detection confidence threshold | Same as `--conf` |
| `--overlap` | | Full box overlap threshold (0.0-1.0) | `0.3` |
| `--bottom-overlap` | | Bottom edge overlap threshold (0.0-1.0) | `0.05` |
| `--video` | | Process as video file | `False` |
| `--device` | | Device: `cpu` or `cuda` | Auto-detect |

### Python API

#### Basic Example

```python
from white_line_violation import WhiteLineViolationDetector
import cv2

# Initialize detector
detector = WhiteLineViolationDetector(
    lane_model_path='runs/segment/train3/weights/best.pt',
    vehicle_model_path='yolov8n.pt',
    conf_threshold=0.25
)

# Load and process image
image = cv2.imread('test.jpg')
result = detector.process_image(
    image,
    visualize=True,
    overlap_threshold=0.3,
    bottom_overlap_threshold=0.05
)

# Save output
cv2.imwrite('output.jpg', result['image'])

# Print statistics
print(f"Vehicles detected: {result['stats']['total_vehicles']}")
print(f"Violations: {result['stats']['violations']}")
```

#### Modular Usage

```python
from white_line_violation import WhiteLineViolationDetector
import cv2

detector = WhiteLineViolationDetector()

image = cv2.imread('test.jpg')

# Step 1: Detect lanes
lane_results = detector.detect_lanes(image, filter_classes=[2, 5])
print(f"Lanes detected: {len(lane_results['class_ids'])}")

# Step 2: Detect vehicles
vehicle_results = detector.detect_vehicles(image)
print(f"Vehicles detected: {len(vehicle_results['boxes'])}")

# Step 3: Check violations
lane_mask = lane_results['masks']
for box in vehicle_results['boxes']:
    is_violation, overlap = detector.check_violation(
        vehicle_box=box,
        lane_mask=lane_mask,
        vehicle_box_overlap_threshold=0.3,
        bottom_edge_overlap_threshold=0.05
    )
    print(f"Violation: {is_violation}, Overlap: {overlap:.2f}")
```

#### Complete Example

See `example_usage.py` for comprehensive examples:

```bash
python example_usage.py
```

---

## 📁 Project Structure

```
white-line-violation-poc/
├── white_line_violation.py          # Main detection system
├── train_lane_model_gpu.py          # GPU training script
├── example_usage.py                  # Usage examples
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── .gitignore                        # Git ignore rules
│
├── data/
│   └── jpj_dataset/                  # Training dataset
│       ├── data.yaml                 # Dataset configuration
│       ├── train2017/                # Training images & labels
│       └── val2017/                  # Validation images & labels
│
├── scripts/
│   └── convert_coco_to_yolo.py      # COCO to YOLO converter
│
├── debug/                            # Debug and analysis tools
│   ├── analyze_detection.py
│   ├── debug_lane_detection.py
│   ├── debug_lane_locations.py
│   └── diagnose_violations.py
│
├── test/                             # Test images
│   ├── test1.jpg
│   ├── test2.jpg
│   └── ...
│
├── outputs/                          # Output directory
│   ├── images/                       # Processed images
│   └── videos/                       # Processed videos
│
├── runs/                             # Training outputs
│   └── segment/
│       └── train3/                   # Latest training run
│           └── weights/
│               ├── best.pt           # Best model (recommended)
│               └── last.pt           # Last checkpoint
│
├── yolov8n-seg.pt                    # Pretrained segmentation model
├── yolov8n.pt                        # Pretrained detection model
│
└── white-line/                       # Virtual environment (not in git)
```

---

## ⚙️ Configuration

### Lane Detection Classes

The system is configured to detect violations for specific lane types:

- **Class 2**: `double-line` (double white line)
- **Class 5**: `solid-line` (solid white line)

These can be modified in the `WhiteLineViolationDetector` class:

```python
VIOLATION_LANE_CLASSES = [2, 5]  # Modify as needed
```

### Vehicle Detection Classes

Default COCO vehicle classes:

- **Class 2**: Car
- **Class 3**: Motorcycle
- **Class 5**: Bus
- **Class 7**: Truck

### Detection Thresholds

#### Confidence Threshold (`--conf`)

- **Purpose**: Minimum confidence score for detections
- **Range**: 0.0 to 1.0
- **Default**: 0.25
- **Lower values**: More detections, more false positives
- **Higher values**: Fewer detections, higher precision

#### Lane Confidence Threshold (`--lane-conf`)

- **Purpose**: Separate threshold for lane detection
- **Default**: Same as `--conf`
- **Recommendation**: Use 0.1-0.15 if lanes are not being detected

#### Overlap Thresholds

**Full Box Overlap (`--overlap`):**
- Minimum ratio of vehicle bounding box overlapping with lane mask
- Default: 0.3 (30%)
- Higher values: Stricter detection

**Bottom Edge Overlap (`--bottom-overlap`):**
- Minimum ratio for bottom 30% of vehicle (wheel area)
- Default: 0.05 (5%)
- More sensitive to subtle violations
- Lower values catch smaller overlaps

### Device Selection

The system automatically detects available devices. To force a specific device:

```bash
--device cpu    # Force CPU
--device cuda   # Force GPU (requires CUDA)
```

---

## 🎓 Model Training

### Training Custom Lane Detection Model

To train your own lane segmentation model on the JPJ dataset:

**Using GPU:**
```bash
python train_lane_model_gpu.py \
    --epochs 100 \
    --batch 4 \
    --device cuda \
    --imgsz 640
```

**Using Windows Batch Script:**
```bash
train_gpu.bat
```

**Using Linux/Mac Shell Script:**
```bash
./train_gpu.sh
```

### Training Parameters

- **Epochs**: Number of training iterations (default: 100)
- **Batch Size**: Adjust based on GPU memory (4 for 4GB GPU, 16+ for larger GPUs)
- **Image Size**: Input resolution (640 recommended for speed/accuracy balance)
- **Device**: `cpu` or `cuda`

### Dataset Structure

The training script expects the dataset in YOLO format:

```
data/jpj_dataset/
├── data.yaml           # Dataset configuration
├── train2017/
│   ├── image1.jpg
│   ├── image1.txt      # YOLO format labels
│   └── ...
└── val2017/
    ├── image1.jpg
    ├── image1.txt
    └── ...
```

### Converting COCO to YOLO Format

If your dataset is in COCO format:

```bash
python scripts/convert_coco_to_yolo.py
```

---

## 🔧 Troubleshooting

### Common Issues

#### Model Not Found

**Error**: `FileNotFoundError: Model file not found`

**Solution**:
1. Ensure model files exist in the project root or specify correct path
2. YOLOv8 will auto-download pretrained models on first use
3. For custom models, use `--lane-model` to specify path

#### Low Detection Accuracy

**Problem**: Few or no violations detected

**Solutions**:
- Lower `--conf` threshold (try 0.15-0.20)
- Lower `--lane-conf` threshold (try 0.1-0.15)
- Lower `--overlap` threshold (try 0.2)
- Lower `--bottom-overlap` threshold (try 0.02)
- Ensure trained model is being used: `--lane-model runs/segment/train3/weights/best.pt`

#### Too Many False Positives

**Problem**: Too many violations detected incorrectly

**Solutions**:
- Increase `--conf` threshold
- Increase `--overlap` threshold
- Check if lane detection is accurate (use debug scripts)

#### CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
- Use `--device cpu` for CPU-only processing
- Reduce batch size in training (use `batch=4` instead of `batch=16`)
- Process smaller images or resize input
- Close other GPU-intensive applications

#### Lanes Not Detected

**Problem**: No lanes detected in image

**Solutions**:
- Lower `--lane-conf` to 0.1-0.15
- Verify model is trained on similar images
- Check if image quality/resolution is adequate
- Use `debug/debug_lane_detection.py` to visualize detections

### Debug Tools

Several debug scripts are available in the `debug/` directory:

```bash
# Visualize lane detections
python debug/debug_lane_detection.py image.jpg model.pt

# Analyze detection overlap
python debug/analyze_detection.py image.jpg

# Diagnose violation detection
python debug/diagnose_violations.py image.jpg model.pt
```

---

## ⚡ Performance

### Processing Speed

Approximate processing times on different hardware:

| Hardware | Image (640x640) | Video (1080p, 30fps) |
|----------|----------------|---------------------|
| CPU (Intel i5) | ~150-200ms | ~5-7 fps |
| GPU (NVIDIA GTX 1660) | ~25-40ms | ~20-25 fps |
| GPU (NVIDIA RTX 3080) | ~15-25ms | ~30-40 fps |

### Memory Usage

- **CPU Mode**: ~2-3 GB RAM
- **GPU Mode**: ~1-2 GB VRAM (4GB GPU minimum)

### Optimization Tips

1. **Use GPU**: Significant speedup (5-10x faster)
2. **Reduce Image Size**: Process at 640x640 for best speed/accuracy balance
3. **Batch Processing**: Process multiple images in a loop for better GPU utilization
4. **Disable Visualization**: Set `visualize=False` for faster processing in API mode

---

## 📄 License

This project is provided as-is for educational and research purposes. Please refer to the individual licenses of dependencies:

- **YOLOv8 (Ultralytics)**: AGPL-3.0 License
- **PyTorch**: BSD-style License
- **OpenCV**: Apache 2.0 License

---

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📝 Notes

- This is a **Proof-of-Concept** system. For production use, additional testing, validation, and optimization are recommended.
- The violation detection logic is tunable via thresholds to balance between false positives and false negatives.
- Model performance depends on training data quality and diversity.
- For best results, train on domain-specific data (similar road conditions, lighting, camera angles).

---

<div align="center">

**Made with ❤️ for Traffic Safety Research**

For questions or issues, please open an issue on the repository.

</div>
