"""
White-Line Violation Detection System using YOLOv8

This module provides a complete solution for detecting vehicles crossing white lines
using YOLOv8 segmentation for lane detection and YOLOv8 object detection for vehicles.

Author: AI Assistant
Date: 2025
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class WhiteLineViolationDetector:
    """
    Main class for detecting white-line violations.
    
    Combines lane segmentation and vehicle detection to identify vehicles
    crossing solid or double white lines.
    """
    
    # Class IDs for violation-relevant lane types
    SOLID_LINE_CLASS = 5  # solid-line
    DOUBLE_LINE_CLASS = 2  # double-line
    VIOLATION_LANE_CLASSES = [SOLID_LINE_CLASS, DOUBLE_LINE_CLASS]
    
    def __init__(
        self,
        lane_model_path: str,
        vehicle_model_path: str = "yolov8n.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize the White-Line Violation Detector.
        
        Args:
            lane_model_path: Path to trained YOLOv8 segmentation model for lanes
            vehicle_model_path: Path to YOLOv8 detection model for vehicles (default: pretrained)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Load models
        logger.info(f"Loading lane segmentation model from {lane_model_path}")
        self.lane_model = YOLO(lane_model_path)
        self.lane_model.to(self.device)
        
        logger.info(f"Loading vehicle detection model from {vehicle_model_path}")
        self.vehicle_model = YOLO(vehicle_model_path)
        self.vehicle_model.to(self.device)
        
        logger.info("Models loaded successfully")
    
    def detect_lanes(
        self,
        image: np.ndarray,
        filter_classes: Optional[List[int]] = None,
        conf_override: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Detect and segment lanes in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            filter_classes: List of class IDs to filter (None = all classes)
                           Default: only SOLID_LINE_CLASS and DOUBLE_LINE_CLASS
            conf_override: Override confidence threshold for lane detection
        
        Returns:
            Dictionary containing:
                - 'masks': Combined binary mask of violation lanes
                - 'masks_by_class': Dictionary mapping class_id to masks
                - 'class_ids': List of detected class IDs
        """
        if filter_classes is None:
            filter_classes = self.VIOLATION_LANE_CLASSES
        
        # Use override confidence or default
        conf = conf_override if conf_override is not None else self.conf_threshold
        
        # Run segmentation
        results = self.lane_model(
            image,
            conf=conf,
            iou=self.iou_threshold,
            verbose=False
        )
        
        result = results[0]
        h, w = image.shape[:2]
        
        # Initialize output
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        masks_by_class = {}
        detected_classes = []
        
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # Shape: (n, h, w)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for mask, cls_id in zip(masks, classes):
                # Resize mask to image dimensions if needed
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Filter by class if specified
                if cls_id in filter_classes:
                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
                    combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
                    
                    if cls_id not in masks_by_class:
                        masks_by_class[cls_id] = np.zeros((h, w), dtype=np.uint8)
                    masks_by_class[cls_id] = cv2.bitwise_or(
                        masks_by_class[cls_id],
                        mask_binary
                    )
                    
                    if cls_id not in detected_classes:
                        detected_classes.append(cls_id)
        
        return {
            'masks': combined_mask,
            'masks_by_class': masks_by_class,
            'class_ids': detected_classes
        }
    
    def detect_vehicles(
        self,
        image: np.ndarray,
        vehicle_classes: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Detect vehicles in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            vehicle_classes: List of COCO class IDs to detect as vehicles
                            Default: [2, 3, 5, 7] (car, motorcycle, bus, truck)
        
        Returns:
            Dictionary containing:
                - 'boxes': Bounding boxes as numpy array (n, 4) [x1, y1, x2, y2]
                - 'scores': Confidence scores (n,)
                - 'class_ids': Class IDs (n,)
                - 'class_names': Class names (n,)
        """
        if vehicle_classes is None:
            # COCO class IDs for vehicles
            vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Run detection
        results = self.vehicle_model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=vehicle_classes,
            verbose=False
        )
        
        result = results[0]
        boxes_list = []
        scores_list = []
        class_ids_list = []
        class_names_list = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_array = result.boxes.xyxy.cpu().numpy()  # (n, 4)
            scores_array = result.boxes.conf.cpu().numpy()  # (n,)
            cls_ids_array = result.boxes.cls.cpu().numpy().astype(int)  # (n,)
            
            for box, score, cls_id in zip(boxes_array, scores_array, cls_ids_array):
                boxes_list.append(box)
                scores_list.append(score)
                class_ids_list.append(cls_id)
                class_names_list.append(result.names[cls_id])
        
        return {
            'boxes': np.array(boxes_list) if boxes_list else np.empty((0, 4)),
            'scores': np.array(scores_list) if scores_list else np.empty((0,)),
            'class_ids': np.array(class_ids_list) if class_ids_list else np.empty((0,), dtype=int),
            'class_names': class_names_list
        }
    
    def check_violation(
        self,
        vehicle_box: np.ndarray,
        lane_mask: np.ndarray,
        vehicle_box_overlap_threshold: float = 0.3,
        bottom_edge_overlap_threshold: float = 0.05,
        use_bottom_edge: bool = True,
        use_edge_proximity: bool = True,
        proximity_threshold: int = 10
    ) -> Tuple[bool, float]:
        """
        Check if a vehicle bounding box overlaps with lane mask (violation).
        
        Uses two methods:
        1. Full bounding box overlap check (default threshold: 0.3 = 30%)
        2. Bottom edge overlap check (default threshold: 0.05 = 5%) - more sensitive
           for subtle violations where only tires cross the line
        
        Args:
            vehicle_box: Bounding box [x1, y1, x2, y2]
            lane_mask: Binary mask of lanes (H, W)
            vehicle_box_overlap_threshold: Minimum overlap ratio for entire vehicle box (0-1)
            bottom_edge_overlap_threshold: Minimum overlap ratio for bottom 30% of vehicle (0-1)
                                          Lower threshold for detecting subtle tire crossings
            use_bottom_edge: If True, check bottom edge (where wheels touch ground)
        
        Returns:
            Tuple of (is_violation, overlap_ratio)
        """
        x1, y1, x2, y2 = vehicle_box.astype(int)
        h, w = lane_mask.shape
        
        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # Extract vehicle region from mask
        vehicle_region = lane_mask[y1:y2, x1:x2]
        vehicle_area = (x2 - x1) * (y2 - y1)
        
        if vehicle_area == 0:
            return False, 0.0
        
        # --- Method 1: Full Bounding Box Check ---
        overlapping_pixels = np.sum(vehicle_region > 0)
        overlap_ratio = overlapping_pixels / vehicle_area
        is_violation = overlap_ratio >= vehicle_box_overlap_threshold
        
        # --- Method 2: Bottom Edge Check (Critical Area Check) ---
        if use_bottom_edge:
            # Check bottom 30% of vehicle (where wheels are)
            bottom_y = int(y1 + (y2 - y1) * 0.7)  # Bottom 30%
            bottom_region = lane_mask[bottom_y:y2, x1:x2]
            bottom_area = (y2 - bottom_y) * (x2 - x1)
            
            if bottom_area > 0:
                bottom_overlapping = np.sum(bottom_region > 0)
                bottom_overlap_ratio = bottom_overlapping / bottom_area
                
                # Check for violation using the dedicated low threshold for bottom edge
                if bottom_overlap_ratio >= bottom_edge_overlap_threshold:
                    # If violation detected in the critical area, override full box check
                    is_violation = True
                    # Update overlap_ratio to the highest one for logging/visualization
                    overlap_ratio = max(overlap_ratio, bottom_overlap_ratio)
        
        # --- Method 3: Edge Proximity Check (if overlap is too low) ---
        # This checks if lane pixels are near vehicle edges (even if not overlapping)
        if use_edge_proximity and not is_violation and np.any(lane_mask > 0):
            # Check bottom edge region with proximity margin
            bottom_y = int(y1 + (y2 - y1) * 0.7)
            
            # Expand bottom edge check with margin
            search_margin = proximity_threshold
            search_bottom_y = max(0, bottom_y - search_margin)
            search_bottom_y2 = min(h, y2 + search_margin)
            search_left_x = max(0, x1 - search_margin)
            search_right_x = min(w, x2 + search_margin)
            
            # Check if ANY lane pixels are in the expanded bottom edge region
            bottom_region_expanded = lane_mask[search_bottom_y:search_bottom_y2, search_left_x:search_right_x]
            bottom_lane_pixels = np.sum(bottom_region_expanded > 0)
            
            # Also check if lanes intersect the bottom edge of vehicle
            bottom_edge_region = lane_mask[bottom_y:y2, x1:x2]
            bottom_edge_pixels = np.sum(bottom_edge_region > 0)
            
            # If ANY lane pixels are near bottom edge (where wheels are), it's a violation
            if bottom_edge_pixels > 0 or bottom_lane_pixels > 0:
                is_violation = True
                # Calculate overlap based on bottom edge
                bottom_area = (y2 - bottom_y) * (x2 - x1)
                if bottom_area > 0:
                    edge_overlap = bottom_edge_pixels / bottom_area
                    overlap_ratio = max(overlap_ratio, edge_overlap)
                else:
                    # Even if area is small, presence of lane pixels means violation
                    overlap_ratio = max(overlap_ratio, 0.01)
        
        return is_violation, overlap_ratio
    
    def process_image(
        self,
        image: np.ndarray,
        visualize: bool = True,
        overlap_threshold: float = 0.3,
        bottom_overlap_threshold: float = 0.05,
        lane_conf: Optional[float] = None
    ) -> Dict:
        """
        Process a single image to detect white-line violations.
        
        Args:
            image: Input image as numpy array (BGR format)
            visualize: Whether to draw visualizations on the image
            overlap_threshold: Minimum overlap ratio for entire vehicle box (default: 0.3)
            bottom_overlap_threshold: Minimum overlap ratio for bottom edge (default: 0.05)
                                     Lower threshold for detecting subtle tire crossings
        
        Returns:
            Dictionary containing:
                - 'image': Processed image (with or without visualization)
                - 'violations': List of violation detections
                - 'stats': Statistics about detections
        """
        # Detect lanes - use lane_conf if provided, otherwise use default
        lane_results = self.detect_lanes(image, conf_override=lane_conf)
        lane_mask = lane_results['masks']
        
        # If no lanes detected and lane_conf not specified, try with lower confidence
        if np.sum(lane_mask > 0) == 0 and lane_conf is None:
            logger.warning("No lanes detected with default confidence, trying lower confidence (0.15)...")
            lane_results = self.detect_lanes(image, conf_override=0.15)
            lane_mask = lane_results['masks']
        
        # Dilate lane mask to make it wider - helps with overlap detection for thin lanes
        if np.any(lane_mask > 0):
            kernel_size = 5  # Make lanes 5 pixels wider on each side
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            lane_mask = cv2.dilate(lane_mask, kernel, iterations=1)
        
        # Detect vehicles
        vehicle_results = self.detect_vehicles(image)
        vehicle_boxes = vehicle_results['boxes']
        vehicle_scores = vehicle_results['scores']
        vehicle_classes = vehicle_results['class_names']
        
        # Check violations
        violations = []
        stats = {
            'total_vehicles': len(vehicle_boxes),
            'violations': 0,
            'lanes_detected': len(lane_results['class_ids'])
        }
        
        output_image = image.copy() if visualize else image
        
        for i, (box, score, cls_name) in enumerate(
            zip(vehicle_boxes, vehicle_scores, vehicle_classes)
        ):
            is_violation, overlap_ratio = self.check_violation(
                box,
                lane_mask,
                vehicle_box_overlap_threshold=overlap_threshold,
                bottom_edge_overlap_threshold=bottom_overlap_threshold,
                use_bottom_edge=True,
                use_edge_proximity=True,
                proximity_threshold=15  # Check 15 pixels around vehicle
            )
            
            if is_violation:
                violations.append({
                    'vehicle_id': i,
                    'box': box,
                    'score': score,
                    'class': cls_name,
                    'overlap_ratio': overlap_ratio
                })
                stats['violations'] += 1
            
            # Visualize
            if visualize:
                color = (0, 0, 255) if is_violation else (0, 255, 0)
                thickness = 3 if is_violation else 2
                
                # Draw vehicle box
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
                
                # Add label
                label = f"{cls_name}: {score:.2f}"
                if is_violation:
                    label += f" VIOLATION ({overlap_ratio:.2f})"
                
                # Draw label background
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    output_image,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w, y1),
                    color,
                    -1
                )
                cv2.putText(
                    output_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        # Overlay lane masks
        if visualize and np.any(lane_mask > 0):
            # Create colored overlay for lanes
            lane_overlay = np.zeros_like(output_image)
            lane_overlay[lane_mask > 0] = [255, 0, 255]  # Magenta for lanes
            output_image = cv2.addWeighted(output_image, 0.7, lane_overlay, 0.3, 0)
        
        # Add statistics text
        if visualize:
            stats_text = [
                f"Vehicles: {stats['total_vehicles']}",
                f"Violations: {stats['violations']}",
                f"Lanes: {stats['lanes_detected']}"
            ]
            y_offset = 30
            for i, text in enumerate(stats_text):
                cv2.putText(
                    output_image,
                    text,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    output_image,
                    text,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    1
                )
        
        return {
            'image': output_image,
            'violations': violations,
            'stats': stats,
            'lane_mask': lane_mask,
            'vehicle_results': vehicle_results
        }
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        overlap_threshold: float = 0.3,
        bottom_overlap_threshold: float = 0.05,
        fps: Optional[float] = None
    ) -> Dict:
        """
        Process a video file to detect white-line violations.
        
        Args:
            input_path: Path to input video file
            output_path: Path to save output video
            overlap_threshold: Minimum overlap ratio for violation detection
            fps: Output video FPS (None = same as input)
        
        Returns:
            Dictionary with processing statistics
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"Processing video: {input_path}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")
        
        frame_count = 0
        total_violations = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(f"Processing frame {frame_count}/{total_frames}")
                
                # Process frame
                result = self.process_image(
                    frame,
                    visualize=True,
                    overlap_threshold=overlap_threshold,
                    bottom_overlap_threshold=bottom_overlap_threshold
                )
                
                # Write frame
                out.write(result['image'])
                total_violations += result['stats']['violations']
        
        finally:
            cap.release()
            out.release()
            logger.info(f"Video processing complete. Saved to: {output_path}")
        
        return {
            'total_frames': frame_count,
            'total_violations': total_violations,
            'output_path': output_path
        }


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description='White-Line Violation Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python white_line_violation.py --input test.jpg --output result.jpg
  
  # Process a video
  python white_line_violation.py --input video.mp4 --output output.mp4 --video
  
  # Use custom models and thresholds
  python white_line_violation.py --input test.jpg --lane-model best.pt --conf 0.5 --overlap 0.4
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input image or video file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output.jpg',
        help='Path to output file (default: output.jpg)'
    )
    parser.add_argument(
        '--lane-model',
        type=str,
        default='yolov8n-seg.pt',
        help='Path to trained lane segmentation model (default: yolov8n-seg.pt)'
    )
    parser.add_argument(
        '--vehicle-model',
        type=str,
        default='yolov8n.pt',
        help='Path to vehicle detection model (default: yolov8n.pt)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections (default: 0.25). Lower values detect more but may have false positives.'
    )
    parser.add_argument(
        '--lane-conf',
        type=float,
        default=None,
        help='Separate confidence threshold for lane detection (default: same as --conf). Use lower values (0.1-0.15) if lanes not detected.'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.3,
        help='Overlap threshold for entire vehicle box (default: 0.3)'
    )
    parser.add_argument(
        '--bottom-overlap',
        type=float,
        default=0.05,
        help='Overlap threshold for bottom edge (where wheels are) - more sensitive (default: 0.05)'
    )
    parser.add_argument(
        '--video',
        action='store_true',
        help='Process as video file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to use for inference (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Initialize detector
    try:
        detector = WhiteLineViolationDetector(
            lane_model_path=args.lane_model,
            vehicle_model_path=args.vehicle_model,
            device=args.device,
            conf_threshold=args.conf,
            iou_threshold=0.45
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1
    
    # Process input
    try:
        if args.video or args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Process video
            stats = detector.process_video(
                input_path=args.input,
                output_path=args.output,
                overlap_threshold=args.overlap,
                bottom_overlap_threshold=args.bottom_overlap
            )
            logger.info(f"Video processing complete!")
            logger.info(f"Total frames: {stats['total_frames']}")
            logger.info(f"Total violations: {stats['total_violations']}")
            logger.info(f"Output saved to: {stats['output_path']}")
        else:
            # Process image
            image = cv2.imread(args.input)
            if image is None:
                logger.error(f"Could not read image: {args.input}")
                return 1
            
            result = detector.process_image(
                image,
                visualize=True,
                overlap_threshold=args.overlap,
                bottom_overlap_threshold=args.bottom_overlap,
                lane_conf=args.lane_conf
            )
            
            # Save output
            cv2.imwrite(args.output, result['image'])
            logger.info(f"Image processed successfully!")
            logger.info(f"Vehicles detected: {result['stats']['total_vehicles']}")
            logger.info(f"Violations detected: {result['stats']['violations']}")
            logger.info(f"Output saved to: {args.output}")
            
            # Print violation details
            if result['violations']:
                logger.info("\nViolations detected:")
                for i, violation in enumerate(result['violations'], 1):
                    logger.info(
                        f"  {i}. {violation['class']} "
                        f"(confidence: {violation['score']:.2f}, "
                        f"overlap: {violation['overlap_ratio']:.2f})"
                    )
    
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

