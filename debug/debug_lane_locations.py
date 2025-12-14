"""
Debug script to visualize lane detection locations vs vehicle locations
"""
import cv2
import numpy as np
from white_line_violation import WhiteLineViolationDetector
from pathlib import Path

def debug_lane_vehicle_overlap(image_path, model_path):
    """Visualize where lanes are detected vs where vehicles are."""
    
    # Initialize detector
    detector = WhiteLineViolationDetector(
        lane_model_path=model_path,
        vehicle_model_path='yolov8n.pt',
        conf_threshold=0.15  # Lower confidence to catch more lanes
    )
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Analyzing: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Detect lanes with lower confidence
    print("\nDetecting lanes with confidence 0.15...")
    lane_results = detector.detect_lanes(image, filter_classes=None)  # Get all lanes first
    lane_mask_all = lane_results['masks']
    
    # Also check with even lower confidence
    results_raw = detector.lane_model(image, conf=0.1, verbose=False)
    result_raw = results_raw[0]
    
    print(f"\nRaw model detections (conf=0.1):")
    if result_raw.boxes is not None:
        print(f"  Total detections: {len(result_raw.boxes)}")
        classes = result_raw.boxes.cls.cpu().numpy().astype(int)
        confidences = result_raw.boxes.conf.cpu().numpy()
        
        for cls_id, conf in zip(classes, confidences):
            class_name = result_raw.names.get(cls_id, f"class_{cls_id}")
            print(f"    - {class_name} (class {cls_id}): confidence {conf:.2f}")
    
    # Detect vehicles
    vehicle_results = detector.detect_vehicles(image)
    vehicle_boxes = vehicle_results['boxes']
    
    print(f"\nVehicles detected: {len(vehicle_boxes)}")
    
    # Create visualization
    vis_image = image.copy()
    
    # Draw all lane pixels in cyan (brighter)
    if np.any(lane_mask_all > 0):
        lane_overlay = np.zeros_like(vis_image)
        lane_overlay[lane_mask_all > 0] = [255, 255, 0]  # Cyan
        vis_image = cv2.addWeighted(vis_image, 0.6, lane_overlay, 0.4, 0)
    
    # Draw vehicle boxes in green
    for i, box in enumerate(vehicle_boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw bottom edge of vehicle in yellow (critical area)
        bottom_y = int(y1 + (y2 - y1) * 0.7)
        cv2.rectangle(vis_image, (x1, bottom_y), (x2, y2), (0, 255, 255), 1)
    
    # Check for violations with very low threshold
    violations_found = 0
    for i, box in enumerate(vehicle_boxes):
        x1, y1, x2, y2 = box.astype(int)
        
        # Check bottom edge specifically
        bottom_y = int(y1 + (y2 - y1) * 0.7)
        bottom_region = lane_mask_all[bottom_y:y2, x1:x2]
        bottom_area = (y2 - bottom_y) * (x2 - x1)
        
        if bottom_area > 0:
            overlap_pixels = np.sum(bottom_region > 0)
            overlap_ratio = overlap_pixels / bottom_area
            
            if overlap_ratio > 0.001:  # Even very small overlap
                print(f"\nVehicle {i+1} bottom edge overlap: {overlap_ratio:.4f}")
                
                # Draw red box if any overlap
                if overlap_ratio > 0.005:
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    violations_found += 1
                    print(f"  → Potential violation (overlap: {overlap_ratio:.4f})")
    
    # Save debug visualization
    output_path = 'outputs/images/debug_lane_locations.jpg'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, vis_image)
    
    print(f"\n{'='*60}")
    print(f"Lane mask stats:")
    print(f"  Total lane pixels: {np.sum(lane_mask_all > 0)}")
    print(f"  Lane coverage: {np.sum(lane_mask_all > 0) / (lane_mask_all.shape[0] * lane_mask_all.shape[1]) * 100:.2f}%")
    print(f"  Detected classes: {lane_results['class_ids']}")
    print(f"\nPotential violations (overlap > 0.5%): {violations_found}")
    print(f"\nDebug visualization saved to: {output_path}")
    print(f"\nCheck the image:")
    print(f"  - Cyan/yellow = detected lanes")
    print(f"  - Green boxes = vehicles")
    print(f"  - Yellow rectangles = bottom 30% (critical area)")
    print(f"  - Red boxes = potential violations")
    print(f"{'='*60}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python debug_lane_locations.py <image_path> [model_path]")
        print("Example: python debug_lane_locations.py test/test3.png runs/segment/train3/weights/best.pt")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'runs/segment/train3/weights/best.pt'
    
    debug_lane_vehicle_overlap(image_path, model_path)

