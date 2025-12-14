"""
Comprehensive diagnostic tool to understand why violations aren't detected
"""
import cv2
import numpy as np
from white_line_violation import WhiteLineViolationDetector
from pathlib import Path

def diagnose_violations(image_path, model_path):
    """Comprehensive diagnosis of why violations aren't detected."""
    
    print("=" * 70)
    print("VIOLATION DETECTION DIAGNOSIS")
    print("=" * 70)
    
    # Initialize detector with low confidence
    detector = WhiteLineViolationDetector(
        lane_model_path=model_path,
        vehicle_model_path='yolov8n.pt',
        conf_threshold=0.15
    )
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"\nImage: {image_path}")
    print(f"Image size: {w}x{h} pixels")
    
    # ============================================================
    # STEP 1: Check Lane Detection
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: LANE DETECTION ANALYSIS")
    print("=" * 70)
    
    # Try multiple confidence levels
    for conf in [0.25, 0.15, 0.1, 0.05]:
        results = detector.lane_model(image, conf=conf, verbose=False)
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            print(f"\nConfidence {conf}: Found {len(result.boxes)} detections")
            for cls_id in set(classes):
                count = np.sum(classes == cls_id)
                print(f"  - Class {cls_id}: {count} instances")
        else:
            print(f"\nConfidence {conf}: No detections")
    
    # Get lane results with very low confidence
    lane_results = detector.detect_lanes(image, conf_override=0.1)
    lane_mask = lane_results['masks']
    
    total_pixels = h * w
    lane_pixels = np.sum(lane_mask > 0)
    coverage = (lane_pixels / total_pixels) * 100
    
    print(f"\nFinal Lane Mask Statistics:")
    print(f"  Lane pixels detected: {lane_pixels:,} ({coverage:.3f}% of image)")
    print(f"  Detected classes: {lane_results['class_ids']}")
    
    if lane_pixels == 0:
        print("\n⚠️  PROBLEM: No lane pixels detected!")
        print("   Solutions:")
        print("   1. Lower confidence threshold further (try --lane-conf 0.05)")
        print("   2. Model may need more training on similar scenes")
        print("   3. Check if lanes are visible in the image")
        return
    
    # ============================================================
    # STEP 2: Check Vehicle Detection
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: VEHICLE DETECTION ANALYSIS")
    print("=" * 70)
    
    vehicle_results = detector.detect_vehicles(image)
    vehicle_boxes = vehicle_results['boxes']
    
    print(f"Vehicles detected: {len(vehicle_boxes)}")
    for i, (box, score, cls_name) in enumerate(
        zip(vehicle_boxes, vehicle_results['scores'], vehicle_results['class_names'])
    ):
        print(f"  Vehicle {i+1}: {cls_name} (confidence: {score:.2f})")
        print(f"    Box: [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")
        print(f"    Size: {int(box[2]-box[0])}x{int(box[3]-box[1])} pixels")
    
    if len(vehicle_boxes) == 0:
        print("\n⚠️  PROBLEM: No vehicles detected!")
        return
    
    # ============================================================
    # STEP 3: Detailed Overlap Analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3: OVERLAP ANALYSIS (Why violations aren't detected)")
    print("=" * 70)
    
    max_overlap = 0
    max_overlap_vehicle = None
    
    for i, box in enumerate(vehicle_boxes):
        x1, y1, x2, y2 = box.astype(int)
        
        # Clamp to image bounds
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # Full box overlap
        vehicle_region = lane_mask[y1:y2, x1:x2]
        vehicle_area = (x2 - x1) * (y2 - y1)
        full_overlap_pixels = np.sum(vehicle_region > 0)
        full_overlap_ratio = full_overlap_pixels / vehicle_area if vehicle_area > 0 else 0
        
        # Bottom edge overlap (critical area)
        bottom_y = int(y1 + (y2 - y1) * 0.7)
        bottom_region = lane_mask[bottom_y:y2, x1:x2]
        bottom_area = (y2 - bottom_y) * (x2 - x1)
        bottom_overlap_pixels = np.sum(bottom_region > 0)
        bottom_overlap_ratio = bottom_overlap_pixels / bottom_area if bottom_area > 0 else 0
        
        print(f"\nVehicle {i+1} ({vehicle_results['class_names'][i]}):")
        print(f"  Full box overlap: {full_overlap_ratio:.6f} ({full_overlap_pixels} pixels / {vehicle_area} area)")
        print(f"  Bottom edge overlap: {bottom_overlap_ratio:.6f} ({bottom_overlap_pixels} pixels / {bottom_area} area)")
        
        # Check if lane pixels exist near vehicle
        # Expand search area around vehicle
        margin = 20
        search_y1 = max(0, y1 - margin)
        search_y2 = min(h, y2 + margin)
        search_x1 = max(0, x1 - margin)
        search_x2 = min(w, x2 + margin)
        search_region = lane_mask[search_y1:search_y2, search_x1:search_x2]
        nearby_lane_pixels = np.sum(search_region > 0)
        print(f"  Nearby lane pixels (within {margin}px margin): {nearby_lane_pixels}")
        
        if bottom_overlap_ratio > max_overlap:
            max_overlap = bottom_overlap_ratio
            max_overlap_vehicle = i
    
    # ============================================================
    # STEP 4: Recommendations
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 4: DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 70)
    
    if max_overlap < 0.001:
        print("\n❌ CRITICAL ISSUE: Vehicles have NO overlap with detected lanes")
        print(f"   Maximum overlap: {max_overlap:.6f} (0.1%)")
        print("\n   Possible causes:")
        print("   1. Lane detection is finding lanes in WRONG LOCATIONS")
        print("   2. Lane mask is too small/thin (only covers 0.29% of image)")
        print("   3. Vehicles and lanes are in different parts of image")
        print("\n   Solutions:")
        print("   A. Check the debug visualization image to see lane locations")
        print("   B. Lower confidence threshold even more: --lane-conf 0.05")
        print("   C. Model may need retraining on similar road scenes")
        print("   D. Use distance-based detection instead of overlap")
    elif max_overlap < 0.05:
        print(f"\n⚠️  LOW OVERLAP: Maximum overlap is only {max_overlap:.4f} ({max_overlap*100:.2f}%)")
        print("\n   Solutions:")
        print(f"   A. Lower bottom-overlap threshold to {max_overlap * 2:.4f} or lower")
        print("   B. Use: --bottom-overlap 0.001")
    else:
        print(f"\n✓ OVERLAP DETECTED: Maximum overlap is {max_overlap:.4f} ({max_overlap*100:.2f}%)")
        print(f"   Vehicle {max_overlap_vehicle+1} has highest overlap")
        print("   But threshold might be too high - try lower --bottom-overlap")
    
    # ============================================================
    # STEP 5: Create Visualization
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 5: Creating Diagnostic Visualization")
    print("=" * 70)
    
    vis_image = image.copy()
    
    # Draw lane mask in bright cyan
    if np.any(lane_mask > 0):
        lane_overlay = np.zeros_like(vis_image)
        lane_overlay[lane_mask > 0] = [255, 255, 0]  # Cyan
        vis_image = cv2.addWeighted(vis_image, 0.5, lane_overlay, 0.5, 0)
    
    # Draw vehicles with bottom edge highlighted
    for i, box in enumerate(vehicle_boxes):
        x1, y1, x2, y2 = box.astype(int)
        
        # Full box in green
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Bottom edge in yellow
        bottom_y = int(y1 + (y2 - y1) * 0.7)
        cv2.rectangle(vis_image, (x1, bottom_y), (x2, y2), (0, 255, 255), 2)
        
        # Check overlap
        bottom_region = lane_mask[bottom_y:y2, x1:x2]
        bottom_area = (y2 - bottom_y) * (x2 - x1)
        if bottom_area > 0:
            bottom_overlap = np.sum(bottom_region > 0) / bottom_area
            label = f"V{i+1}: {bottom_overlap:.4f}"
            cv2.putText(vis_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save visualization
    output_path = 'outputs/images/diagnosis_result.jpg'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, vis_image)
    print(f"\n✓ Diagnostic visualization saved to: {output_path}")
    print("\n   Legend:")
    print("   - Cyan/yellow overlay = Detected lane pixels")
    print("   - Green boxes = Vehicle bounding boxes")
    print("   - Yellow rectangles = Bottom 30% (critical area)")
    print("   - Numbers = Overlap ratio for each vehicle")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_violations.py <image_path> [model_path]")
        print("Example: python diagnose_violations.py test/test5.png runs/segment/train3/weights/best.pt")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'runs/segment/train3/weights/best.pt'
    
    diagnose_violations(image_path, model_path)

