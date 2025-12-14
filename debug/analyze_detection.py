"""
Analyze why violations are missed or false positives occur
"""
import cv2
import numpy as np
from white_line_violation import WhiteLineViolationDetector

def analyze_detection(image_path, model_path, overlap_thresholds=[0.3, 0.2, 0.4, 0.5]):
    """
    Analyze detection with different overlap thresholds to understand why violations are missed.
    """
    print("=" * 60)
    print("Analyzing Detection Issues")
    print("=" * 60)
    
    # Initialize detector
    detector = WhiteLineViolationDetector(
        lane_model_path=model_path,
        vehicle_model_path='yolov8n.pt',
        conf_threshold=0.25
    )
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"\nImage: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Detect lanes
    print("\n" + "-" * 60)
    print("LANE DETECTION ANALYSIS")
    print("-" * 60)
    lane_results = detector.detect_lanes(image)
    lane_mask = lane_results['masks']
    
    print(f"Lane mask shape: {lane_mask.shape}")
    print(f"Non-zero pixels (lane pixels): {np.sum(lane_mask > 0)}")
    print(f"Total image pixels: {lane_mask.shape[0] * lane_mask.shape[1]}")
    print(f"Lane coverage: {np.sum(lane_mask > 0) / (lane_mask.shape[0] * lane_mask.shape[1]) * 100:.2f}%")
    print(f"Detected lane classes: {lane_results['class_ids']}")
    
    if len(lane_results['class_ids']) == 0:
        print("\n⚠️  WARNING: No lanes detected!")
        print("Possible reasons:")
        print("  - Model confidence threshold too high")
        print("  - Lanes not visible or clear in image")
        print("  - Model needs more training on similar scenes")
    
    # Detect vehicles
    print("\n" + "-" * 60)
    print("VEHICLE DETECTION ANALYSIS")
    print("-" * 60)
    vehicle_results = detector.detect_vehicles(image)
    vehicle_boxes = vehicle_results['boxes']
    vehicle_scores = vehicle_results['scores']
    vehicle_classes = vehicle_results['class_names']
    
    print(f"Vehicles detected: {len(vehicle_boxes)}")
    for i, (box, score, cls_name) in enumerate(zip(vehicle_boxes, vehicle_scores, vehicle_classes)):
        print(f"  Vehicle {i+1}: {cls_name} (confidence: {score:.2f})")
    
    if len(vehicle_boxes) == 0:
        print("\n⚠️  WARNING: No vehicles detected!")
        print("Possible reasons:")
        print("  - Vehicle confidence threshold too high")
        print("  - Vehicles too small or occluded")
    
    # Check violations with different thresholds
    print("\n" + "-" * 60)
    print("VIOLATION ANALYSIS - Different Overlap Thresholds")
    print("-" * 60)
    
        for threshold in overlap_thresholds:
            violations = []
            overlaps = []
            
            for i, box in enumerate(vehicle_boxes):
                is_violation, overlap_ratio = detector.check_violation(
                    box,
                    lane_mask,
                    vehicle_box_overlap_threshold=threshold,
                    bottom_edge_overlap_threshold=0.05
                )
            overlaps.append(overlap_ratio)
            
            if is_violation:
                violations.append({
                    'vehicle_id': i,
                    'overlap': overlap_ratio,
                    'class': vehicle_classes[i],
                    'score': vehicle_scores[i]
                })
        
        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Violations detected: {len(violations)}")
        print(f"  Average overlap ratio: {np.mean(overlaps):.3f}")
        print(f"  Max overlap ratio: {np.max(overlaps) if overlaps else 0:.3f}")
        print(f"  Min overlap ratio: {np.min(overlaps) if overlaps else 0:.3f}")
        
        if violations:
            print("  Violation details:")
            for v in violations:
                print(f"    - {v['class']}: overlap={v['overlap']:.3f}, confidence={v['score']:.2f}")
    
    # Detailed per-vehicle analysis
    print("\n" + "-" * 60)
    print("DETAILED PER-VEHICLE OVERLAP ANALYSIS")
    print("-" * 60)
    
    for i, (box, score, cls_name) in enumerate(zip(vehicle_boxes, vehicle_scores, vehicle_classes)):
        is_violation_03, overlap_03 = detector.check_violation(
            box, lane_mask, vehicle_box_overlap_threshold=0.3, bottom_edge_overlap_threshold=0.05
        )
        is_violation_02, overlap_02 = detector.check_violation(
            box, lane_mask, vehicle_box_overlap_threshold=0.2, bottom_edge_overlap_threshold=0.05
        )
        is_violation_04, overlap_04 = detector.check_violation(
            box, lane_mask, vehicle_box_overlap_threshold=0.4, bottom_edge_overlap_threshold=0.05
        )
        
        print(f"\nVehicle {i+1}: {cls_name} (confidence: {score:.2f})")
        print(f"  Box coordinates: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
        print(f"  Overlap ratios:")
        print(f"    - Threshold 0.2: {overlap_02:.3f} {'✓ VIOLATION' if is_violation_02 else '✗ OK'}")
        print(f"    - Threshold 0.3: {overlap_03:.3f} {'✓ VIOLATION' if is_violation_03 else '✗ OK'}")
        print(f"    - Threshold 0.4: {overlap_04:.3f} {'✓ VIOLATION' if is_violation_04 else '✗ OK'}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if len(lane_results['class_ids']) == 0:
        print("1. ⚠️  Lower lane detection confidence threshold (--conf 0.15-0.2)")
        print("2. ⚠️  Check if lanes are clearly visible in image")
        print("3. ⚠️  Model may need more training on similar scenes")
    
    if len(vehicle_boxes) == 0:
        print("1. ⚠️  Lower vehicle detection confidence threshold")
        print("2. ⚠️  Check vehicle visibility in image")
    
    max_overlap = np.max([detector.check_violation(box, lane_mask, 0.0)[1] for box in vehicle_boxes]) if len(vehicle_boxes) > 0 else 0
    if max_overlap < 0.3:
        print(f"1. ⚠️  Maximum overlap is {max_overlap:.3f} - vehicles don't overlap much with detected lanes")
        print("   - May indicate lane detection is inaccurate")
        print("   - Try lowering overlap threshold to 0.2 or 0.15")
    elif max_overlap > 0.5:
        print(f"1. ✓ High overlap detected ({max_overlap:.3f}) - violations should be caught")
        print("   - If violations still missed, overlap threshold may be too high")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_detection.py <image_path> [model_path]")
        print("Example: python analyze_detection.py test/test1.jpg runs/segment/train3/weights/best.pt")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'runs/segment/train3/weights/best.pt'
    
    analyze_detection(image_path, model_path)

