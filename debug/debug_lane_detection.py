"""
Debug script to visualize what lanes are being detected
"""
import cv2
import numpy as np
from white_line_violation import WhiteLineViolationDetector

def debug_lane_detection(image_path):
    """Debug what the lane model is actually detecting."""
    
    # Initialize detector
    detector = WhiteLineViolationDetector(
        lane_model_path='yolov8n-seg.pt',
        vehicle_model_path='yolov8n.pt',
        conf_threshold=0.25
    )
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Processing: {image_path}")
    print("-" * 60)
    
    # Detect lanes - get detailed info
    results = detector.lane_model(image, conf=0.25, verbose=False)
    result = results[0]
    
    print(f"Total detections: {len(result.boxes) if result.boxes is not None else 0}")
    
    if result.boxes is not None:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        
        print("\nDetected classes (COCO classes, not lanes!):")
        for i, (cls_id, conf) in enumerate(zip(classes, confidences)):
            class_name = result.names[cls_id]
            print(f"  {i+1}. {class_name} (class {cls_id}) - confidence: {conf:.2f}")
    
    # Now use our detect_lanes method
    lane_results = detector.detect_lanes(image)
    
    print(f"\nLane mask info:")
    print(f"  Lane mask shape: {lane_results['masks'].shape}")
    print(f"  Non-zero pixels: {np.sum(lane_results['masks'] > 0)}")
    print(f"  Detected class IDs: {lane_results['class_ids']}")
    
    # Check what classes we're filtering for
    print(f"\nFiltering for violation classes: {detector.VIOLATION_LANE_CLASSES}")
    print("Note: These are indices 2 and 5 from YOUR dataset classes:")
    print("  - Index 2: double-line")
    print("  - Index 5: solid-line")
    
    # The problem: pretrained model uses COCO classes, not our lane classes!
    print("\n" + "!" * 60)
    print("PROBLEM: The pretrained yolov8n-seg.pt uses COCO classes")
    print("It doesn't know what 'solid-line' or 'double-line' means!")
    print("You need to train a model on your jpj_dataset first.")
    print("!" * 60)
    
    # Create visualization
    output = image.copy()
    
    if lane_results['masks'] is not None and np.any(lane_results['masks'] > 0):
        # Overlay lane mask
        mask_overlay = np.zeros_like(output)
        mask_overlay[lane_results['masks'] > 0] = [255, 0, 255]
        output = cv2.addWeighted(output, 0.7, mask_overlay, 0.3, 0)
    
    # Save debug image
    debug_path = 'outputs/images/debug_lanes.jpg'
    cv2.imwrite(debug_path, output)
    print(f"\nDebug visualization saved to: {debug_path}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'test/test1.jpg'
    
    debug_lane_detection(image_path)

