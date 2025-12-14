"""
Example usage script for White-Line Violation Detection System

This script demonstrates how to use the WhiteLineViolationDetector class
programmatically in your own code.
"""

import cv2
from pathlib import Path
from white_line_violation import WhiteLineViolationDetector

def example_image_processing():
    """Example: Process a single image."""
    print("=" * 60)
    print("Example 1: Processing a single image")
    print("=" * 60)
    
    # Initialize detector
    detector = WhiteLineViolationDetector(
        lane_model_path='yolov8n-seg.pt',  # Your trained model
        vehicle_model_path='yolov8n.pt',   # Pretrained COCO model
        conf_threshold=0.25
    )
    
    # Load image
    image_path = 'data/jpj_dataset/val2017'
    image_files = list(Path(image_path).glob('*.jpg'))
    
    if not image_files:
        print(f"No images found in {image_path}")
        return
    
    # Process first image
    image_file = image_files[0]
    print(f"Processing: {image_file}")
    
    image = cv2.imread(str(image_file))
    if image is None:
        print(f"Could not load image: {image_file}")
        return
    
    # Process image
    result = detector.process_image(
        image,
        visualize=True,
        overlap_threshold=0.3,
        bottom_overlap_threshold=0.05
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Vehicles detected: {result['stats']['total_vehicles']}")
    print(f"  Violations: {result['stats']['violations']}")
    print(f"  Lanes detected: {result['stats']['lanes_detected']}")
    
    if result['violations']:
        print("\n  Violation details:")
        for i, violation in enumerate(result['violations'], 1):
            print(f"    {i}. {violation['class']} "
                  f"(confidence: {violation['score']:.2f}, "
                  f"overlap: {violation['overlap_ratio']:.2f})")
    
    # Save output
    output_path = 'outputs/images/example_output.jpg'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, result['image'])
    print(f"\nOutput saved to: {output_path}")


def example_video_processing():
    """Example: Process a video file."""
    print("\n" + "=" * 60)
    print("Example 2: Processing a video file")
    print("=" * 60)
    
    # Initialize detector
    detector = WhiteLineViolationDetector(
        lane_model_path='yolov8n-seg.pt',
        vehicle_model_path='yolov8n.pt',
        conf_threshold=0.25
    )
    
    # Process video
    input_video = 'path/to/input_video.mp4'
    output_video = 'outputs/videos/output_video.mp4'
    
    if not Path(input_video).exists():
        print(f"Video file not found: {input_video}")
        print("Skipping video example...")
        return
    
    print(f"Processing video: {input_video}")
    
    stats = detector.process_video(
        input_path=input_video,
        output_path=output_video,
        overlap_threshold=0.3
    )
    
    print(f"\nVideo processing complete!")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Total violations: {stats['total_violations']}")
    print(f"  Output saved to: {stats['output_path']}")


def example_modular_usage():
    """Example: Using individual detection functions."""
    print("\n" + "=" * 60)
    print("Example 3: Modular usage - separate detections")
    print("=" * 60)
    
    # Initialize detector
    detector = WhiteLineViolationDetector(
        lane_model_path='yolov8n-seg.pt',
        vehicle_model_path='yolov8n.pt'
    )
    
    # Load image
    image_path = 'data/jpj_dataset/val2017'
    image_files = list(Path(image_path).glob('*.jpg'))
    
    if not image_files:
        print(f"No images found in {image_path}")
        return
    
    image = cv2.imread(str(image_files[0]))
    if image is None:
        return
    
    # Step 1: Detect lanes separately
    print("\n1. Detecting lanes...")
    lane_results = detector.detect_lanes(image)
    print(f"   Detected {len(lane_results['class_ids'])} lane types")
    print(f"   Classes: {lane_results['class_ids']}")
    
    # Step 2: Detect vehicles separately
    print("\n2. Detecting vehicles...")
    vehicle_results = detector.detect_vehicles(image)
    print(f"   Detected {len(vehicle_results['boxes'])} vehicles")
    for i, (box, score, cls_name) in enumerate(
        zip(vehicle_results['boxes'], vehicle_results['scores'], vehicle_results['class_names'])
    ):
        print(f"   Vehicle {i+1}: {cls_name} (confidence: {score:.2f})")
    
    # Step 3: Check violations for each vehicle
    print("\n3. Checking violations...")
    for i, box in enumerate(vehicle_results['boxes']):
        is_violation, overlap = detector.check_violation(
            box,
            lane_results['masks'],
            vehicle_box_overlap_threshold=0.3,
            bottom_edge_overlap_threshold=0.05
        )
        status = "VIOLATION" if is_violation else "OK"
        print(f"   Vehicle {i+1}: {status} (overlap: {overlap:.2f})")


if __name__ == '__main__':
    # Run examples
    example_image_processing()
    # example_video_processing()  # Uncomment if you have a video file
    example_modular_usage()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)

