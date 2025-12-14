"""
Convert COCO format annotations to YOLO format for segmentation.
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import cv2


def convert_coco_to_yolo_segmentation(coco_json_path, images_dir, labels_dir, category_mapping=None):
    """
    Convert COCO format segmentation annotations to YOLO format.
    
    Args:
        coco_json_path: Path to COCO JSON annotation file
        images_dir: Directory containing images
        labels_dir: Directory to save YOLO format label files
        category_mapping: Dict mapping COCO category IDs to YOLO class IDs (0-based)
    """
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create category mapping if not provided
    if category_mapping is None:
        # Create mapping from COCO category ID to YOLO class ID (0-based)
        categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
        print(f"Category mapping: {category_mapping}")
    
    # YOLO expects labels in same directory as images, not in subdirectory
    # labels_dir parameter is kept for compatibility but we'll use images_dir instead
    images_path = Path(images_dir)
    
    # Create image ID to filename mapping
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        annotations_by_image[image_id].append(ann)
    
    # Get image dimensions
    image_dims = {}
    images_path = Path(images_dir)
    
    # Process each image
    for image_id, annotations in annotations_by_image.items():
        image_info = image_id_to_info[image_id]
        image_filename = image_info['file_name']
        image_path = images_path / image_filename
        
        # Get image dimensions
        if image_path.exists():
            img = cv2.imread(str(image_path))
            if img is not None:
                height, width = img.shape[:2]
                image_dims[image_id] = (width, height)
            else:
                print(f"Warning: Could not read image {image_path}")
                continue
        else:
            print(f"Warning: Image not found {image_path}")
            continue
        
        # Create label file (YOLO expects labels in same directory as images, not in subdirectory)
        label_filename = Path(image_filename).stem + '.txt'
        # Put labels in same directory as images (not in labels subdirectory)
        label_path = images_path / label_filename
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                category_id = ann['category_id']
                
                # Skip if category not in mapping (e.g., 'road-roads')
                if category_id not in category_mapping:
                    continue
                
                yolo_class_id = category_mapping[category_id]
                segmentation = ann['segmentation']
                
                # Handle different segmentation formats
                if isinstance(segmentation, list) and len(segmentation) > 0:
                    # Polygon format: [x1, y1, x2, y2, ...]
                    if isinstance(segmentation[0], list):
                        # Multiple polygons - use the first one (or largest)
                        polygons = segmentation
                        # Use the largest polygon
                        largest_poly = max(polygons, key=len)
                        polygon = largest_poly
                    else:
                        # Single polygon
                        polygon = segmentation
                    
                    # Normalize coordinates (0-1 range)
                    normalized_polygon = []
                    for i in range(0, len(polygon), 2):
                        if i + 1 < len(polygon):
                            x = polygon[i] / width
                            y = polygon[i + 1] / height
                            # Clamp to [0, 1]
                            x = max(0, min(1, x))
                            y = max(0, min(1, y))
                            normalized_polygon.extend([x, y])
                    
                    if len(normalized_polygon) >= 6:  # At least 3 points
                        # Write in YOLO format: class_id x1 y1 x2 y2 x3 y3 ...
                        line = f"{yolo_class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_polygon])
                        f.write(line + "\n")
    
    print(f"Converted annotations. Labels saved to {images_path}")
    return category_mapping


def main():
    """Main conversion function."""
    base_dir = Path("data/jpj_dataset")
    
    # Define category mapping (excluding 'road-roads' category 0, remapping others)
    # COCO category IDs: 0='road-roads', 1='divider-line', 2='dotted-line', 3='double-line', 
    #                    4='random-line', 5='road-sign-line', 6='solid-line'
    # YOLO class IDs: 0='divider-line', 1='dotted-line', 2='double-line', 
    #                 3='random-line', 4='road-sign-line', 5='solid-line'
    category_mapping = {
        1: 0,  # divider-line
        2: 1,  # dotted-line
        3: 2,  # double-line
        4: 3,  # random-line
        5: 4,  # road-sign-line
        6: 5,  # solid-line
        # Skip category 0 (road-roads)
    }
    
    # Convert validation set
    print("Converting validation set...")
    convert_coco_to_yolo_segmentation(
        coco_json_path=base_dir / "annotations" / "instances_val2017.json",
        images_dir=base_dir / "val2017",
        labels_dir=base_dir / "val2017" / "labels",
        category_mapping=category_mapping
    )
    
    # Convert training set
    print("\nConverting training set...")
    convert_coco_to_yolo_segmentation(
        coco_json_path=base_dir / "annotations" / "instances_train2017.json",
        images_dir=base_dir / "train2017",
        labels_dir=base_dir / "train2017" / "labels",
        category_mapping=category_mapping
    )
    
    print("\nConversion complete!")


if __name__ == "__main__":
    main()

