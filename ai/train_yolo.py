"""
YOLOv8 Training Script for Electronic Component Detection
"""
import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import albumentations as A
from ai.detector import ComponentClasses

def prepare_dataset(images_dir='images', output_dir='dataset_yolo', train_ratio=0.8):
    """
    Prepare YOLO dataset from component images
    
    Args:
        images_dir: Directory containing component folders
        output_dir: Output directory for YOLO dataset
        train_ratio: Ratio of training data
    """
    print("Preparing YOLO dataset...")
    
    # Create directory structure
    train_img_dir = Path(output_dir) / 'images' / 'train'
    val_img_dir = Path(output_dir) / 'images' / 'val'
    train_lbl_dir = Path(output_dir) / 'labels' / 'train'
    val_lbl_dir = Path(output_dir) / 'labels' / 'val'
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Process each component class
    for class_id, class_name in enumerate(ComponentClasses.CLASSES):
        class_dir = Path(images_dir) / class_name
        
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping...")
            continue
        
        # Get all images
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        
        if not image_files:
            print(f"No images found in {class_dir}")
            continue
        
        print(f"Processing {class_name}: {len(image_files)} images")
        
        # Split train/val
        np.random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Process training images with augmentation
        for img_file in train_files:
            _process_image(img_file, class_id, train_img_dir, train_lbl_dir, augment=True)
        
        # Process validation images without augmentation
        for img_file in val_files:
            _process_image(img_file, class_id, val_img_dir, val_lbl_dir, augment=False)
    
    # Create data.yaml
    data_yaml = {
        'path': str(Path(output_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(ComponentClasses.CLASSES),
        'names': ComponentClasses.CLASSES
    }
    
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\nDataset prepared at {output_dir}")
    print(f"Config saved to {yaml_path}")
    
    return str(yaml_path)


def _process_image(img_file, class_id, img_dir, lbl_dir, augment=False):
    """Process single image and create YOLO label with optional augmentation"""
    img = cv2.imread(str(img_file))
    if img is None:
        return
    
    h, w = img.shape[:2]
    
    # Apply augmentation if requested
    if augment:
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
        ])
        augmented = transform(image=img)
        img = augmented['image']
    
    # Save image
    dest_img = img_dir / img_file.name
    cv2.imwrite(str(dest_img), img)
    
    # YOLO format: class_id x_center y_center width height (normalized)
    # Full image detection
    x_center = 0.5
    y_center = 0.5
    width = 0.9  # Slightly smaller to avoid edge issues
    height = 0.9
    
    # Create label file
    label_file = lbl_dir / f"{img_file.stem}.txt"
    with open(label_file, 'w') as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def train_yolo(data_yaml='dataset_yolo/data.yaml', epochs=100, imgsz=640, batch=16):
    """
    Train YOLOv8 model
    
    Args:
        data_yaml: Path to data.yaml
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
    """
    print(f"\nTraining YOLOv8 on {data_yaml}")
    print(f"Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}")
    
    # Initialize model
    model = YOLO('yolov8n.pt')  # Start with nano model
    
    # Check GPU availability
    import torch
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='component_detector',
        patience=20,
        save=True,
        device=device,
        workers=4,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: runs/detect/component_detector/weights/best.pt")
    
    return results


def validate_yolo(model_path='runs/detect/component_detector/weights/best.pt', 
                  data_yaml='dataset_yolo/data.yaml'):
    """Validate trained YOLO model"""
    print(f"\nValidating model: {model_path}")
    
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    
    print("\nValidation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results


def export_yolo(model_path='runs/detect/component_detector/weights/best.pt', format='onnx'):
    """Export YOLO model to different formats"""
    print(f"\nExporting model to {format}...")
    
    model = YOLO(model_path)
    model.export(format=format)
    
    print(f"Model exported successfully!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 for component detection')
    parser.add_argument('--prepare', action='store_true', help='Prepare dataset')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--validate', action='store_true', help='Validate model')
    parser.add_argument('--export', action='store_true', help='Export model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    
    args = parser.parse_args()
    
    if args.prepare or not any([args.train, args.validate, args.export]):
        yaml_path = prepare_dataset()
    else:
        yaml_path = 'dataset_yolo/data.yaml'
    
    if args.train:
        train_yolo(yaml_path, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
    
    if args.validate:
        validate_yolo()
    
    if args.export:
        export_yolo()
