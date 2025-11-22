"""
PatchCore Training Script for Anomaly Detection
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ai.detector import PatchCoreModel, ComponentClasses
from tqdm import tqdm

def load_normal_images(component_dir: str, max_images=200):
    """Load normal (good) images for a component"""
    images = []
    image_files = list(Path(component_dir).glob('*.jpg')) + \
                  list(Path(component_dir).glob('*.jpeg')) + \
                  list(Path(component_dir).glob('*.png'))
    
    # Limit number of images for training
    if len(image_files) > max_images:
        image_files = np.random.choice(image_files, max_images, replace=False)
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    
    return images


def train_patchcore_for_component(component_name: str, 
                                   images_dir='images',
                                   output_dir='models/patchcore',
                                   max_images=200,
                                   coreset_ratio=0.1):
    """
    Train PatchCore model for a specific component
    
    Args:
        component_name: Name of the component class
        images_dir: Root directory containing component folders
        output_dir: Output directory for trained models
        max_images: Maximum number of images to use for training
        coreset_ratio: Ratio of coreset samples
    """
    print(f"\n{'='*60}")
    print(f"Training PatchCore for: {component_name}")
    print(f"{'='*60}")
    
    # Load normal images
    component_dir = Path(images_dir) / component_name
    if not component_dir.exists():
        print(f"Error: {component_dir} not found!")
        return False
    
    print(f"Loading images from {component_dir}...")
    normal_images = load_normal_images(str(component_dir), max_images)
    
    if len(normal_images) < 10:
        print(f"Warning: Only {len(normal_images)} images found. Need at least 10.")
        return False
    
    print(f"Loaded {len(normal_images)} normal images")
    
    # Initialize and train model
    model = PatchCoreModel()
    model.fit(normal_images, coreset_ratio=coreset_ratio)
    
    # Save model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(output_dir) / f"{component_name}.pkl"
    model.save(str(model_path))
    
    print(f"✓ Model saved to {model_path}")
    return True


def train_all_components(images_dir='images', 
                         output_dir='models/patchcore',
                         max_images=200,
                         coreset_ratio=0.1):
    """Train PatchCore models for all component classes"""
    print("Training PatchCore models for all components...")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max images per class: {max_images}")
    print(f"Coreset ratio: {coreset_ratio}")
    
    success_count = 0
    failed_components = []
    
    for component_name in tqdm(ComponentClasses.CLASSES, desc="Training components"):
        try:
            success = train_patchcore_for_component(
                component_name,
                images_dir=images_dir,
                output_dir=output_dir,
                max_images=max_images,
                coreset_ratio=coreset_ratio
            )
            if success:
                success_count += 1
            else:
                failed_components.append(component_name)
        except Exception as e:
            print(f"Error training {component_name}: {e}")
            failed_components.append(component_name)
    
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    print(f"Successfully trained: {success_count}/{len(ComponentClasses.CLASSES)}")
    
    if failed_components:
        print(f"\nFailed components:")
        for comp in failed_components:
            print(f"  - {comp}")
    
    return success_count, failed_components


def test_patchcore_model(component_name: str,
                         model_dir='models/patchcore',
                         test_images_dir='images',
                         num_test=5):
    """Test a trained PatchCore model"""
    print(f"\nTesting PatchCore model for: {component_name}")
    
    # Load model
    model_path = Path(model_dir) / f"{component_name}.pkl"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    model = PatchCoreModel()
    model.load(str(model_path))
    
    # Load test images
    component_dir = Path(test_images_dir) / component_name
    image_files = list(component_dir.glob('*.jpg'))[:num_test]
    
    print(f"Testing on {len(image_files)} images...")
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Predict
        score, heatmap = model.predict(img_rgb)
        
        print(f"  {img_file.name}: Anomaly Score = {score:.4f}")
        
        # Visualize
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Overlay
        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        
        # Save visualization
        vis_dir = Path('visualizations') / component_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_dir / f"{img_file.stem}_anomaly.jpg"), overlay)
    
    print(f"Visualizations saved to visualizations/{component_name}/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PatchCore for anomaly detection')
    parser.add_argument('--component', type=str, help='Train specific component')
    parser.add_argument('--all', action='store_true', help='Train all components')
    parser.add_argument('--test', type=str, help='Test specific component')
    parser.add_argument('--images-dir', type=str, default='images', help='Images directory')
    parser.add_argument('--output-dir', type=str, default='models/patchcore', help='Output directory')
    parser.add_argument('--max-images', type=int, default=200, help='Max images per component')
    parser.add_argument('--coreset-ratio', type=float, default=0.1, help='Coreset sampling ratio')
    
    args = parser.parse_args()
    
    if args.test:
        test_patchcore_model(args.test)
    elif args.component:
        train_patchcore_for_component(
            args.component,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            max_images=args.max_images,
            coreset_ratio=args.coreset_ratio
        )
    elif args.all:
        train_all_components(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            max_images=args.max_images,
            coreset_ratio=args.coreset_ratio
        )
    else:
        print("Please specify --component, --all, or --test")
        parser.print_help()
