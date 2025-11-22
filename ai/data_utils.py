"""
Data utilities for YOLO + PatchCore training
Includes preprocessing, augmentation, and dataset management
"""
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import List, Tuple, Optional
import random

class DataAugmentor:
    """Data augmentation pipeline for industrial images"""
    
    def __init__(self, mode='train'):
        """
        Initialize augmentor
        
        Args:
            mode: 'train' for aggressive augmentation, 'val' for minimal
        """
        self.mode = mode
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
    
    def _get_train_transform(self):
        """Get training augmentation pipeline"""
        return A.Compose([
            # Geometric transforms
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT,
                p=0.6
            ),
            
            # Color transforms
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(),
                A.MultiplicativeNoise(),
            ], p=0.4),
            
            A.OneOf([
                A.Blur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            
            # Lighting effects
            A.RandomShadow(p=0.3),
            A.RandomFog(p=0.2),
            
            # Quality degradation
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
        ])
    
    def _get_val_transform(self):
        """Get validation transform (minimal)"""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation"""
        if self.mode == 'train':
            augmented = self.train_transform(image=image)
            return augmented['image']
        else:
            augmented = self.val_transform(image=image)
            return augmented['image']


class ComponentDataset:
    """Dataset manager for component images"""
    
    def __init__(self, root_dir: str, split: str = 'train', train_ratio: float = 0.8):
        """
        Initialize dataset
        
        Args:
            root_dir: Root directory containing component folders
            split: 'train' or 'val'
            train_ratio: Ratio of training data
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.train_ratio = train_ratio
        self.samples = []
        
        self._load_samples()
    
    def _load_samples(self):
        """Load all image samples"""
        for comp_dir in self.root_dir.iterdir():
            if not comp_dir.is_dir():
                continue
            
            # Get all images
            images = list(comp_dir.glob('*.jpg')) + \
                    list(comp_dir.glob('*.jpeg')) + \
                    list(comp_dir.glob('*.png'))
            
            if not images:
                continue
            
            # Shuffle and split
            random.shuffle(images)
            split_idx = int(len(images) * self.train_ratio)
            
            if self.split == 'train':
                selected = images[:split_idx]
            else:
                selected = images[split_idx:]
            
            for img_path in selected:
                self.samples.append({
                    'path': img_path,
                    'component': comp_dir.name
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(str(sample['path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return {
            'image': image,
            'component': sample['component'],
            'path': sample['path']
        }


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess image for model input
    
    Args:
        image: Input image (RGB)
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed image
    """
    # Resize while maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size
    pad_w = target_size[0] - new_w
    pad_h = target_size[1] - new_h
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    return padded


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range"""
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from [0, 1] to [0, 255]"""
    return (image * 255).astype(np.uint8)


def create_defect_mask(image: np.ndarray, defect_type: str = 'scratch') -> np.ndarray:
    """
    Create synthetic defect mask for testing
    
    Args:
        image: Input image
        defect_type: Type of defect ('scratch', 'spot', 'crack')
    
    Returns:
        Defect mask
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if defect_type == 'scratch':
        # Random line
        pt1 = (random.randint(0, w), random.randint(0, h))
        pt2 = (random.randint(0, w), random.randint(0, h))
        cv2.line(mask, pt1, pt2, 255, thickness=random.randint(2, 5))
    
    elif defect_type == 'spot':
        # Random circle
        center = (random.randint(0, w), random.randint(0, h))
        radius = random.randint(10, 30)
        cv2.circle(mask, center, radius, 255, -1)
    
    elif defect_type == 'crack':
        # Random polygon
        points = np.array([
            [random.randint(0, w), random.randint(0, h)]
            for _ in range(random.randint(3, 6))
        ])
        cv2.fillPoly(mask, [points], 255)
    
    return mask


def apply_synthetic_defect(image: np.ndarray, defect_type: str = 'scratch') -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply synthetic defect to image
    
    Args:
        image: Input image (RGB)
        defect_type: Type of defect
    
    Returns:
        Defective image and defect mask
    """
    mask = create_defect_mask(image, defect_type)
    
    # Apply defect
    defective = image.copy()
    
    if defect_type == 'scratch':
        # Darken scratched area
        defective[mask > 0] = (defective[mask > 0] * 0.5).astype(np.uint8)
    
    elif defect_type == 'spot':
        # Add bright spot
        defective[mask > 0] = np.minimum(defective[mask > 0] + 50, 255)
    
    elif defect_type == 'crack':
        # Add dark crack
        defective[mask > 0] = (defective[mask > 0] * 0.3).astype(np.uint8)
    
    return defective, mask


def balance_dataset(samples: List[dict], max_per_class: Optional[int] = None) -> List[dict]:
    """
    Balance dataset by limiting samples per class
    
    Args:
        samples: List of sample dictionaries
        max_per_class: Maximum samples per class
    
    Returns:
        Balanced samples
    """
    if max_per_class is None:
        return samples
    
    # Group by component
    grouped = {}
    for sample in samples:
        comp = sample['component']
        if comp not in grouped:
            grouped[comp] = []
        grouped[comp].append(sample)
    
    # Limit each class
    balanced = []
    for comp, comp_samples in grouped.items():
        if len(comp_samples) > max_per_class:
            comp_samples = random.sample(comp_samples, max_per_class)
        balanced.extend(comp_samples)
    
    random.shuffle(balanced)
    return balanced


def compute_class_weights(samples: List[dict]) -> dict:
    """
    Compute class weights for imbalanced dataset
    
    Args:
        samples: List of sample dictionaries
    
    Returns:
        Dictionary of class weights
    """
    # Count samples per class
    class_counts = {}
    for sample in samples:
        comp = sample['component']
        class_counts[comp] = class_counts.get(comp, 0) + 1
    
    # Compute weights (inverse frequency)
    total = len(samples)
    weights = {}
    for comp, count in class_counts.items():
        weights[comp] = total / (len(class_counts) * count)
    
    return weights
