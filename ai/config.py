"""
Configuration management for YOLO + PatchCore system
Centralized configuration for all models and training parameters
"""
from dataclasses import dataclass
from typing import List, Tuple
import yaml
from pathlib import Path

@dataclass
class YOLOConfig:
    """YOLO training configuration"""
    model_size: str = 'yolov8n.pt'  # n, s, m, l, x
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    patience: int = 20
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', '0', '1', etc.
    workers: int = 4
    
    # Augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    
    # Paths
    data_yaml: str = 'dataset_yolo/data.yaml'
    output_dir: str = 'runs/detect'
    weights_dir: str = 'runs/detect/component_detector/weights'

@dataclass
class PatchCoreConfig:
    """PatchCore training configuration"""
    backbone: str = 'wide_resnet50_2'
    max_images: int = 200
    coreset_ratio: float = 0.1
    device: str = 'auto'
    threshold: float = 0.5
    
    # Feature extraction
    feature_layers: List[int] = None
    target_size: Tuple[int, int] = (224, 224)
    
    # Paths
    output_dir: str = 'models/patchcore'
    
    def __post_init__(self):
        if self.feature_layers is None:
            self.feature_layers = [2, 3]

@dataclass
class DataConfig:
    """Dataset configuration"""
    images_dir: str = 'images'
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    
    # Preprocessing
    normalize: bool = True
    resize_mode: str = 'pad'  # 'pad', 'stretch', 'crop'
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_strength: str = 'medium'  # 'light', 'medium', 'heavy'

@dataclass
class InspectionConfig:
    """Inspection pipeline configuration"""
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    patchcore_threshold: float = 0.5
    
    # Multi-camera fusion
    fusion_method: str = 'max'  # 'max', 'mean', 'vote'
    min_views: int = 1
    
    # Output
    save_crops: bool = False
    save_heatmaps: bool = True
    save_visualizations: bool = True

@dataclass
class SystemConfig:
    """Complete system configuration"""
    yolo: YOLOConfig = None
    patchcore: PatchCoreConfig = None
    data: DataConfig = None
    inspection: InspectionConfig = None
    
    def __post_init__(self):
        if self.yolo is None:
            self.yolo = YOLOConfig()
        if self.patchcore is None:
            self.patchcore = PatchCoreConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.inspection is None:
            self.inspection = InspectionConfig()
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'yolo': self.yolo.__dict__,
            'patchcore': self.patchcore.__dict__,
            'data': self.data.__dict__,
            'inspection': self.inspection.__dict__
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        print(f"Configuration saved to: {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            yolo=YOLOConfig(**config_dict.get('yolo', {})),
            patchcore=PatchCoreConfig(**config_dict.get('patchcore', {})),
            data=DataConfig(**config_dict.get('data', {})),
            inspection=InspectionConfig(**config_dict.get('inspection', {}))
        )
    
    def print_summary(self):
        """Print configuration summary"""
        print("="*60)
        print("SYSTEM CONFIGURATION")
        print("="*60)
        
        print("\nYOLO Configuration:")
        print(f"  Model: {self.yolo.model_size}")
        print(f"  Epochs: {self.yolo.epochs}")
        print(f"  Batch size: {self.yolo.batch_size}")
        print(f"  Image size: {self.yolo.image_size}")
        
        print("\nPatchCore Configuration:")
        print(f"  Backbone: {self.patchcore.backbone}")
        print(f"  Max images: {self.patchcore.max_images}")
        print(f"  Coreset ratio: {self.patchcore.coreset_ratio}")
        
        print("\nData Configuration:")
        print(f"  Images dir: {self.data.images_dir}")
        print(f"  Train ratio: {self.data.train_ratio}")
        print(f"  Augmentation: {self.data.use_augmentation}")
        
        print("\nInspection Configuration:")
        print(f"  YOLO threshold: {self.inspection.yolo_conf_threshold}")
        print(f"  PatchCore threshold: {self.inspection.patchcore_threshold}")
        print(f"  Fusion method: {self.inspection.fusion_method}")
        print("="*60)

# Default configuration
DEFAULT_CONFIG = SystemConfig()

# Quick training configuration (for testing)
QUICK_CONFIG = SystemConfig(
    yolo=YOLOConfig(epochs=30, batch_size=16),
    patchcore=PatchCoreConfig(max_images=100, coreset_ratio=0.1)
)

# Production configuration (best accuracy)
PRODUCTION_CONFIG = SystemConfig(
    yolo=YOLOConfig(epochs=150, batch_size=32, model_size='yolov8m.pt'),
    patchcore=PatchCoreConfig(max_images=300, coreset_ratio=0.05)
)

def get_config(mode: str = 'default') -> SystemConfig:
    """
    Get configuration by mode
    
    Args:
        mode: 'default', 'quick', or 'production'
    
    Returns:
        SystemConfig instance
    """
    if mode == 'quick':
        return QUICK_CONFIG
    elif mode == 'production':
        return PRODUCTION_CONFIG
    else:
        return DEFAULT_CONFIG

if __name__ == '__main__':
    # Example: Save default configuration
    config = DEFAULT_CONFIG
    config.save('config/default_config.yaml')
    config.print_summary()
