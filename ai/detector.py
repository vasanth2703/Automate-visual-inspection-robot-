"""
YOLO + PatchCore Industrial Inspection System
Comprehensive 2-stage defect detection pipeline
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import base64
from ultralytics import YOLO
import timm
from sklearn.random_projection import SparseRandomProjection
from scipy.ndimage import gaussian_filter
import pickle

class ComponentClasses:
    """Component class definitions"""
    CLASSES = [
        'armature', 'attenuator', 'Bypass-capacitor', 'cartridge-fuse', 
        'clip-lead', 'electric-relay', 'Electrolytic-capacitor', 'filament',
        'heat-sink', 'induction-coil', 'Integrated-micro-circuit', 'jumper-cable',
        'junction-transistor', 'LED', 'light-circuit', 'limiter-clipper',
        'local-oscillator', 'memory-chip', 'microchip', 'microprocessor',
        'multiplexer', 'omni-directional-antenna', 'PNP-transistor', 
        'potential-divider', 'potentiometer', 'pulse-generator', 'relay',
        'rheostat', 'semi-conductor', 'semiconductor-diode', 'shunt',
        'solenoid', 'stabilizer', 'step-down-transformer', 'step-up-transformer',
        'transistor'
    ]
    
    @classmethod
    def get_class_id(cls, name: str) -> int:
        """Get class ID from name"""
        try:
            return cls.CLASSES.index(name)
        except ValueError:
            return -1
    
    @classmethod
    def get_class_name(cls, idx: int) -> str:
        """Get class name from ID"""
        if 0 <= idx < len(cls.CLASSES):
            return cls.CLASSES[idx]
        return "unknown"


class PatchCoreModel:
    """PatchCore anomaly detection model"""
    
    def __init__(self, backbone='wide_resnet50_2', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.backbone_name = backbone
        self.feature_extractor = None
        self.memory_bank = None
        self.projection = None
        self.mean = None
        self.std = None
        self.threshold = 0.5
        
        self._init_feature_extractor()
    
    def _init_feature_extractor(self):
        """Initialize feature extraction backbone"""
        self.feature_extractor = timm.create_model(
            self.backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=[2, 3]
        )
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract patch features from image"""
        # Preprocess
        img = cv2.resize(image, (224, 224))
        img = img.astype(np.float32) / 255.0
        
        if self.mean is None:
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
        
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        img = img.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(img)
            
            # Aggregate multi-scale features
            feat_list = []
            for feat in features:
                # Resize to common size
                feat_resized = F.interpolate(
                    feat, size=(28, 28), mode='bilinear', align_corners=False
                )
                feat_list.append(feat_resized)
            
            # Concatenate
            features = torch.cat(feat_list, dim=1)
            
            # Reshape to patch features
            b, c, h, w = features.shape
            features = features.reshape(b, c, h * w).permute(0, 2, 1)
            
            return features.cpu().numpy()[0]  # Shape: (N_patches, C)
    
    def fit(self, normal_images: List[np.ndarray], coreset_ratio=0.1):
        """Train PatchCore on normal images with memory optimization"""
        print(f"Training PatchCore on {len(normal_images)} normal images...")
        
        # Limit number of images to prevent memory issues
        max_train_images = min(len(normal_images), 50)
        if len(normal_images) > max_train_images:
            import random
            normal_images = random.sample(normal_images, max_train_images)
            print(f"Limited to {max_train_images} images to prevent memory issues")
        
        # Extract features from all normal images
        all_features = []
        for img in normal_images:
            features = self.extract_features(img)
            all_features.append(features)
        
        # Stack all features
        all_features = np.vstack(all_features)  # Shape: (N_total_patches, C)
        print(f"Extracted {all_features.shape[0]} patches with {all_features.shape[1]} features")
        
        # Apply coreset subsampling with smaller ratio
        n_samples = int(len(all_features) * coreset_ratio)
        n_samples = max(min(n_samples, 500), 50)  # Between 50-500 samples
        
        indices = self._greedy_coreset_selection(all_features, n_samples)
        self.memory_bank = all_features[indices]
        
        print(f"Coreset size: {len(self.memory_bank)}")
        
        # Always apply dimensionality reduction to save memory
        if self.memory_bank.shape[1] > 512:
            self.projection = SparseRandomProjection(n_components=512, random_state=42)
            self.memory_bank = self.projection.fit_transform(self.memory_bank)
            print(f"Reduced dimensions to 512")
    
    def _greedy_coreset_selection(self, features: np.ndarray, n_samples: int) -> np.ndarray:
        """Greedy coreset subsampling"""
        n_total = len(features)
        
        if n_samples >= n_total:
            return np.arange(n_total)
        
        # Random sampling for efficiency (greedy is too slow for large datasets)
        indices = np.random.choice(n_total, n_samples, replace=False)
        return indices
    
    def predict(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Predict anomaly score and generate heatmap"""
        if self.memory_bank is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Extract features
        features = self.extract_features(image)
        
        # Apply projection if exists
        if self.projection is not None:
            features = self.projection.transform(features)
        
        # Compute distances to memory bank
        distances = self._compute_distances(features, self.memory_bank)
        
        # Anomaly score per patch (minimum distance)
        patch_scores = np.min(distances, axis=1)
        
        # Overall anomaly score (max of patch scores)
        anomaly_score = np.max(patch_scores)
        
        # Normalize score to 0-1 range using sigmoid-like function
        # This prevents extremely large values
        normalized_score = 1.0 / (1.0 + np.exp(-0.1 * (anomaly_score - 10)))
        
        # Generate heatmap
        heatmap = self._generate_heatmap(patch_scores, image.shape[:2])
        
        return float(normalized_score), heatmap
    
    def _compute_distances(self, features: np.ndarray, memory_bank: np.ndarray) -> np.ndarray:
        """Compute pairwise distances efficiently"""
        # Use batched computation to avoid memory issues
        batch_size = 100
        n_features = features.shape[0]
        distances = np.zeros((n_features, memory_bank.shape[0]))
        
        for i in range(0, n_features, batch_size):
            end_idx = min(i + batch_size, n_features)
            batch = features[i:end_idx]
            # Compute distances for this batch
            distances[i:end_idx] = np.linalg.norm(
                batch[:, np.newaxis, :] - memory_bank[np.newaxis, :, :],
                axis=2
            )
        
        return distances
    
    def _generate_heatmap(self, patch_scores: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Generate anomaly heatmap"""
        # Reshape patch scores to spatial grid
        grid_size = int(np.sqrt(len(patch_scores)))
        heatmap = patch_scores.reshape(grid_size, grid_size)
        
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Resize to target shape
        heatmap = cv2.resize(heatmap, (target_shape[1], target_shape[0]))
        
        # Apply Gaussian smoothing
        heatmap = gaussian_filter(heatmap, sigma=4)
        
        return heatmap
    
    def save(self, path: str):
        """Save model"""
        data = {
            'memory_bank': self.memory_bank,
            'projection': self.projection,
            'mean': self.mean,
            'std': self.std,
            'threshold': self.threshold
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"PatchCore model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.memory_bank = data['memory_bank']
        self.projection = data.get('projection')
        self.mean = data.get('mean')
        self.std = data.get('std')
        self.threshold = data.get('threshold', 0.5)
        print(f"PatchCore model loaded from {path}")


class YOLODetector:
    """YOLOv8 multi-class component detector"""
    
    def __init__(self, model_path: Optional[str] = None, conf_threshold=0.25):
        self.conf_threshold = conf_threshold
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with pretrained YOLOv8
            self.model = YOLO('yolov8n.pt')
    
    def load_model(self, path: str):
        """Load trained YOLO model"""
        self.model = YOLO(path)
        print(f"YOLO model loaded from {path}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect components in image"""
        if self.model is None:
            return []
        
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                detections.append({
                    'class_id': cls_id,
                    'class_name': ComponentClasses.get_class_name(cls_id),
                    'bbox': box.tolist(),
                    'confidence': conf
                })
        
        return detections


class IndustrialInspector:
    """Combined YOLO + PatchCore inspection system"""
    
    def __init__(self, yolo_path: Optional[str] = None, patchcore_dir: str = 'models/patchcore'):
        self.yolo = YOLODetector(yolo_path)
        self.patchcore_models = {}
        self.patchcore_dir = patchcore_dir
        
        # Load PatchCore models if available
        self._load_patchcore_models()
    
    def _load_patchcore_models(self):
        """Load pre-trained PatchCore models for each component"""
        if not os.path.exists(self.patchcore_dir):
            return
        
        for class_name in ComponentClasses.CLASSES:
            model_path = os.path.join(self.patchcore_dir, f"{class_name}.pkl")
            if os.path.exists(model_path):
                try:
                    model = PatchCoreModel()
                    model.load(model_path)
                    self.patchcore_models[class_name] = model
                except Exception as e:
                    print(f"Failed to load PatchCore for {class_name}: {e}")
    
    def inspect(self, image: np.ndarray, return_crops: bool = False) -> List[Dict]:
        """
        Full inspection pipeline: YOLO detection + PatchCore anomaly detection
        
        Args:
            image: Input image (RGB format)
            return_crops: If True, include cropped component images in results
        
        Returns:
            List of detection results with anomaly scores
        """
        # Stage 1: YOLO detection
        detections = self.yolo.detect(image)
        
        if not detections:
            return []
        
        # Stage 2: PatchCore anomaly detection for each component
        results = []
        for det in detections:
            class_name = det['class_name']
            bbox = det['bbox']
            
            # Crop component region with padding
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add small padding
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image.shape[1], x2 + pad)
            y2 = min(image.shape[0], y2 + pad)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = image[y1:y2, x1:x2]
            
            # Ensure crop is valid
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            
            # Run PatchCore if model exists
            anomaly_score = 0.0
            status = "UNKNOWN"
            heatmap_b64 = None
            
            if class_name in self.patchcore_models:
                try:
                    score, heatmap = self.patchcore_models[class_name].predict(crop)
                    anomaly_score = float(score)
                    
                    # Determine status
                    threshold = self.patchcore_models[class_name].threshold
                    status = "DEFECTIVE" if score > threshold else "NORMAL"
                    
                    # Encode heatmap
                    heatmap_b64 = self._encode_heatmap(heatmap)
                except Exception as e:
                    print(f"PatchCore failed for {class_name}: {e}")
                    status = "ERROR"
                    anomaly_score = 0.0
            
            result = {
                'component': class_name,
                'bbox': bbox,
                'yolo_conf': det['confidence'],
                'anomaly_score': anomaly_score,
                'status': status,
                'heatmap': heatmap_b64
            }
            
            if return_crops:
                # Encode crop as base64
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                crop_b64 = base64.b64encode(buffer).decode('utf-8')
                result['crop'] = crop_b64
            
            results.append(result)
        
        return results
    
    def _encode_heatmap(self, heatmap: np.ndarray) -> str:
        """Encode heatmap as base64 image"""
        # Convert to colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', heatmap_colored)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return heatmap_b64
    
    def fuse_multi_camera(self, detections_list: List[List[Dict]]) -> List[Dict]:
        """
        Fuse detections from multiple cameras
        
        Args:
            detections_list: List of detection results from different cameras
        
        Returns:
            Fused detection results
        """
        if not detections_list:
            return []
        
        # Group by component class
        component_groups = {}
        for detections in detections_list:
            for det in detections:
                comp = det['component']
                if comp not in component_groups:
                    component_groups[comp] = []
                component_groups[comp].append(det)
        
        # Fuse each group
        fused = []
        for comp, dets in component_groups.items():
            # Take detection with highest confidence
            best_det = max(dets, key=lambda x: x['yolo_conf'])
            
            # Average anomaly scores
            avg_score = np.mean([d['anomaly_score'] for d in dets])
            
            # Determine final status
            statuses = [d['status'] for d in dets]
            if 'DEFECTIVE' in statuses:
                final_status = 'DEFECTIVE'
            elif 'NORMAL' in statuses:
                final_status = 'NORMAL'
            else:
                final_status = 'UNKNOWN'
            
            fused.append({
                'component': comp,
                'bbox': best_det['bbox'],
                'yolo_conf': best_det['yolo_conf'],
                'anomaly_score': float(avg_score),
                'status': final_status,
                'heatmap': best_det['heatmap'],
                'num_views': len(dets)
            })
        
        return fused
