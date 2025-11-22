import random
import numpy as np
from . import config

class DefectDetector:
    """Mock AI defect detection system"""
    
    def __init__(self):
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.defect_types = config.DEFECT_TYPES
    
    def detect(self, image):
        """
        Run detection on a single image
        
        Args:
            image: numpy array (H, W, 3)
            
        Returns:
            list: Detection results [{"bbox": [x, y, w, h], "label": str, "confidence": float}]
        """
        if image is None:
            return []
        
        h, w = image.shape[:2]
        
        # Generate random number of detections (0-5)
        num_detections = random.randint(0, 5)
        
        detections = []
        for _ in range(num_detections):
            # Random bounding box
            x = random.randint(0, w - 100)
            y = random.randint(0, h - 100)
            box_w = random.randint(50, min(200, w - x))
            box_h = random.randint(50, min(200, h - y))
            
            # Random defect type and confidence
            label = random.choice(self.defect_types)
            confidence = random.uniform(self.confidence_threshold, 1.0)
            
            detections.append({
                "bbox": [x, y, box_w, box_h],
                "label": label,
                "confidence": round(confidence, 3)
            })
        
        return detections
    
    def fuse_detections(self, det_left, det_center, det_right):
        """
        Fuse detections from 3 cameras
        
        Args:
            det_left, det_center, det_right: Detection lists
            
        Returns:
            dict: Fused results with statistics
        """
        all_detections = []
        
        # Add camera source to each detection
        for det in det_left:
            det["camera"] = "left"
            all_detections.append(det)
        
        for det in det_center:
            det["camera"] = "center"
            all_detections.append(det)
        
        for det in det_right:
            det["camera"] = "right"
            all_detections.append(det)
        
        # Calculate statistics
        defect_counts = {}
        for det in all_detections:
            label = det["label"]
            defect_counts[label] = defect_counts.get(label, 0) + 1
        
        return {
            "total_detections": len(all_detections),
            "detections": all_detections,
            "defect_counts": defect_counts,
            "has_defects": len(all_detections) > 0
        }
