"""
Evaluation module for YOLO + PatchCore system
Computes metrics and generates evaluation reports
"""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import json
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class InspectionEvaluator:
    """Evaluator for industrial inspection system"""
    
    def __init__(self):
        self.results = []
        self.ground_truth = []
    
    def add_result(self, prediction: Dict, ground_truth: Dict):
        """Add a prediction result with ground truth"""
        self.results.append(prediction)
        self.ground_truth.append(ground_truth)
    
    def compute_detection_metrics(self) -> Dict:
        """Compute YOLO detection metrics"""
        if not self.results:
            return {}
        
        total_detections = sum(len(r.get('detections', [])) for r in self.results)
        total_gt = sum(len(gt.get('objects', [])) for gt in self.ground_truth)
        
        # Compute IoU-based metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, gt in zip(self.results, self.ground_truth):
            pred_boxes = [d['bbox'] for d in pred.get('detections', [])]
            gt_boxes = [obj['bbox'] for obj in gt.get('objects', [])]
            
            matched_gt = set()
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self._compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou > 0.5:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1
            
            false_negatives += len(gt_boxes) - len(matched_gt)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'total_predictions': total_detections,
            'total_ground_truth': total_gt,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def compute_anomaly_metrics(self, threshold: float = 0.5) -> Dict:
        """Compute PatchCore anomaly detection metrics"""
        if not self.results:
            return {}
        
        y_true = []
        y_scores = []
        
        for pred, gt in zip(self.results, self.ground_truth):
            for det in pred.get('detections', []):
                anomaly_score = det.get('anomaly_score', 0)
                is_defective = gt.get('is_defective', False)
                
                y_scores.append(anomaly_score)
                y_true.append(1 if is_defective else 0)
        
        if not y_true:
            return {}
        
        y_pred = [1 if score > threshold else 0 for score in y_scores]
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC AUC
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.0
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'auc_roc': auc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def compute_per_class_metrics(self) -> Dict:
        """Compute metrics per component class"""
        class_metrics = {}
        
        for pred, gt in zip(self.results, self.ground_truth):
            for det in pred.get('detections', []):
                comp = det['component']
                
                if comp not in class_metrics:
                    class_metrics[comp] = {
                        'count': 0,
                        'anomaly_scores': [],
                        'statuses': []
                    }
                
                class_metrics[comp]['count'] += 1
                class_metrics[comp]['anomaly_scores'].append(det.get('anomaly_score', 0))
                class_metrics[comp]['statuses'].append(det.get('status', 'UNKNOWN'))
        
        # Compute statistics
        for comp, metrics in class_metrics.items():
            scores = metrics['anomaly_scores']
            metrics['mean_anomaly'] = float(np.mean(scores)) if scores else 0
            metrics['std_anomaly'] = float(np.std(scores)) if scores else 0
            metrics['max_anomaly'] = float(np.max(scores)) if scores else 0
            metrics['min_anomaly'] = float(np.min(scores)) if scores else 0
            
            statuses = metrics['statuses']
            metrics['defective_count'] = statuses.count('DEFECTIVE')
            metrics['normal_count'] = statuses.count('NORMAL')
            metrics['defect_rate'] = metrics['defective_count'] / len(statuses) if statuses else 0
        
        return class_metrics
    
    def generate_report(self, output_path: str = 'evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        report = {
            'detection_metrics': self.compute_detection_metrics(),
            'anomaly_metrics': self.compute_anomaly_metrics(),
            'per_class_metrics': self.compute_per_class_metrics(),
            'total_samples': len(self.results)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to: {output_path}")
        return report
    
    def plot_roc_curve(self, output_path: str = 'roc_curve.png'):
        """Plot ROC curve for anomaly detection"""
        y_true = []
        y_scores = []
        
        for pred, gt in zip(self.results, self.ground_truth):
            for det in pred.get('detections', []):
                anomaly_score = det.get('anomaly_score', 0)
                is_defective = gt.get('is_defective', False)
                
                y_scores.append(anomaly_score)
                y_true.append(1 if is_defective else 0)
        
        if not y_true:
            return
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Anomaly Detection')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve saved to: {output_path}")
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def evaluate_model_on_dataset(inspector, dataset_path: str, output_dir: str = 'evaluation'):
    """
    Evaluate model on a dataset
    
    Args:
        inspector: IndustrialInspector instance
        dataset_path: Path to test dataset
        output_dir: Output directory for results
    """
    evaluator = InspectionEvaluator()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load test images
    test_images = list(Path(dataset_path).rglob('*.jpg'))
    
    print(f"Evaluating on {len(test_images)} images...")
    
    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inspection
        results = inspector.inspect(img_rgb)
        
        # Mock ground truth (in real scenario, load from annotations)
        ground_truth = {
            'objects': [],
            'is_defective': False
        }
        
        evaluator.add_result({'detections': results}, ground_truth)
    
    # Generate report
    report = evaluator.generate_report(str(output_path / 'report.json'))
    evaluator.plot_roc_curve(str(output_path / 'roc_curve.png'))
    
    print("\nEvaluation Summary:")
    print(f"  Detection Precision: {report['detection_metrics'].get('precision', 0):.3f}")
    print(f"  Detection Recall: {report['detection_metrics'].get('recall', 0):.3f}")
    print(f"  Anomaly AUC: {report['anomaly_metrics'].get('auc_roc', 0):.3f}")
    
    return report
