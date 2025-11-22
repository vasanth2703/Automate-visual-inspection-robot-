import time
import json
import os
from datetime import datetime
from pathlib import Path
from enum import Enum
from . import config
from .camera_sim import CameraSimulator
from .detection_stub import DefectDetector

class ScanState(Enum):
    """State machine states"""
    IDLE = "idle"
    MOVE_TO_SIDE = "move_to_side"
    MOVE_TO_HEIGHT = "move_to_height"
    STABILIZE = "stabilize"
    CAPTURE = "capture"
    RUN_AI = "run_ai"
    SAVE_RESULT = "save_result"
    NEXT_HEIGHT = "next_height"
    NEXT_SIDE = "next_side"
    COMPLETE = "complete"

class AMRScanner:
    """Main AMR scanning simulation with state machine"""
    
    def __init__(self, camera_mode="crop"):
        self.state = ScanState.IDLE
        self.camera_sim = CameraSimulator(mode=camera_mode)
        self.detector = DefectDetector()
        
        # Scan state
        self.current_side = 1
        self.current_height_idx = 0
        self.current_amr_x = 0
        self.scan_id = None
        self.scan_results = []
        self.status_message = "Ready"
        
        # Callbacks for UI updates
        self.status_callback = None
        self.progress_callback = None
        self.side_callback = None
    
    def set_status_callback(self, callback):
        """Set callback for status updates"""
        self.status_callback = callback
    
    def set_progress_callback(self, callback):
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def set_side_callback(self, callback):
        """Set callback for side updates"""
        self.side_callback = callback
    
    def _update_status(self, message):
        """Update status message"""
        self.status_message = message
        print(f"[{self.state.value}] {message}")
        if self.status_callback:
            self.status_callback(message)
    
    def _update_progress(self, current, total):
        """Update progress"""
        if self.progress_callback:
            self.progress_callback(current, total)
    
    def start_scan(self, amr_x=0, four_side_scan=True):
        """
        Start a complete scanning cycle
        
        Args:
            amr_x: AMR horizontal position in meters (ignored if four_side_scan=True)
            four_side_scan: If True, scan all 4 sides of the target
            
        Returns:
            dict: Scan results summary
        """
        self.scan_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_amr_x = amr_x
        self.current_height_idx = 0
        self.current_side = 1
        self.scan_results = []
        
        if four_side_scan:
            self._update_status(f"Starting 4-side scan {self.scan_id}")
            return self._scan_four_sides()
        else:
            self._update_status(f"Starting single-side scan {self.scan_id} at position {amr_x}m")
            return self._scan_single_side(amr_x)
    
    def _scan_four_sides(self):
        """Scan all 4 sides of the inspection target"""
        total_scans = len(config.SIDES) * len(config.HEIGHT_LEVELS)
        scan_count = 0
        
        for side_num in sorted(config.SIDES.keys()):
            side_info = config.SIDES[side_num]
            self.current_side = side_num
            
            # State: MOVE_TO_SIDE
            self.state = ScanState.MOVE_TO_SIDE
            self._update_status(f"Moving to {side_info['name']} side (Side {side_num})...")
            if self.side_callback:
                self.side_callback(side_num, side_info['name'])
            time.sleep(config.MOVEMENT_DELAY * 2)  # Longer delay for side movement
            
            # Scan all heights for this side
            for idx, height in enumerate(config.HEIGHT_LEVELS):
                self._update_progress(scan_count, total_scans)
                
                # State: MOVE_TO_HEIGHT
                self.state = ScanState.MOVE_TO_HEIGHT
                self._update_status(f"Side {side_num} - Moving to height {height:.2f}m")
                time.sleep(config.MOVEMENT_DELAY)
                
                # State: STABILIZE
                self.state = ScanState.STABILIZE
                self._update_status(f"Side {side_num} - Stabilizing...")
                time.sleep(config.STABILIZATION_DELAY)
                
                # State: CAPTURE
                self.state = ScanState.CAPTURE
                self._update_status(f"Side {side_num} - Capturing images from 3 cameras...")
                left, center, right = self.camera_sim.capture_at_height(height, side_num)
                time.sleep(config.CAPTURE_DELAY)
                
                # State: RUN_AI
                self.state = ScanState.RUN_AI
                self._update_status(f"Side {side_num} - Running AI detection...")
                det_left = self.detector.detect(left)
                det_center = self.detector.detect(center)
                det_right = self.detector.detect(right)
                
                fused = self.detector.fuse_detections(det_left, det_center, det_right)
                
                # State: SAVE_RESULT
                self.state = ScanState.SAVE_RESULT
                self._update_status(f"Side {side_num} - Saving results...")
                image_paths = self.camera_sim.save_images(left, center, right, height, 
                                                         f"{self.scan_id}_side{side_num}")
                
                result = {
                    "scan_id": self.scan_id,
                    "side": side_num,
                    "side_name": side_info['name'],
                    "height": height,
                    "timestamp": datetime.now().isoformat(),
                    "images": image_paths,
                    "detections": fused
                }
                
                self.scan_results.append(result)
                
                # State: NEXT_HEIGHT
                self.state = ScanState.NEXT_HEIGHT
                self.current_height_idx = idx + 1
                scan_count += 1
            
            # State: NEXT_SIDE
            self.state = ScanState.NEXT_SIDE
            self._update_status(f"Side {side_num} complete. Moving to next side...")
            self.current_height_idx = 0
        
        # State: COMPLETE
        self.state = ScanState.COMPLETE
        self._update_progress(total_scans, total_scans)
        self._update_status("4-side scan complete!")
        
        # Save summary
        summary = self._save_scan_summary()
        
        self.state = ScanState.IDLE
        return summary
    
    def _scan_single_side(self, amr_x):
        """Scan a single side (legacy mode)"""
        total_heights = len(config.HEIGHT_LEVELS)
        
        for idx, height in enumerate(config.HEIGHT_LEVELS):
            self._update_progress(idx, total_heights)
            
            # State: MOVE_TO_HEIGHT
            self.state = ScanState.MOVE_TO_HEIGHT
            self._update_status(f"Moving to height {height:.2f}m")
            time.sleep(config.MOVEMENT_DELAY)
            
            # State: STABILIZE
            self.state = ScanState.STABILIZE
            self._update_status("Stabilizing...")
            time.sleep(config.STABILIZATION_DELAY)
            
            # State: CAPTURE
            self.state = ScanState.CAPTURE
            self._update_status("Capturing images from 3 cameras...")
            left, center, right = self.camera_sim.capture_at_height(height, amr_x)
            time.sleep(config.CAPTURE_DELAY)
            
            # State: RUN_AI
            self.state = ScanState.RUN_AI
            self._update_status("Running AI detection...")
            det_left = self.detector.detect(left)
            det_center = self.detector.detect(center)
            det_right = self.detector.detect(right)
            
            fused = self.detector.fuse_detections(det_left, det_center, det_right)
            
            # State: SAVE_RESULT
            self.state = ScanState.SAVE_RESULT
            self._update_status("Saving results...")
            image_paths = self.camera_sim.save_images(left, center, right, height, self.scan_id)
            
            result = {
                "scan_id": self.scan_id,
                "side": 1,
                "side_name": "Front",
                "height": height,
                "amr_x": amr_x,
                "timestamp": datetime.now().isoformat(),
                "images": image_paths,
                "detections": fused
            }
            
            self.scan_results.append(result)
            
            # State: NEXT_HEIGHT
            self.state = ScanState.NEXT_HEIGHT
            self.current_height_idx = idx + 1
        
        # State: COMPLETE
        self.state = ScanState.COMPLETE
        self._update_progress(total_heights, total_heights)
        self._update_status("Scan complete!")
        
        # Save summary
        summary = self._save_scan_summary()
        
        self.state = ScanState.IDLE
        return summary
    
    def _save_scan_summary(self):
        """Save scan results to JSON"""
        Path(config.RESULTS_PATH).mkdir(parents=True, exist_ok=True)
        
        # Group results by side
        sides_data = {}
        for result in self.scan_results:
            side = result.get("side", 1)
            if side not in sides_data:
                sides_data[side] = {
                    "side_name": result.get("side_name", "Unknown"),
                    "heights": [],
                    "total_detections": 0
                }
            sides_data[side]["heights"].append(result)
            sides_data[side]["total_detections"] += result["detections"]["total_detections"]
        
        summary = {
            "scan_id": self.scan_id,
            "scan_type": "4-side" if len(sides_data) > 1 else "single-side",
            "total_sides": len(sides_data),
            "total_heights": len(config.HEIGHT_LEVELS),
            "total_scans": len(self.scan_results),
            "sides": sides_data,
            "all_results": self.scan_results,
            "total_detections": sum(r["detections"]["total_detections"] for r in self.scan_results),
            "has_defects": any(r["detections"]["has_defects"] for r in self.scan_results)
        }
        
        summary_path = os.path.join(config.RESULTS_PATH, f"scan_{self.scan_id}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def get_status(self):
        """Get current scan status"""
        return {
            "state": self.state.value,
            "message": self.status_message,
            "current_side": self.current_side,
            "current_height_idx": self.current_height_idx,
            "total_heights": len(config.HEIGHT_LEVELS),
            "total_sides": len(config.SIDES),
            "scan_id": self.scan_id
        }
    
    def get_results(self):
        """Get scan results"""
        return self.scan_results

# Convenience function
def run_simulation(amr_x=0, camera_mode="crop", four_side_scan=True):
    """Run a complete simulation"""
    scanner = AMRScanner(camera_mode=camera_mode)
    return scanner.start_scan(amr_x=amr_x, four_side_scan=four_side_scan)
