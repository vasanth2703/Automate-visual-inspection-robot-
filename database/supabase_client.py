"""
Supabase client for industrial inspection system
Handles database operations and storage
"""
import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import base64
from dotenv import load_dotenv
from supabase import create_client, Client
import json

# Load environment variables
load_dotenv()

class SupabaseClient:
    """Supabase database and storage client"""
    
    def __init__(self):
        """Initialize Supabase client"""
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_KEY')
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        
        self.client: Client = create_client(self.url, self.key)
        self.storage = self.client.storage
    
    # ========================================================================
    # SCAN OPERATIONS
    # ========================================================================
    
    def create_scan(self, side: str = 'all', amr_position: float = 0.0, 
                    camera_mode: str = 'crop', metadata: dict = None) -> str:
        """
        Create a new scan session
        
        Returns:
            scan_id (str)
        """
        data = {
            'side': side,
            'amr_position': amr_position,
            'camera_mode': camera_mode,
            'status': 'running',
            'metadata': metadata or {}
        }
        
        result = self.client.table('scans').insert(data).execute()
        scan_id = result.data[0]['scan_id']
        print(f"✓ Created scan: {scan_id}")
        return scan_id
    
    def complete_scan(self, scan_id: str):
        """Mark scan as completed"""
        self.client.table('scans').update({
            'end_time': datetime.now().isoformat(),
            'status': 'completed'
        }).eq('scan_id', scan_id).execute()
        
        print(f"✓ Scan completed: {scan_id}")
    
    def get_scan(self, scan_id: str) -> Dict:
        """Get scan details"""
        result = self.client.table('scans').select('*').eq('scan_id', scan_id).execute()
        return result.data[0] if result.data else None
    
    def get_recent_scans(self, limit: int = 10) -> List[Dict]:
        """Get recent scans"""
        result = self.client.table('recent_scans').select('*').limit(limit).execute()
        return result.data
    
    # ========================================================================
    # FRAME OPERATIONS
    # ========================================================================
    
    def create_frame(self, scan_id: str, camera_id: str, height_level: float,
                     side: str, image_path: str = None, metadata: dict = None) -> str:
        """
        Create a frame record
        
        Returns:
            frame_id (str)
        """
        data = {
            'scan_id': scan_id,
            'camera_id': camera_id,
            'height_level': height_level,
            'side': side,
            'image_path': image_path,
            'metadata': metadata or {}
        }
        
        result = self.client.table('frames').insert(data).execute()
        frame_id = result.data[0]['frame_id']
        return frame_id
    
    def update_frame_image_url(self, frame_id: str, image_url: str):
        """Update frame with image URL"""
        self.client.table('frames').update({
            'image_url': image_url
        }).eq('frame_id', frame_id).execute()
    
    def get_frames_by_scan(self, scan_id: str) -> List[Dict]:
        """Get all frames for a scan"""
        result = self.client.table('frames').select('*').eq('scan_id', scan_id).execute()
        return result.data
    
    # ========================================================================
    # DETECTION OPERATIONS
    # ========================================================================
    
    def create_detection(self, frame_id: str, scan_id: str, detection: Dict) -> str:
        """
        Create a detection record
        
        Args:
            frame_id: Frame ID
            scan_id: Scan ID
            detection: Detection dict with keys:
                - component_name
                - bbox (list)
                - yolo_conf
                - patchcore_status
                - patchcore_score
                - heatmap_url (optional)
                - crop_url (optional)
        
        Returns:
            detection_id (str)
        """
        data = {
            'frame_id': frame_id,
            'scan_id': scan_id,
            'component_name': detection['component'],
            'bbox': json.dumps(detection['bbox']),
            'yolo_conf': detection['yolo_conf'],
            'patchcore_status': detection['status'],
            'patchcore_score': detection['anomaly_score'],
            'heatmap_url': detection.get('heatmap_url'),
            'crop_url': detection.get('crop_url')
        }
        
        result = self.client.table('detections').insert(data).execute()
        detection_id = result.data[0]['detection_id']
        return detection_id
    
    def create_detections_batch(self, frame_id: str, scan_id: str, 
                                detections: List[Dict]) -> List[str]:
        """Create multiple detections at once"""
        data = []
        for det in detections:
            data.append({
                'frame_id': frame_id,
                'scan_id': scan_id,
                'component_name': det['component'],
                'bbox': json.dumps(det['bbox']),
                'yolo_conf': det['yolo_conf'],
                'patchcore_status': det['status'],
                'patchcore_score': det['anomaly_score'],
                'heatmap_url': det.get('heatmap_url'),
                'crop_url': det.get('crop_url')
            })
        
        result = self.client.table('detections').insert(data).execute()
        return [d['detection_id'] for d in result.data]
    
    def get_detections_by_scan(self, scan_id: str) -> List[Dict]:
        """Get all detections for a scan"""
        result = self.client.table('detections').select('*').eq('scan_id', scan_id).execute()
        return result.data
    
    def get_defective_detections(self, scan_id: str) -> List[Dict]:
        """Get only defective detections"""
        result = self.client.table('detections').select('*')\
            .eq('scan_id', scan_id)\
            .eq('patchcore_status', 'DEFECTIVE')\
            .execute()
        return result.data
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_component_stats(self) -> List[Dict]:
        """Get component statistics"""
        result = self.client.table('component_stats').select('*')\
            .order('total_detected', desc=True)\
            .execute()
        return result.data
    
    def get_detection_summary(self) -> List[Dict]:
        """Get detection summary by component"""
        result = self.client.table('detection_summary').select('*').execute()
        return result.data
    
    # ========================================================================
    # STORAGE OPERATIONS
    # ========================================================================
    
    def upload_image(self, scan_id: str, image_data: bytes, 
                     filename: str, bucket: str = 'scans') -> str:
        """
        Upload image to Supabase storage
        
        Returns:
            Public URL of uploaded image
        """
        path = f"{scan_id}/{filename}"
        
        # Upload to storage
        self.storage.from_(bucket).upload(
            path=path,
            file=image_data,
            file_options={"content-type": "image/jpeg"}
        )
        
        # Get public URL
        url = self.storage.from_(bucket).get_public_url(path)
        return url
    
    def upload_heatmap(self, scan_id: str, heatmap_data: bytes, 
                      filename: str) -> str:
        """Upload heatmap to storage"""
        return self.upload_image(scan_id, heatmap_data, filename, bucket='heatmaps')
    
    def upload_crop(self, scan_id: str, crop_data: bytes, 
                   filename: str) -> str:
        """Upload component crop to storage"""
        return self.upload_image(scan_id, crop_data, filename, bucket='crops')
    
    def upload_image_from_path(self, scan_id: str, image_path: str, 
                               bucket: str = 'scans') -> str:
        """Upload image from file path"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        filename = Path(image_path).name
        return self.upload_image(scan_id, image_data, filename, bucket)
    
    def upload_base64_image(self, scan_id: str, base64_str: str, 
                           filename: str, bucket: str = 'scans') -> str:
        """Upload base64 encoded image"""
        image_data = base64.b64decode(base64_str)
        return self.upload_image(scan_id, image_data, filename, bucket)
    
    # ========================================================================
    # REALTIME SUBSCRIPTIONS
    # ========================================================================
    
    def subscribe_to_scans(self, callback):
        """Subscribe to scan updates"""
        return self.client.table('scans').on('*', callback).subscribe()
    
    def subscribe_to_detections(self, scan_id: str, callback):
        """Subscribe to detection updates for a specific scan"""
        return self.client.table('detections')\
            .on('INSERT', callback)\
            .eq('scan_id', scan_id)\
            .subscribe()
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def delete_scan(self, scan_id: str):
        """Delete scan and all related data (cascade)"""
        self.client.table('scans').delete().eq('scan_id', scan_id).execute()
        print(f"✓ Deleted scan: {scan_id}")
    
    def get_scan_summary(self, scan_id: str) -> Dict:
        """Get complete scan summary with all data"""
        scan = self.get_scan(scan_id)
        frames = self.get_frames_by_scan(scan_id)
        detections = self.get_detections_by_scan(scan_id)
        
        return {
            'scan': scan,
            'frames': frames,
            'detections': detections,
            'summary': {
                'total_frames': len(frames),
                'total_detections': len(detections),
                'defective_count': sum(1 for d in detections if d['patchcore_status'] == 'DEFECTIVE'),
                'normal_count': sum(1 for d in detections if d['patchcore_status'] == 'NORMAL')
            }
        }


# Singleton instance
_supabase_client = None

def get_supabase_client() -> SupabaseClient:
    """Get or create Supabase client singleton"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = SupabaseClient()
    return _supabase_client
