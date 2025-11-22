from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import base64
import cv2
import numpy as np
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simulator import AMRScanner
from simulation import config
from ai.detector import IndustrialInspector, ComponentClasses
from database import get_supabase_client

app = FastAPI(title="AMR Scanning Simulation API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global scanner instance
scanner = None
scan_running = False

# Global AI inspector
inspector = None

# Global Supabase client
supabase = None

class ScanRequest(BaseModel):
    amr_x: float = 0
    camera_mode: str = "crop"
    four_side_scan: bool = True

@app.on_event("startup")
async def startup_event():
    """Initialize scanner, AI inspector, and Supabase on startup"""
    global scanner, inspector, supabase
    scanner = AMRScanner(camera_mode="crop")
    
    # Initialize AI inspector
    yolo_path = 'runs/detect/component_detector/weights/best.pt'
    if not os.path.exists(yolo_path):
        yolo_path = None
    inspector = IndustrialInspector(yolo_path=yolo_path)
    
    # Initialize Supabase client
    try:
        supabase = get_supabase_client()
        print("✓ Supabase client initialized")
    except Exception as e:
        print(f"⚠ Supabase initialization failed: {e}")
        supabase = None

@app.get("/")
async def root():
    """API root"""
    return {
        "message": "AMR Scanning Simulation API",
        "version": "1.0",
        "endpoints": {
            "start_scan": "POST /scan/start",
            "get_status": "GET /scan/status",
            "get_results": "GET /scan/results",
            "get_config": "GET /config"
        }
    }

@app.post("/scan/start")
async def start_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    """Start a new scan and save to database"""
    global scanner, scan_running, supabase, inspector
    
    if scan_running:
        raise HTTPException(status_code=400, detail="Scan already in progress")
    
    # Create scan in database if Supabase is available
    scan_id = None
    if supabase:
        try:
            scan_id = supabase.create_scan(
                side='all' if request.four_side_scan else 'front',
                amr_position=request.amr_x,
                camera_mode=request.camera_mode
            )
        except Exception as e:
            print(f"Failed to create scan in DB: {e}")
    
    # Reinitialize scanner with requested mode
    scanner = AMRScanner(camera_mode=request.camera_mode)
    
    def run_scan():
        global scan_running
        scan_running = True
        try:
            scanner.start_scan(amr_x=request.amr_x, four_side_scan=request.four_side_scan)
            
            # Save to database if available
            if supabase and scan_id:
                results = scanner.get_results()
                for result in results:
                    images = result.get('images', {})
                    
                    for camera_name, image_path in images.items():
                        camera_id = camera_name[0].upper()
                        
                        try:
                            # Create frame
                            frame_id = supabase.create_frame(
                                scan_id=scan_id,
                                camera_id=camera_id,
                                height_level=result['height'],
                                side=str(result.get('side', 'front')),
                                image_path=image_path
                            )
                            
                            # Run AI detection first
                            if image_path and os.path.exists(image_path):
                                img = cv2.imread(image_path)
                                if img is not None:
                                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    ai_detections = inspector.inspect(img_rgb)
                                    
                                    if ai_detections:
                                        supabase.create_detections_batch(frame_id, scan_id, ai_detections)
                            
                            # Try to upload image (optional - will skip if buckets don't exist)
                            if image_path and os.path.exists(image_path):
                                try:
                                    image_url = supabase.upload_image_from_path(scan_id, image_path)
                                    supabase.update_frame_image_url(frame_id, image_url)
                                except Exception as upload_error:
                                    # Storage bucket not created - that's OK, we have the data
                                    pass
                        except Exception as e:
                            print(f"Error saving frame: {e}")
                
                # Complete scan
                supabase.complete_scan(scan_id)
        finally:
            scan_running = False
    
    background_tasks.add_task(run_scan)
    
    return {
        "message": "Scan started",
        "scan_id": scan_id,
        "amr_x": request.amr_x,
        "camera_mode": request.camera_mode,
        "four_side_scan": request.four_side_scan
    }

@app.get("/scan/status")
async def get_status():
    """Get current scan status"""
    global scanner, scan_running
    
    if scanner is None:
        return {"state": "idle", "message": "No scan initialized"}
    
    status = scanner.get_status()
    status["running"] = scan_running
    return status

@app.get("/scan/results")
async def get_results():
    """Get scan results"""
    global scanner
    
    if scanner is None:
        raise HTTPException(status_code=404, detail="No scan data available")
    
    results = scanner.get_results()
    
    if not results:
        raise HTTPException(status_code=404, detail="No results available yet")
    
    return {
        "scan_id": scanner.scan_id,
        "total_heights": len(results),
        "results": results
    }

@app.get("/config")
async def get_config():
    """Get simulation configuration"""
    return {
        "height_levels": config.HEIGHT_LEVELS,
        "amr_positions": config.AMR_POSITIONS,
        "assembly_height": config.ASSEMBLY_HEIGHT,
        "num_cameras": config.NUM_CAMERAS,
        "image_size": {
            "width": config.IMAGE_WIDTH,
            "height": config.IMAGE_HEIGHT
        },
        "defect_types": config.DEFECT_TYPES,
        "component_classes": len(ComponentClasses.CLASSES)
    }

@app.post("/scan/frame")
async def scan_frame(file: UploadFile = File(...)):
    """
    Scan a single frame with AI detection
    
    Args:
        file: Image file (JPEG/PNG)
    
    Returns:
        Detection results with anomaly scores
    """
    global inspector
    
    if inspector is None:
        raise HTTPException(status_code=503, detail="AI inspector not initialized")
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inspection
    results = inspector.inspect(image_rgb)
    
    return {
        "detections": results,
        "total_components": len(results),
        "defective_count": sum(1 for r in results if r['status'] == 'DEFECTIVE')
    }

@app.post("/scan/frame/base64")
async def scan_frame_base64(data: dict):
    """
    Scan a frame from base64 encoded image
    
    Args:
        data: {"image": "base64_string"}
    
    Returns:
        Detection results
    """
    global inspector
    
    if inspector is None:
        raise HTTPException(status_code=503, detail="AI inspector not initialized")
    
    # Decode base64 image
    try:
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    
    # Run inspection
    results = inspector.inspect(image_rgb)
    
    return {
        "detections": results,
        "total_components": len(results),
        "defective_count": sum(1 for r in results if r['status'] == 'DEFECTIVE')
    }

@app.get("/scan/latest")
async def get_latest_scan():
    """Get latest scan results with AI detections"""
    global scanner
    
    if scanner is None or not scanner.scan_results:
        raise HTTPException(status_code=404, detail="No scan data available")
    
    return {
        "scan_id": scanner.scan_id,
        "results": scanner.scan_results[-1] if scanner.scan_results else None
    }

@app.get("/scan/progress")
async def get_scan_progress():
    """Get current scan progress"""
    global scanner, scan_running
    
    if scanner is None:
        return {"progress": 0, "status": "idle"}
    
    status = scanner.get_status()
    total_scans = status['total_heights'] * status['total_sides']
    current_scan = (status['current_side'] - 1) * status['total_heights'] + status['current_height_idx']
    
    return {
        "progress": int((current_scan / total_scans) * 100) if total_scans > 0 else 0,
        "status": status['state'],
        "message": status['message'],
        "running": scan_running
    }

@app.get("/ai/status")
async def get_ai_status():
    """Get AI system status"""
    global inspector
    
    if inspector is None:
        return {"status": "not_initialized"}
    
    return {
        "status": "ready",
        "yolo_loaded": inspector.yolo.model is not None,
        "patchcore_models": len(inspector.patchcore_models),
        "available_classes": list(inspector.patchcore_models.keys())
    }

@app.post("/scan/multi-camera")
async def scan_multi_camera(data: dict):
    """
    Scan multiple camera views and fuse results
    
    Args:
        data: {"images": ["base64_1", "base64_2", ...]}
    
    Returns:
        Fused detection results
    """
    global inspector
    
    if inspector is None:
        raise HTTPException(status_code=503, detail="AI inspector not initialized")
    
    images_b64 = data.get('images', [])
    if not images_b64:
        raise HTTPException(status_code=400, detail="No images provided")
    
    # Process each camera view
    all_detections = []
    
    for i, img_b64 in enumerate(images_b64):
        try:
            # Decode image
            img_data = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inspection
            detections = inspector.inspect(image_rgb)
            all_detections.append(detections)
            
        except Exception as e:
            print(f"Error processing camera {i}: {e}")
            continue
    
    # Fuse detections
    fused = inspector.fuse_multi_camera(all_detections)
    
    return {
        "cameras_processed": len(all_detections),
        "fused_detections": fused,
        "total_components": len(fused),
        "defective_count": sum(1 for r in fused if r['status'] == 'DEFECTIVE')
    }

@app.post("/train/yolo")
async def train_yolo_endpoint(background_tasks: BackgroundTasks, epochs: int = 50):
    """Trigger YOLO training in background"""
    from ai.train_yolo import prepare_dataset, train_yolo
    
    def train():
        yaml_path = prepare_dataset(images_dir='images')
        train_yolo(data_yaml=yaml_path, epochs=epochs)
    
    background_tasks.add_task(train)
    return {"message": "YOLO training started", "epochs": epochs}

@app.post("/train/patchcore")
async def train_patchcore_endpoint(background_tasks: BackgroundTasks, max_images: int = 200):
    """Trigger PatchCore training in background"""
    from ai.train_patchcore import train_all_components
    
    def train():
        train_all_components(images_dir='images', max_images=max_images)
    
    background_tasks.add_task(train)
    return {"message": "PatchCore training started", "max_images": max_images}

# ============================================================================
# SUPABASE-INTEGRATED ENDPOINTS
# ============================================================================

@app.post("/scan/start/db")
async def start_scan_with_db(request: ScanRequest, background_tasks: BackgroundTasks):
    """Start scan and save to Supabase database"""
    global scanner, scan_running, supabase, inspector
    
    if scan_running:
        raise HTTPException(status_code=400, detail="Scan already in progress")
    
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not initialized")
    
    # Create scan in database
    scan_id = supabase.create_scan(
        side='all' if request.four_side_scan else 'front',
        amr_position=request.amr_x,
        camera_mode=request.camera_mode
    )
    
    # Reinitialize scanner
    scanner = AMRScanner(camera_mode=request.camera_mode)
    
    def run_scan_with_db():
        global scan_running
        scan_running = True
        try:
            # Run scan
            scanner.start_scan(amr_x=request.amr_x, four_side_scan=request.four_side_scan)
            results = scanner.get_results()
            
            # Save results to database
            for result in results:
                # Handle both old and new result formats
                images = result.get('images', {})
                detections = result.get('detections', [])
                
                # Process each camera (left, center, right)
                for camera_name, image_path in images.items():
                    camera_id = camera_name[0].upper()  # L, C, R
                    
                    # Create frame
                    frame_id = supabase.create_frame(
                        scan_id=scan_id,
                        camera_id=camera_id,
                        height_level=result['height'],
                        side=str(result.get('side', 'front')),
                        image_path=image_path
                    )
                    
                    # Upload image if exists
                    if image_path and os.path.exists(image_path):
                        try:
                            image_url = supabase.upload_image_from_path(scan_id, image_path)
                            supabase.update_frame_image_url(frame_id, image_url)
                        except Exception as e:
                            print(f"Failed to upload image: {e}")
                    
                    # Run AI detection on the image
                    if image_path and os.path.exists(image_path):
                        try:
                            img = cv2.imread(image_path)
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                ai_detections = inspector.inspect(img_rgb)
                                
                                # Save detections to database
                                if ai_detections:
                                    supabase.create_detections_batch(frame_id, scan_id, ai_detections)
                        except Exception as e:
                            print(f"Failed to process detections: {e}")
            
            # Mark scan as completed
            supabase.complete_scan(scan_id)
            
        finally:
            scan_running = False
    
    background_tasks.add_task(run_scan_with_db)
    
    return {
        "message": "Scan started with database logging",
        "scan_id": scan_id,
        "amr_x": request.amr_x,
        "camera_mode": request.camera_mode
    }

@app.post("/scan/frame/db")
async def scan_frame_with_db(file: UploadFile = File(...)):
    """Scan frame and save to database"""
    global inspector, supabase
    
    if not inspector:
        raise HTTPException(status_code=503, detail="AI inspector not initialized")
    
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not initialized")
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create scan
    scan_id = supabase.create_scan(side='single', camera_mode='upload')
    
    # Upload image
    image_url = supabase.upload_image(scan_id, contents, file.filename)
    
    # Create frame
    frame_id = supabase.create_frame(
        scan_id=scan_id,
        camera_id='upload',
        height_level=0.0,
        side='single'
    )
    supabase.update_frame_image_url(frame_id, image_url)
    
    # Run inspection
    detections = inspector.inspect(image_rgb)
    
    # Save detections
    supabase.create_detections_batch(frame_id, scan_id, detections)
    
    # Complete scan
    supabase.complete_scan(scan_id)
    
    return {
        "scan_id": scan_id,
        "frame_id": frame_id,
        "image_url": image_url,
        "detections": detections,
        "total_components": len(detections),
        "defective_count": sum(1 for r in detections if r['status'] == 'DEFECTIVE')
    }

@app.get("/db/scans")
async def get_scans(limit: int = 10):
    """Get recent scans from database"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not initialized")
    
    scans = supabase.get_recent_scans(limit=limit)
    return {"scans": scans}

@app.get("/db/scan/{scan_id}")
async def get_scan_details(scan_id: str):
    """Get scan details with all detections"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not initialized")
    
    summary = supabase.get_scan_summary(scan_id)
    return summary

@app.get("/db/stats")
async def get_statistics():
    """Get component statistics"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not initialized")
    
    component_stats = supabase.get_component_stats()
    detection_summary = supabase.get_detection_summary()
    
    return {
        "component_stats": component_stats,
        "detection_summary": detection_summary
    }

@app.get("/db/defects/{scan_id}")
async def get_defects(scan_id: str):
    """Get all defective detections for a scan"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not initialized")
    
    defects = supabase.get_defective_detections(scan_id)
    return {"defects": defects, "count": len(defects)}

@app.get("/debug/status")
async def debug_status():
    """Debug endpoint to check system status"""
    global scanner, inspector, supabase, scan_running
    
    status = {
        "scanner_initialized": scanner is not None,
        "inspector_initialized": inspector is not None,
        "supabase_initialized": supabase is not None,
        "scan_running": scan_running,
        "yolo_loaded": inspector.yolo.model is not None if inspector else False,
        "patchcore_models": len(inspector.patchcore_models) if inspector else 0
    }
    
    if supabase:
        try:
            scans = supabase.get_recent_scans(limit=1)
            status["latest_scan"] = scans[0] if scans else None
        except Exception as e:
            status["supabase_error"] = str(e)
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
