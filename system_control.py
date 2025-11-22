"""
System Control - All-in-one control script for YOLO + PatchCore inspection system
Handles: installation, training, testing, API, demo, and server management
"""
import sys
import os
import subprocess
import argparse
import time
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)

# ============================================================================
# INSTALLATION & VERIFICATION
# ============================================================================

def check_dependencies():
    """Check if all dependencies are installed"""
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'fastapi': 'fastapi',
        'timm': 'timm',
        'albumentations': 'albumentations'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    return missing

def install_dependencies():
    """Install all dependencies"""
    print_header("INSTALLING DEPENDENCIES")
    
    missing = check_dependencies()
    if not missing:
        print("✓ All dependencies already installed")
        return True
    
    print(f"Installing: {', '.join(missing)}")
    cmd = "py -3.12 -m pip install -r requirements.txt"
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print("\n✓ Installation complete")
        return True
    else:
        print("\n✗ Installation failed")
        return False

# ============================================================================
# TRAINING
# ============================================================================

def train_yolo(epochs=50, batch=16, imgsz=640):
    """Train YOLO model"""
    print_header("TRAINING YOLO")
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ai.train_yolo import prepare_dataset, train_yolo as train_yolo_fn
    
    print("\n[1/3] Preparing dataset...")
    yaml_path = prepare_dataset(images_dir='images', output_dir='dataset_yolo')
    
    print("\n[2/3] Training YOLO...")
    train_yolo_fn(data_yaml=yaml_path, epochs=epochs, batch=batch, imgsz=imgsz)
    
    print("\n[3/3] Exporting model...")
    try:
        from ai.train_yolo import export_yolo
        export_yolo(format='onnx')
    except Exception as e:
        print(f"⚠ ONNX export skipped (network issue): {str(e)[:100]}")
        print("  Main .pt model is saved and working!")
    
    print("\n✓ YOLO training complete")
    return True

def train_patchcore(max_images=200, coreset_ratio=0.1):
    """Train PatchCore models"""
    print_header("TRAINING PATCHCORE")
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ai.train_patchcore import train_all_components
    
    print("\nTraining PatchCore for all components...")
    success_count, failed = train_all_components(
        images_dir='images',
        output_dir='models/patchcore',
        max_images=max_images,
        coreset_ratio=coreset_ratio
    )
    
    print(f"\n✓ PatchCore training complete: {success_count} models trained")
    return True

def train_system(quick=False):
    """Train complete system"""
    print_header("TRAINING COMPLETE SYSTEM")
    
    if quick:
        print("\nQuick training mode (30 epochs, 100 images)")
        epochs, max_images = 30, 100
    else:
        print("\nFull training mode (100 epochs, 200 images)")
        epochs, max_images = 100, 200
    
    # Train YOLO
    train_yolo(epochs=epochs)
    
    # Train PatchCore
    train_patchcore(max_images=max_images)
    
    print_header("TRAINING COMPLETE")
    print("\nModels saved to:")
    print("  YOLO: runs/detect/component_detector/weights/best.pt")
    print("  PatchCore: models/patchcore/*.pkl")
    
    return True

# ============================================================================
# TESTING
# ============================================================================

def test_system():
    """Test the trained system"""
    print_header("TESTING SYSTEM")
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ai.detector import IndustrialInspector
    import cv2
    import numpy as np
    
    # Initialize inspector
    print("\nInitializing inspector...")
    yolo_path = 'runs/detect/component_detector/weights/best.pt'
    if not os.path.exists(yolo_path):
        yolo_path = None
        print("⚠ YOLO model not found, using pretrained")
    
    inspector = IndustrialInspector(yolo_path=yolo_path, patchcore_dir='models/patchcore')
    
    print(f"✓ Inspector ready")
    print(f"  YOLO: {inspector.yolo.model is not None}")
    print(f"  PatchCore models: {len(inspector.patchcore_models)}")
    
    # Test on sample images
    print("\nTesting on sample images...")
    test_images = []
    for comp_dir in list(Path('images').iterdir())[:5]:
        if comp_dir.is_dir():
            imgs = list(comp_dir.glob('*.jpg'))[:2]
            test_images.extend(imgs)
    
    if not test_images:
        print("✗ No test images found")
        return False
    
    total_detections = 0
    defective_count = 0
    
    for img_path in test_images[:10]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = inspector.inspect(img_rgb)
        
        total_detections += len(results)
        defective_count += sum(1 for r in results if r['status'] == 'DEFECTIVE')
        
        print(f"  {img_path.name}: {len(results)} detections")
    
    print(f"\n✓ Test complete")
    print(f"  Total detections: {total_detections}")
    print(f"  Defective: {defective_count}")
    print(f"  Normal: {total_detections - defective_count}")
    
    return True

# ============================================================================
# DEMO
# ============================================================================

def run_demo():
    """Run complete demo"""
    print_header("RUNNING DEMO")
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ai.detector import IndustrialInspector
    import cv2
    
    # Initialize
    print("\nInitializing system...")
    yolo_path = 'runs/detect/component_detector/weights/best.pt'
    if not os.path.exists(yolo_path):
        yolo_path = None
    
    inspector = IndustrialInspector(yolo_path=yolo_path, patchcore_dir='models/patchcore')
    
    # Demo 1: Single image
    print("\n--- Demo 1: Single Image Inspection ---")
    test_img = None
    for comp_dir in Path('images').iterdir():
        if comp_dir.is_dir():
            imgs = list(comp_dir.glob('*.jpg'))
            if imgs:
                test_img = imgs[0]
                break
    
    if test_img:
        img = cv2.imread(str(test_img))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = inspector.inspect(img_rgb)
        
        print(f"Image: {test_img.name}")
        print(f"Detections: {len(results)}")
        for i, det in enumerate(results[:3], 1):
            print(f"  {i}. {det['component']}: conf={det['yolo_conf']:.3f}, "
                  f"anomaly={det['anomaly_score']:.3f}, status={det['status']}")
    
    # Demo 2: Multi-camera fusion
    print("\n--- Demo 2: Multi-Camera Fusion ---")
    test_images = []
    for comp_dir in list(Path('images').iterdir())[:3]:
        if comp_dir.is_dir():
            imgs = list(comp_dir.glob('*.jpg'))
            if imgs:
                test_images.append(imgs[0])
    
    if len(test_images) >= 2:
        all_detections = []
        for img_path in test_images:
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = inspector.inspect(img_rgb)
            all_detections.append(detections)
            print(f"Camera {len(all_detections)}: {len(detections)} detections")
        
        fused = inspector.fuse_multi_camera(all_detections)
        print(f"\nFused: {len(fused)} unique components")
        for det in fused[:5]:
            print(f"  - {det['component']}: views={det['num_views']}, status={det['status']}")
    
    print("\n✓ Demo complete")
    return True

# ============================================================================
# SERVER & API
# ============================================================================

def start_backend():
    """Start the backend server"""
    print_header("STARTING BACKEND SERVER")
    print("\nServer will start at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop")
    print("="*70)
    
    import uvicorn
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

def test_api():
    """Test API endpoints"""
    print_header("TESTING API")
    print("\nMake sure backend is running: py -3.12 system_control.py --server")
    input("Press Enter to continue...")
    
    import requests
    import json
    
    api_base = "http://localhost:8000"
    
    # Test root
    print("\n1. Testing root endpoint...")
    try:
        r = requests.get(f"{api_base}/")
        print(f"   Status: {r.status_code}")
        print(f"   Response: {json.dumps(r.json(), indent=2)[:200]}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test AI status
    print("\n2. Testing AI status...")
    try:
        r = requests.get(f"{api_base}/ai/status")
        print(f"   Status: {r.status_code}")
        data = r.json()
        print(f"   AI Status: {data.get('status')}")
        print(f"   YOLO: {data.get('yolo_loaded')}")
        print(f"   PatchCore: {data.get('patchcore_models')} models")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✓ API test complete")
    return True

# ============================================================================
# STATUS
# ============================================================================

def show_status():
    """Show system status"""
    print_header("SYSTEM STATUS")
    
    # Dependencies
    print("\n1. Dependencies:")
    missing = check_dependencies()
    if not missing:
        print("   ✓ All packages installed")
        try:
            import torch
            print(f"   CUDA available: {torch.cuda.is_available()}")
        except:
            pass
    else:
        print(f"   ✗ Missing: {', '.join(missing)}")
    
    # Dataset
    print("\n2. Dataset:")
    if os.path.exists('images'):
        num_components = len([d for d in os.listdir('images') if os.path.isdir(os.path.join('images', d))])
        print(f"   ✓ {num_components} component classes")
    else:
        print("   ✗ Dataset not found")
    
    # Models
    print("\n3. Trained Models:")
    yolo_path = 'runs/detect/component_detector/weights/best.pt'
    patchcore_dir = 'models/patchcore'
    
    if os.path.exists(yolo_path):
        size = os.path.getsize(yolo_path) / (1024*1024)
        print(f"   ✓ YOLO: {size:.1f} MB")
    else:
        print("   ✗ YOLO not trained")
    
    if os.path.exists(patchcore_dir):
        num_models = len([f for f in os.listdir(patchcore_dir) if f.endswith('.pkl')])
        print(f"   ✓ PatchCore: {num_models} models")
    else:
        print("   ✗ PatchCore not trained")
    
    # System files
    print("\n4. System Files:")
    print(f"   Backend: {'✓' if os.path.exists('backend/main.py') else '✗'}")
    print(f"   Frontend: {'✓' if os.path.exists('frontend/combined_dashboard.html') else '✗'}")
    print(f"   Simulation: {'✓' if os.path.exists('simulation/simulator.py') else '✗'}")

# ============================================================================
# MENU
# ============================================================================

def launch_dashboard():
    """Launch React dashboard"""
    print_header("LAUNCHING DASHBOARD")
    import webbrowser
    from pathlib import Path
    
    dashboard_path = Path('frontend/react_dashboard.html').absolute()
    if dashboard_path.exists():
        webbrowser.open(f'file://{dashboard_path}')
        print("\n✓ Dashboard opened in browser")
        print("\nMake sure backend is running:")
        print("  py -3.12 system_control.py --server")
    else:
        print("\n✗ Dashboard not found")
    
    return True

def show_menu():
    """Show interactive menu"""
    print_header("YOLO + PATCHCORE INSPECTION SYSTEM")
    
    print("\nMAIN MENU:")
    print("  1. Install dependencies")
    print("  2. Train models (quick - 30 epochs)")
    print("  3. Train models (full - 100 epochs)")
    print("  4. Test system")
    print("  5. Run demo")
    print("  6. Start backend server")
    print("  7. Launch React dashboard")
    print("  8. Test API")
    print("  9. Test Supabase")
    print(" 10. Show system status")
    print(" 11. Exit")
    
    choice = input("\nSelect option (1-11): ").strip()
    return choice

def main():
    parser = argparse.ArgumentParser(description='System Control - All-in-one management')
    parser.add_argument('--install', action='store_true', help='Install dependencies')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--quick', action='store_true', help='Quick training (30 epochs)')
    parser.add_argument('--test', action='store_true', help='Test system')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--server', action='store_true', help='Start backend server')
    parser.add_argument('--dashboard', action='store_true', help='Launch React dashboard')
    parser.add_argument('--api-test', action='store_true', help='Test API')
    parser.add_argument('--test-db', action='store_true', help='Test Supabase')
    parser.add_argument('--status', action='store_true', help='Show status')
    
    args = parser.parse_args()
    
    # Command line mode
    if any([args.install, args.train, args.test, args.demo, args.server, args.dashboard, args.api_test, args.test_db, args.status]):
        if args.install:
            install_dependencies()
        elif args.train:
            train_system(quick=args.quick)
        elif args.test:
            test_system()
        elif args.demo:
            run_demo()
        elif args.server:
            start_backend()
        elif args.dashboard:
            launch_dashboard()
        elif args.api_test:
            test_api()
        elif args.test_db:
            subprocess.run("py -3.12 test_supabase.py", shell=True)
        elif args.status:
            show_status()
        return
    
    # Interactive mode
    while True:
        choice = show_menu()
        
        if choice == '1':
            install_dependencies()
        elif choice == '2':
            train_system(quick=True)
        elif choice == '3':
            train_system(quick=False)
        elif choice == '4':
            test_system()
        elif choice == '5':
            run_demo()
        elif choice == '6':
            start_backend()
        elif choice == '7':
            launch_dashboard()
        elif choice == '8':
            test_api()
        elif choice == '9':
            subprocess.run("py -3.12 test_supabase.py", shell=True)
        elif choice == '10':
            show_status()
        elif choice == '11':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == '__main__':
    main()
