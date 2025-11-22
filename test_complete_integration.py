"""
Complete Integration Test
Tests: ThreeJS → Backend → AI → Database → Dashboard
"""
import requests
import time
import sys

API_BASE = "http://localhost:8000"

def print_step(step, message):
    print(f"\n{'='*70}")
    print(f"STEP {step}: {message}")
    print('='*70)

def test_backend_connection():
    """Test 1: Backend is running"""
    print_step(1, "Testing Backend Connection")
    try:
        response = requests.get(f"{API_BASE}/")
        print(f"✓ Backend is running")
        print(f"  Response: {response.json()['message']}")
        return True
    except Exception as e:
        print(f"✗ Backend not running: {e}")
        print("\n  Start backend first:")
        print("  py -3.12 run_server.py")
        return False

def test_ai_system():
    """Test 2: AI system is loaded"""
    print_step(2, "Testing AI System")
    try:
        response = requests.get(f"{API_BASE}/ai/status")
        data = response.json()
        print(f"✓ AI System Status:")
        print(f"  YOLO loaded: {data['yolo_loaded']}")
        print(f"  PatchCore models: {data['patchcore_models']}")
        print(f"  Available classes: {len(data.get('available_classes', []))}")
        return data['yolo_loaded']
    except Exception as e:
        print(f"✗ AI system check failed: {e}")
        return False

def test_database_connection():
    """Test 3: Database is connected"""
    print_step(3, "Testing Database Connection")
    try:
        response = requests.get(f"{API_BASE}/db/scans?limit=1")
        data = response.json()
        print(f"✓ Database connected")
        print(f"  Recent scans: {len(data.get('scans', []))}")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("\n  Check:")
        print("  1. Supabase credentials in .env")
        print("  2. Database schema created")
        print("  3. Run: py -3.12 test_supabase.py")
        return False

def test_scan_execution():
    """Test 4: Run a complete scan"""
    print_step(4, "Running Complete Scan")
    try:
        print("Starting scan...")
        response = requests.post(
            f"{API_BASE}/scan/start",
            json={
                "amr_x": 0,
                "camera_mode": "crop",
                "four_side_scan": False
            }
        )
        data = response.json()
        scan_id = data.get('scan_id')
        print(f"✓ Scan started")
        print(f"  Scan ID: {scan_id}")
        
        # Monitor progress
        print("\nMonitoring scan progress...")
        for i in range(15):
            time.sleep(2)
            try:
                status_response = requests.get(f"{API_BASE}/scan/status")
                status = status_response.json()
                state = status.get('state', 'unknown')
                print(f"  [{i+1}/15] State: {state}")
                
                if state == 'idle' or state == 'complete':
                    print("✓ Scan completed")
                    break
            except:
                pass
        
        return scan_id
    except Exception as e:
        print(f"✗ Scan failed: {e}")
        return None

def test_scan_results(scan_id):
    """Test 5: Verify scan results in database"""
    print_step(5, "Verifying Scan Results")
    if not scan_id:
        print("✗ No scan ID to verify")
        return False
    
    try:
        response = requests.get(f"{API_BASE}/db/scan/{scan_id}")
        data = response.json()
        
        scan = data.get('scan', {})
        frames = data.get('frames', [])
        detections = data.get('detections', [])
        
        print(f"✓ Scan results retrieved")
        print(f"\n  Scan Details:")
        print(f"    Status: {scan.get('status')}")
        print(f"    Start: {scan.get('start_time')}")
        print(f"    End: {scan.get('end_time')}")
        print(f"\n  Data Collected:")
        print(f"    Frames: {len(frames)}")
        print(f"    Detections: {len(detections)}")
        print(f"    Total Components: {scan.get('total_components', 0)}")
        print(f"    Defective: {scan.get('defective_count', 0)}")
        print(f"    Normal: {scan.get('normal_count', 0)}")
        
        if detections:
            print(f"\n  Sample Detections:")
            for det in detections[:3]:
                print(f"    - {det['component_name']}: "
                      f"YOLO={det['yolo_conf']:.2f}, "
                      f"Anomaly={det['patchcore_score']:.2f}, "
                      f"Status={det['patchcore_status']}")
        
        return len(detections) > 0
    except Exception as e:
        print(f"✗ Failed to verify results: {e}")
        return False

def test_dashboard_data():
    """Test 6: Dashboard data endpoints"""
    print_step(6, "Testing Dashboard Data")
    try:
        # Test stats
        response = requests.get(f"{API_BASE}/db/stats")
        data = response.json()
        
        component_stats = data.get('component_stats', [])
        print(f"✓ Dashboard data available")
        print(f"  Component stats: {len(component_stats)} components tracked")
        
        if component_stats:
            print(f"\n  Top Components:")
            for stat in component_stats[:5]:
                print(f"    - {stat['component_name']}: "
                      f"{stat['total_detected']} detected, "
                      f"{stat['total_defective']} defective")
        
        return True
    except Exception as e:
        print(f"✗ Dashboard data test failed: {e}")
        return False

def main():
    print("="*70)
    print("COMPLETE INTEGRATION TEST")
    print("ThreeJS → Backend → AI → Database → Dashboard")
    print("="*70)
    
    results = {}
    
    # Run all tests
    results['backend'] = test_backend_connection()
    if not results['backend']:
        print("\n✗ Cannot proceed without backend")
        sys.exit(1)
    
    results['ai'] = test_ai_system()
    results['database'] = test_database_connection()
    
    if results['database']:
        scan_id = test_scan_execution()
        results['scan'] = scan_id is not None
        
        if scan_id:
            results['results'] = test_scan_results(scan_id)
            results['dashboard'] = test_dashboard_data()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {test}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("✓ ALL SYSTEMS OPERATIONAL!")
        print("="*70)
        print("\nYour system is fully integrated:")
        print("  ✓ ThreeJS robot scanning")
        print("  ✓ AI detection working")
        print("  ✓ Database saving data")
        print("  ✓ Dashboard ready")
        print("\nOpen dashboard:")
        print("  py -3.12 launch_dashboard.py")
        print("="*70)
    else:
        print("\n⚠ Some tests failed. Check errors above.")

if __name__ == '__main__':
    main()
