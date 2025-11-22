"""
Test Supabase integration
Run this to verify database connection and operations
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_supabase_client
import cv2
import numpy as np

def test_connection():
    """Test Supabase connection"""
    print("="*60)
    print("TESTING SUPABASE CONNECTION")
    print("="*60)
    
    try:
        client = get_supabase_client()
        print("✓ Supabase client initialized")
        print(f"  URL: {client.url}")
        return client
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return None

def test_scan_operations(client):
    """Test scan CRUD operations"""
    print("\n" + "="*60)
    print("TESTING SCAN OPERATIONS")
    print("="*60)
    
    # Create scan
    print("\n1. Creating scan...")
    scan_id = client.create_scan(side='front', amr_position=0.5)
    print(f"   Scan ID: {scan_id}")
    
    # Get scan
    print("\n2. Getting scan...")
    scan = client.get_scan(scan_id)
    print(f"   Status: {scan['status']}")
    print(f"   Start time: {scan['start_time']}")
    
    # Complete scan
    print("\n3. Completing scan...")
    client.complete_scan(scan_id)
    scan = client.get_scan(scan_id)
    print(f"   Status: {scan['status']}")
    print(f"   End time: {scan['end_time']}")
    
    return scan_id

def test_frame_operations(client, scan_id):
    """Test frame operations"""
    print("\n" + "="*60)
    print("TESTING FRAME OPERATIONS")
    print("="*60)
    
    # Create frame
    print("\n1. Creating frame...")
    frame_id = client.create_frame(
        scan_id=scan_id,
        camera_id='C',
        height_level=1.5,
        side='front'
    )
    print(f"   Frame ID: {frame_id}")
    
    return frame_id

def test_detection_operations(client, scan_id, frame_id):
    """Test detection operations"""
    print("\n" + "="*60)
    print("TESTING DETECTION OPERATIONS")
    print("="*60)
    
    # Create detection
    print("\n1. Creating detection...")
    detection = {
        'component': 'microchip',
        'bbox': [100, 100, 200, 200],
        'yolo_conf': 0.95,
        'status': 'NORMAL',
        'anomaly_score': 0.23
    }
    
    detection_id = client.create_detection(frame_id, scan_id, detection)
    print(f"   Detection ID: {detection_id}")
    
    # Get detections
    print("\n2. Getting detections...")
    detections = client.get_detections_by_scan(scan_id)
    print(f"   Total detections: {len(detections)}")
    
    return detection_id

def test_storage_operations(client, scan_id):
    """Test storage operations"""
    print("\n" + "="*60)
    print("TESTING STORAGE OPERATIONS")
    print("="*60)
    
    # Create test image
    print("\n1. Creating test image...")
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', test_img)
    img_bytes = buffer.tobytes()
    
    # Upload image
    print("\n2. Uploading image...")
    try:
        url = client.upload_image(scan_id, img_bytes, 'test_image.jpg')
        print(f"   ✓ Image uploaded")
        print(f"   URL: {url[:80]}...")
    except Exception as e:
        print(f"   ✗ Upload failed: {e}")
        print("   Note: Make sure storage buckets are created in Supabase")

def test_statistics(client):
    """Test statistics queries"""
    print("\n" + "="*60)
    print("TESTING STATISTICS")
    print("="*60)
    
    # Get component stats
    print("\n1. Getting component stats...")
    stats = client.get_component_stats()
    print(f"   Components tracked: {len(stats)}")
    if stats:
        print(f"   Sample: {stats[0]['component_name']} - {stats[0]['total_detected']} detections")
    
    # Get detection summary
    print("\n2. Getting detection summary...")
    summary = client.get_detection_summary()
    print(f"   Components: {len(summary)}")

def main():
    """Run all tests"""
    print("="*60)
    print("SUPABASE INTEGRATION TEST SUITE")
    print("="*60)
    
    # Test connection
    client = test_connection()
    if not client:
        print("\n✗ Cannot proceed without connection")
        return
    
    # Test operations
    scan_id = test_scan_operations(client)
    frame_id = test_frame_operations(client, scan_id)
    detection_id = test_detection_operations(client, scan_id, frame_id)
    test_storage_operations(client, scan_id)
    test_statistics(client)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"\n✓ All tests completed!")
    print(f"\nCreated:")
    print(f"  Scan ID: {scan_id}")
    print(f"  Frame ID: {frame_id}")
    print(f"  Detection ID: {detection_id}")
    print(f"\nYou can view these in Supabase dashboard:")
    print(f"  {client.url}")
    
    # Cleanup option
    print("\n" + "="*60)
    cleanup = input("Delete test data? (y/n): ").strip().lower()
    if cleanup == 'y':
        client.delete_scan(scan_id)
        print("✓ Test data deleted")

if __name__ == '__main__':
    main()
