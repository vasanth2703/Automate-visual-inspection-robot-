"""
Standalone test script for AMR scanning simulation
Run with: py -3.12 test_simulation.py
"""

from simulation.simulator import AMRScanner
from simulation import config
import json

def main():
    print("=" * 60)
    print("AMR SCANNING SIMULATION TEST")
    print("=" * 60)
    print()
    
    print("Configuration:")
    print(f"  Height levels: {config.HEIGHT_LEVELS}")
    print(f"  Assembly height: {config.ASSEMBLY_HEIGHT}m")
    print(f"  Number of cameras: {config.NUM_CAMERAS}")
    print(f"  Camera mode: {config.CAMERA_MODE}")
    print()
    
    # Test with crop mode
    print("Starting scan with CROP mode...")
    print("-" * 60)
    scanner = AMRScanner(camera_mode="crop")
    results = scanner.start_scan(amr_x=0)
    
    print()
    print("=" * 60)
    print("SCAN RESULTS SUMMARY")
    print("=" * 60)
    print(f"Scan ID: {results['scan_id']}")
    print(f"AMR Position: {results['amr_x']}m")
    print(f"Heights Scanned: {results['total_heights']}")
    print(f"Total Detections: {results['total_detections']}")
    print(f"Has Defects: {results['has_defects']}")
    print()
    
    print("Results by height:")
    for height_result in results['heights_scanned']:
        h = height_result['height']
        det = height_result['detections']
        print(f"  Height {h:.1f}m: {det['total_detections']} detections")
        if det['defect_counts']:
            for defect_type, count in det['defect_counts'].items():
                print(f"    - {defect_type}: {count}")
    
    print()
    print(f"Results saved to: data/results/scan_{results['scan_id']}_summary.json")
    print()
    
    # Test with random mode
    print("=" * 60)
    print("Starting scan with RANDOM mode...")
    print("-" * 60)
    scanner2 = AMRScanner(camera_mode="random")
    results2 = scanner2.start_scan(amr_x=1)
    
    print()
    print(f"Scan ID: {results2['scan_id']}")
    print(f"Total Detections: {results2['total_detections']}")
    print()
    
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
