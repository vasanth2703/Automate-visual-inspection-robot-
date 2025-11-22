"""
Test 4-Side Scanning System
Run with: py -3.12 test_4side_scan.py
"""

from simulation.simulator import AMRScanner
from simulation import config

def main():
    print("=" * 70)
    print("AMR 4-SIDE INDUSTRIAL INSPECTION TEST")
    print("=" * 70)
    print()
    
    print("Configuration:")
    print(f"  Sides to scan: {len(config.SIDES)}")
    for side_num, side_info in config.SIDES.items():
        print(f"    Side {side_num}: {side_info['name']}")
    print(f"  Heights per side: {len(config.HEIGHT_LEVELS)}")
    print(f"  Total scans: {len(config.SIDES) * len(config.HEIGHT_LEVELS)}")
    print()
    
    input("Press ENTER to start 4-side scan...")
    
    print()
    print("-" * 70)
    print("STARTING 4-SIDE SCAN")
    print("-" * 70)
    print()
    
    scanner = AMRScanner(camera_mode="crop")
    
    # Add callbacks for better visualization
    def status_callback(msg):
        print(f"  {msg}")
    
    def side_callback(side_num, side_name):
        print()
        print(f"{'='*70}")
        print(f"  SCANNING SIDE {side_num}: {side_name.upper()}")
        print(f"{'='*70}")
    
    scanner.set_status_callback(status_callback)
    scanner.set_side_callback(side_callback)
    
    # Run 4-side scan
    results = scanner.start_scan(amr_x=0, four_side_scan=True)
    
    print()
    print("=" * 70)
    print("4-SIDE SCAN RESULTS")
    print("=" * 70)
    print()
    
    print(f"Scan ID: {results['scan_id']}")
    print(f"Scan Type: {results['scan_type']}")
    print(f"Total Sides: {results['total_sides']}")
    print(f"Total Scans: {results['total_scans']}")
    print(f"Total Detections: {results['total_detections']}")
    print(f"Has Defects: {results['has_defects']}")
    print()
    
    print("Results by Side:")
    print("-" * 70)
    for side_num in sorted(results['sides'].keys()):
        side_data = results['sides'][side_num]
        print(f"\nSide {side_num}: {side_data['side_name']}")
        print(f"  Heights scanned: {len(side_data['heights'])}")
        print(f"  Total detections: {side_data['total_detections']}")
        
        # Show first 3 heights
        for height_result in side_data['heights'][:3]:
            h = height_result['height']
            det = height_result['detections']['total_detections']
            print(f"    Height {h:.1f}m: {det} detections")
        
        if len(side_data['heights']) > 3:
            print(f"    ... and {len(side_data['heights']) - 3} more heights")
    
    print()
    print("=" * 70)
    print("Files Generated:")
    print(f"  Summary: data/results/scan_{results['scan_id']}_summary.json")
    print(f"  Images: {results['total_scans'] * 3} images (3 cameras per scan)")
    print("=" * 70)
    print()
    
    # Test single-side scan for comparison
    print()
    print("=" * 70)
    print("TESTING SINGLE-SIDE SCAN (for comparison)")
    print("=" * 70)
    print()
    
    scanner2 = AMRScanner(camera_mode="crop")
    results2 = scanner2.start_scan(amr_x=0, four_side_scan=False)
    
    print()
    print(f"Single-side scan completed: {results2['scan_id']}")
    print(f"Scan Type: {results2['scan_type']}")
    print(f"Total Scans: {results2['total_scans']}")
    print(f"Total Detections: {results2['total_detections']}")
    print()
    
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Run 'py -3.12 launch_3d.py' and select option 1")
    print("  2. Open the 4-side visualizer to see animated scanning")
    print("  3. Check data/results/ for generated images and JSON")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()
