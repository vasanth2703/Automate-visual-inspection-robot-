"""
Example usage of the AMR Scanning Simulation
Run with: py -3.12 examples.py
"""

from simulation.simulator import AMRScanner
from simulation import config
import json

def example_1_basic_scan():
    """Example 1: Basic scan with default settings"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Scan")
    print("=" * 60)
    
    scanner = AMRScanner(camera_mode="crop")
    results = scanner.start_scan(amr_x=0)
    
    print(f"\nScan completed: {results['scan_id']}")
    print(f"Total detections: {results['total_detections']}")
    print(f"Has defects: {results['has_defects']}")

def example_2_with_callbacks():
    """Example 2: Scan with status callbacks"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Scan with Callbacks")
    print("=" * 60)
    
    def status_callback(message):
        print(f"  STATUS: {message}")
    
    def progress_callback(current, total):
        percent = (current / total) * 100
        print(f"  PROGRESS: {current}/{total} ({percent:.1f}%)")
    
    scanner = AMRScanner(camera_mode="random")
    scanner.set_status_callback(status_callback)
    scanner.set_progress_callback(progress_callback)
    
    results = scanner.start_scan(amr_x=1)
    print(f"\nCompleted with {results['total_detections']} detections")

def example_3_multiple_positions():
    """Example 3: Scan at multiple AMR positions"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multiple AMR Positions")
    print("=" * 60)
    
    all_results = []
    
    for amr_x in config.AMR_POSITIONS:
        print(f"\nScanning at AMR position {amr_x}m...")
        scanner = AMRScanner(camera_mode="crop")
        results = scanner.start_scan(amr_x=amr_x)
        all_results.append(results)
        print(f"  Detections: {results['total_detections']}")
    
    total_detections = sum(r['total_detections'] for r in all_results)
    print(f"\nTotal detections across all positions: {total_detections}")

def example_4_analyze_results():
    """Example 4: Detailed result analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Detailed Result Analysis")
    print("=" * 60)
    
    scanner = AMRScanner(camera_mode="crop")
    results = scanner.start_scan(amr_x=0)
    
    # Analyze by height
    print("\nDetections by height:")
    for height_data in results['heights_scanned']:
        h = height_data['height']
        det = height_data['detections']
        print(f"\n  Height {h:.1f}m:")
        print(f"    Total: {det['total_detections']}")
        
        if det['defect_counts']:
            print("    Defects:")
            for defect_type, count in det['defect_counts'].items():
                print(f"      - {defect_type}: {count}")
    
    # Analyze by camera
    print("\n\nDetections by camera:")
    camera_counts = {"left": 0, "center": 0, "right": 0}
    
    for height_data in results['heights_scanned']:
        for detection in height_data['detections']['detections']:
            camera_counts[detection['camera']] += 1
    
    for camera, count in camera_counts.items():
        print(f"  {camera.capitalize()}: {count}")
    
    # Analyze by defect type
    print("\n\nTotal defects by type:")
    defect_totals = {}
    
    for height_data in results['heights_scanned']:
        for defect_type, count in height_data['detections']['defect_counts'].items():
            defect_totals[defect_type] = defect_totals.get(defect_type, 0) + count
    
    for defect_type, count in sorted(defect_totals.items(), key=lambda x: x[1], reverse=True):
        print(f"  {defect_type}: {count}")

def example_5_custom_config():
    """Example 5: Using custom configuration"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Configuration")
    print("=" * 60)
    
    # Show current config
    print("\nCurrent configuration:")
    print(f"  Height levels: {config.HEIGHT_LEVELS}")
    print(f"  Movement delay: {config.MOVEMENT_DELAY}s")
    print(f"  Stabilization delay: {config.STABILIZATION_DELAY}s")
    print(f"  Defect types: {config.DEFECT_TYPES}")
    
    # You can modify config at runtime
    original_delay = config.MOVEMENT_DELAY
    config.MOVEMENT_DELAY = 0.2  # Faster simulation
    
    print("\nRunning faster simulation...")
    scanner = AMRScanner(camera_mode="crop")
    results = scanner.start_scan(amr_x=0)
    
    print(f"Completed in faster mode: {results['total_detections']} detections")
    
    # Restore original
    config.MOVEMENT_DELAY = original_delay

def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("AMR SCANNING SIMULATION - EXAMPLES")
    print("=" * 60)
    
    try:
        example_1_basic_scan()
        example_2_with_callbacks()
        example_3_multiple_positions()
        example_4_analyze_results()
        example_5_custom_config()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED")
        print("=" * 60)
        print("\nCheck data/results/ for generated images and JSON files")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
