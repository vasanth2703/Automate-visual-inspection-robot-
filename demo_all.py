"""
Complete Demo - Shows all features
Run with: py -3.12 demo_all.py
"""

import webbrowser
from pathlib import Path
import time

def print_header(text):
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60 + "\n")

def main():
    print_header("AMR SCANNING SIMULATION - COMPLETE DEMO")
    
    print("This demo will showcase all features of the system:")
    print("  1. Run a quick simulation")
    print("  2. Open the 3D visualizer")
    print("  3. Show you where results are saved")
    print()
    
    input("Press ENTER to start the demo...")
    
    # Part 1: Quick simulation
    print_header("PART 1: Running Simulation")
    print("Simulating a scan at AMR position 0m with crop mode...")
    print()
    
    from simulation.simulator import AMRScanner
    
    scanner = AMRScanner(camera_mode="crop")
    
    def status_update(msg):
        print(f"  → {msg}")
    
    scanner.set_status_callback(status_update)
    
    print("Starting scan...")
    results = scanner.start_scan(amr_x=0)
    
    print()
    print(f"✓ Scan completed: {results['scan_id']}")
    print(f"  • Heights scanned: {results['total_heights']}")
    print(f"  • Total detections: {results['total_detections']}")
    print(f"  • Has defects: {results['has_defects']}")
    
    # Part 2: Show results
    print_header("PART 2: Scan Results")
    
    print("Results by height:")
    for height_data in results['heights_scanned'][:3]:  # Show first 3
        h = height_data['height']
        det = height_data['detections']
        print(f"  Height {h:.1f}m: {det['total_detections']} detections")
    print(f"  ... and {results['total_heights'] - 3} more heights")
    
    print()
    print(f"Results saved to: data/results/scan_{results['scan_id']}_*")
    print(f"  • 24 images (3 cameras × 8 heights)")
    print(f"  • 1 JSON summary file")
    
    # Part 3: 3D Visualization
    print_header("PART 3: 3D Visualization")
    
    print("Opening the combined dashboard in your browser...")
    print()
    print("In the dashboard you can:")
    print("  • See the 3D robot model")
    print("  • Control AMR position with sliders")
    print("  • Start animated scans")
    print("  • View real-time results")
    print()
    
    html_path = Path(__file__).parent / "frontend" / "combined_dashboard.html"
    
    input("Press ENTER to open the 3D visualizer...")
    
    webbrowser.open(html_path.as_uri())
    
    print("✓ Dashboard opened in browser!")
    
    # Part 4: Summary
    print_header("DEMO COMPLETE")
    
    print("What you've seen:")
    print("  ✓ State machine-based scanning workflow")
    print("  ✓ Multi-height vertical scanning")
    print("  ✓ 3-camera image capture")
    print("  ✓ Mock AI defect detection")
    print("  ✓ Result fusion and storage")
    print("  ✓ 3D visualization")
    print()
    print("Next steps:")
    print("  • Run 'py -3.12 examples.py' for more examples")
    print("  • Run 'py -3.12 run_server.py' to start API server")
    print("  • Edit simulation/config.py to customize")
    print("  • Replace detection_stub.py with real AI models")
    print()
    print("Files to explore:")
    print("  • simulation/simulator.py - Main scanning logic")
    print("  • simulation/camera_sim.py - Camera simulation")
    print("  • backend/main.py - FastAPI server")
    print("  • frontend/*.html - Web dashboards")
    print()
    print("Check QUICKSTART.txt and README.txt for more info!")
    print()
    print("=" * 60)
    print("Thank you for trying the AMR Scanning Simulation! 🤖")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
