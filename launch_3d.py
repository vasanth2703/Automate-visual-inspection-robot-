"""Launch 3D Visualizer - py -3.12 launch_3d.py"""
import webbrowser
from pathlib import Path

print("=" * 60)
print("AMR 3D VISUALIZATION LAUNCHER")
print("=" * 60)
print()
print("Available dashboards:")
print("  1. 4-Side Industrial Inspection (RECOMMENDED)")
print("  2. Combined Dashboard (3D + Data)")
print("  3. 3D Visualizer Only")
print("  4. Data Dashboard Only")
print()

choice = input("Select dashboard (1-4) [1]: ").strip() or "1"

dashboards = {
    "1": "visualizer_4side.html",
    "2": "combined_dashboard.html",
    "3": "visualizer_3d.html",
    "4": "dashboard.html"
}

html_file = dashboards.get(choice, "visualizer_4side.html")
html_path = Path(__file__).parent / "frontend" / html_file

print(f"\nOpening: {html_file}")
print(f"Path: {html_path}")
print()
print("CONTROLS:")
print("  • Sliders: Control AMR position and lift height")
print("  • Start Scan: Animate full scanning sequence")
print("  • Reset: Return to initial position")
print()
print("TIP: Run 'py -3.12 run_server.py' for backend integration")
print("=" * 60)

webbrowser.open(html_path.as_uri())
print("\n✓ Browser opened successfully!")
