"""
Launch React Dashboard
Opens the professional industry-level dashboard in browser
"""
import webbrowser
import os
from pathlib import Path
import time

def launch_dashboard():
    """Launch the React dashboard"""
    print("="*60)
    print("LAUNCHING REACT DASHBOARD")
    print("="*60)
    
    dashboard_path = Path('frontend/react_dashboard.html').absolute()
    
    if not dashboard_path.exists():
        print(f"✗ Dashboard not found: {dashboard_path}")
        return
    
    print(f"\n✓ Dashboard found: {dashboard_path}")
    print("\nOpening in browser...")
    
    # Open in default browser
    webbrowser.open(f'file://{dashboard_path}')
    
    print("\n" + "="*60)
    print("DASHBOARD LAUNCHED")
    print("="*60)
    print("\nFeatures:")
    print("  ✓ Real-time scan monitoring")
    print("  ✓ Component statistics")
    print("  ✓ Detection details with heatmaps")
    print("  ✓ Visual analytics and charts")
    print("  ✓ Scan history browser")
    print("\nMake sure:")
    print("  1. Backend is running: py -3.12 run_server.py")
    print("  2. Supabase is configured in .env")
    print("  3. Database schema is created")
    print("\nTo test Supabase:")
    print("  py -3.12 test_supabase.py")
    print("="*60)

if __name__ == '__main__':
    launch_dashboard()
