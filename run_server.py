"""
Quick start script for the AMR Scanning API server
Run with: py -3.12 run_server.py
"""

import uvicorn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("=" * 60)
    print("AMR SCANNING SIMULATION API SERVER")
    print("=" * 60)
    print()
    print("Starting server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Dashboard at: frontend/dashboard.html")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
