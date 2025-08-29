#!/usr/bin/env python3
"""
Startup script for Pattern Detection API
"""
import sys
import os
import subprocess
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import yfinance
        import matplotlib
        import plotly
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        return False

def install_requirements():
    """Install requirements from requirements.txt"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install requirements")
        return False

def start_api():
    """Start the FastAPI server"""
    print("Starting Pattern Detection API...")
    print("API will be available at:")
    print("  - Main API: http://localhost:8000")
    print("  - Documentation: http://localhost:8000/api/docs")
    print("  - Alternative docs: http://localhost:8000/api/redoc")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        import uvicorn
        uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("FastAPI/Uvicorn not installed. Installing requirements first...")
        if install_requirements():
            import uvicorn
            uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
        else:
            print("Failed to install requirements. Please install manually.")

def main():
    """Main startup function"""
    print("Pattern Detection API Startup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("main_api.py").exists():
        print("Error: main_api.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        print("\nWould you like to install requirements now? (y/n): ", end="")
        response = input().lower().strip()
        if response in ['y', 'yes']:
            if not install_requirements():
                sys.exit(1)
        else:
            print("Please install requirements manually using: pip install -r requirements.txt")
            sys.exit(1)
    
    # Create output directories
    output_dirs = ["outputs", "outputs/charts", "outputs/reports", "outputs/reports/score"]
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Start the API
    start_api()

if __name__ == "__main__":
    main()
