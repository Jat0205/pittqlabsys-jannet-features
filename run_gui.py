#!/usr/bin/env python3
"""
Simple launcher script for the AQuISS GUI.
This script sets up the Python path correctly and launches the GUI.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path

project_root = Path(__file__).parent
#sys.path.insert(0, str(project_root))
# Add the absolute path to the 'src' folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


# Now import and run the app
from src.app import launch_gui

if __name__ == "__main__":
    launch_gui()