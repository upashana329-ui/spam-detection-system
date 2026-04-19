#!/usr/bin/env python3
"""
Spam Detection System - Main Entry Point
A comprehensive machine learning-based spam detection system with GUI
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui import SpamDetectionGUI

def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    
    # Set high DPI support
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Set application properties
    app.setApplicationName("Spam Detection System")
    app.setApplicationDisplayName("AI-Powered Spam Detection System")
    
    # Create and show main window
    window = SpamDetectionGUI()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()