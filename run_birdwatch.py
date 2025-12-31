#!/usr/bin/env python
"""
Convenience script to run birdwatch-ai commands from project root.
This script ensures the package can be imported correctly.
"""

import sys
import os

# Add src directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import and run the main function
from birdwatch_ai.run import main

if __name__ == "__main__":
    main()

