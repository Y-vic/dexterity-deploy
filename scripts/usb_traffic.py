#!/usr/bin/env python3
"""Standalone entry point; does not load ROS, SDKs, or camera streams."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ur_common"))
from ur_common.usb_traffic import main

if __name__ == "__main__":
    main()
