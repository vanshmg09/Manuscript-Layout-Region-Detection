"""
Convenience test runner script ensuring proper C-extension DLL initialization on Windows.
Usage:
    python run_tests.py
"""

import sys
import cv2
import pytest

if __name__ == "__main__":
    exit_code = pytest.main(["-v"] + sys.argv[1:])
    sys.exit(exit_code)

