from __future__ import annotations

import platform
import sys


def main() -> None:
    print("=== Environment Check ===")
    print(f"Python: {sys.version.replace(chr(10), ' ')}")
    print(f"Platform: {platform.platform()}")

    try:
        import cv2  # type: ignore
        print(f"OpenCV: {cv2.__version__}")
    except Exception as e:
        print("OpenCV import FAILED:", repr(e))

    try:
        import torch  # type: ignore
        print(f"Torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"MPS available: {mps}")
    except Exception as e:
        print("Torch import FAILED:", repr(e))

    try:
        import ultralytics  # type: ignore
        print(f"Ultralytics: {ultralytics.__version__}")
    except Exception as e:
        print("Ultralytics import FAILED:", repr(e))


if __name__ == "__main__":
    main()