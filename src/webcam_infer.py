from __future__ import annotations

import argparse
from pathlib import Path

import cv2  # type: ignore
from torch.backends.mkl import verbose
from ultralytics import YOLO  # type: ignore

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()

def main() -> None:
    args = parse_args()

    weights = Path("runs/detect/runs/deskitems_baseline/weights/best.pt")
    if not weights.exists():
        weights = Path("runs/detect/runs/deskitems_baseline/weights/last.pt")
    if not weights.exists():
        raise FileNotFoundError("No trained weights found. Train first!")

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.cam}")

    print("Press 'q' to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        annotated = result[0].plot()

        cv2.imshow("Desk Items (Trained) – Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()