from __future__ import annotations

import argparse
from typing import Optional

import cv2  # type: ignore
from ultralytics import YOLO  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0, help="Webcam index (0, 1, ...)")
    p.add_argument("--model", type=str, default="yolo26n.pt", help="Pretrained model name/path")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.cam}")

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        annotated = results[0].plot()  # returns an annotated image

        cv2.imshow("Pretrained YOLO Demo", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()