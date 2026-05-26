from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2 # type: ignore
import numpy as np # type: ignore
from torch.fx.experimental.unification.unification_tools import first
from ultralytics import YOLO # type: ignore

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0, help="Webcam index")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    from __future__ import annotations

    import argparse
    import time
    from collections import defaultdict, deque
    from pathlib import Path

    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from ultralytics import YOLO  # type: ignore

    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser()
        p.add_argument("--cam", type=int, default=0, help="Webcam index")
        p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
        p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
        p.add_argument(
            "--tracker",
            type=str,
            default="bytetrack.yaml",
            help="Tracker config: bytetrack.yaml or botsort.yaml",
        )
        p.add_argument(
            "--trail-len",
            type=int,
            default=30,
            help="How many previous centre points to keep per track",
        )
        p.add_argument(
            "--show-centres",
            action="store_true",
            help="Draw centre points for tracked boxes",
        )
        return p.parse_args()

    def find_weights() -> Path:
        candidates = [
            Path("runs/detect/deskitems_baseline/weights/best.pt"),
            Path("runs/detect/deskitems_baseline/weights/last.pt"),
        ]

        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError(
            "No trained weights found.\n"
            "Expected one of:\n"
            "  runs/detect/deskitems_baseline/weights/best.pt\n"
            "  runs/detect/deskitems_baseline/weights/last.pt"
        )

    def make_colour(track_id: int) -> tuple[int, int, int]:
        """
        Deterministic colour from track ID.
        """
        rng = np.random.default_rng(track_id)
        colour = rng.integers(80, 256, size=3)
        return int(colour[0]), int(colour[1]), int(colour[2])

    def draw_label(
            img: np.ndarray,
            text: str,
            x1: int,
            y1: int,
            colour: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1

        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        top = max(0, y1 - th - 10)
        cv2.rectangle(img, (x1, top), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(
            img,
            text,
            (x1 + 3, max(12, y1 - 5)),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
def main() -> None:
    args = parse_args()

    print("Working directory:", Path.cwd())
    weights = find_weights()
    print("Loading weights from", weights.resolve())
    print("Using tracker:", args.tracker)

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.cam}")

    # Track History ID: Deque ("Deck" and dequeue (operation, remove from front of queue), enqueue (add to back of queue)
    track_history: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.trail_len))

    #Timing Data
    first_seen: dict[int, float] = {}
    last_seen: dict[int, float] = {}
    frames_seen: dict[int, int] = defaultdict(int)

    prev_time = time.time()

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame from webcam.")
            break

        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0

        # Track on current frame
        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            imgsz=args.imgsz,
            tracker=args.tracker,
            verbose=False,
        )

        annotated = frame.copy()
        result = results[0]

        current_ids: set[int] = set()

# April 28 Ending



