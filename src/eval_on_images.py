from __future__ import annotations

from pathlib import Path
import random
from tabnanny import verbose

import cv2  # type: ignore
from torchvision.datasets.folder import IMG_EXTENSIONS
from ultralytics import YOLO  # type: ignore

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def pick_images(root: Path, n: int = 12) -> list[Path]:
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    random.shuffle(imgs)
    return imgs[:n]

def main() -> None:
    weights = Path("runs/detect/runs/deskitems_baseline/weights/best.pt")
    if not weights.exists():
        weights = Path("runs/detect/runs/deskitems_baseline/weights/last.pt")
    if not weights.exists():
        raise FileNotFoundError("No trained weights found. Train first!")

    model = YOLO(str(weights))

    valid_images_dir = Path("../data/roboflow/study-desk-items/valid/images")
    if not valid_images_dir.exists():
        valid_images_dir = Path("../data/roboflow/study-desk-items/val/images")

    out_dir = Path("runs/annotated_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = pick_images(valid_images_dir, n=12)
    if not samples:
        raise FileNotFoundError(f"No images found under {valid_images_dir}")

    for img_path in samples:
        results = model.predict(str(img_path), conf=0.25, imgsz=640, verbose=False)
        annotated = results[0].plot()
        out_path = out_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
