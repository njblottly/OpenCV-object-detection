from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2  # type: ignore
import numpy as np  # type: ignore
import yaml  # type: ignore


# ── CONFIG ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "roboflow" / "study-desk-items"
N_SAMPLES = 30          # images to annotate & save (set to -1 for ALL)
SEED = 42               # reproducible sample; change or set None
OUT_DIR = PROJECT_ROOT / "runs" / "gt_review"
TINY_THRESH = 0.01      # flag boxes smaller than 1 % of image area
LARGE_THRESH = 0.90     # flag boxes larger  than 90% of image area
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ─────────────────────────────────────────────────────────────────────


# ── COLOUR PALETTE (up to 30 classes) ────────────────────────────────
def _make_palette(n: int = 30) -> list[tuple[int, int, int]]:
    """Deterministic, high-contrast BGR colours via the HSV wheel."""
    colours: list[tuple[int, int, int]] = []
    for i in range(n):
        hue = int(180 * i / n)  # OpenCV hue range 0-179
        hsv = np.uint8([[[hue, 220, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colours.append(tuple(int(c) for c in bgr))
    return colours


PALETTE = _make_palette()


# ── HELPERS ──────────────────────────────────────────────────────────
def find_yaml(dataset_dir: Path) -> Path:
    for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        p = dataset_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No dataset YAML in {dataset_dir}")


def load_config(yaml_path: Path) -> dict:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if isinstance(names, dict):
        cfg["_class_list"] = [names[k] for k in sorted(names.keys())]
    elif isinstance(names, list):
        cfg["_class_list"] = names
    else:
        cfg["_class_list"] = [str(i) for i in range(cfg.get("nc", 0))]
    return cfg


def is_image_dir(p: Path) -> bool:
    """Return True if p exists and contains at least one image file."""
    return p.is_dir() and any(child.suffix.lower() in IMG_EXTS for child in p.iterdir())


def resolve_split(yaml_dir: Path, dataset_dir: Path, split_value: Optional[str]) -> Optional[Path]:
    """
    Return the *images* directory for a split, or None.

    Tries both:
    - paths relative to the YAML file location
    - paths relative to DATASET_DIR

    Accepts either:
    - a direct images directory
    - a split root containing an images/ subfolder
    """
    if not split_value:
        return None

    raw = Path(split_value)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((yaml_dir / raw).resolve())
        candidates.append((dataset_dir / raw).resolve())

    expanded: list[Path] = []
    for p in candidates:
        expanded.append(p)
        expanded.append((p / "images").resolve())

    # Also try common Roboflow split names directly under dataset_dir.
    name = raw.name.lower()
    if name in {"train", "valid", "val", "test"}:
        expanded.append((dataset_dir / name).resolve())
        expanded.append((dataset_dir / name / "images").resolve())

    for p in expanded:
        if is_image_dir(p):
            return p

    return None


def label_path_for(img_path: Path) -> Path:
    """Given .../images/foo.jpg → .../labels/foo.txt"""
    return img_path.parent.parent / "labels" / f"{img_path.stem}.txt"


def parse_label(txt_path: Path) -> list[dict]:
    """Parse a YOLO-format label file → list of {cls, cx, cy, w, h}."""
    boxes: list[dict] = []
    if not txt_path.exists():
        return boxes

    text = txt_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append({"cls": cls_id, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def draw_gt(img: np.ndarray, boxes: list[dict], class_names: list[str]) -> np.ndarray:
    """Draw ground-truth boxes and class labels on the image."""
    h_img, w_img = img.shape[:2]
    out = img.copy()

    for b in boxes:
        cls = b["cls"]
        cx, cy, bw, bh = b["cx"], b["cy"], b["w"], b["h"]

        # Convert normalised xywh → pixel xyxy
        x1 = int((cx - bw / 2) * w_img)
        y1 = int((cy - bh / 2) * h_img)
        x2 = int((cx + bw / 2) * w_img)
        y2 = int((cy + bh / 2) * h_img)

        colour = PALETTE[cls % len(PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        label = class_names[cls] if cls < len(class_names) else f"cls_{cls}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top_y = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, top_y), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


# ── MAIN ─────────────────────────────────────────────────────────────
def main() -> None:
    yaml_path = find_yaml(DATASET_DIR)
    cfg = load_config(yaml_path)
    class_names: list[str] = cfg["_class_list"]

    dataset_base = DATASET_DIR.resolve()
    yaml_dir = yaml_path.parent.resolve()

    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Dataset dir   : {dataset_base}")
    print(f"Dataset YAML  : {yaml_path}")
    print(f"YAML dir      : {yaml_dir}")
    print(f"Classes ({len(class_names)}): {class_names}\n")

    # ── Gather ALL images across every split ──
    all_images: dict[str, list[Path]] = {}
    for split_key in ("train", "val", "valid", "test"):
        split_val = cfg.get(split_key)
        img_dir = resolve_split(yaml_dir, dataset_base, split_val)
        print(f"[DEBUG] split={split_key!r}, yaml value={split_val!r}, resolved={img_dir}")

        if img_dir and img_dir.is_dir():
            imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            if imgs:
                all_images[split_key] = imgs

    if not all_images:
        raise FileNotFoundError(
            "No image directories found.\n"
            f"DATASET_DIR = {DATASET_DIR}\n"
            f"yaml_path = {yaml_path}\n"
            "Open data.yaml and inspect the train/val/valid/test entries."
        )

    for split, imgs in all_images.items():
        print(f"  {split:>6s}: {len(imgs)} images")

    # ── Full-dataset class distribution & issue detection ──
    global_class_counts: Counter[str] = Counter()
    total_boxes = 0
    images_without_labels = 0
    empty_label_files = 0
    tiny_boxes: list[tuple[Path, dict]] = []
    large_boxes: list[tuple[Path, dict]] = []
    class_id_oob: list[tuple[Path, int]] = []

    print("\nScanning every label file …")
    for split, imgs in all_images.items():
        for img_path in imgs:
            lbl = label_path_for(img_path)
            if not lbl.exists():
                images_without_labels += 1
                continue

            boxes = parse_label(lbl)
            if not boxes:
                empty_label_files += 1
                continue

            for b in boxes:
                total_boxes += 1
                cid = b["cls"]
                if cid < len(class_names):
                    global_class_counts[class_names[cid]] += 1
                else:
                    class_id_oob.append((img_path, cid))
                    global_class_counts[f"UNKNOWN_{cid}"] += 1

                area = b["w"] * b["h"]
                if area < TINY_THRESH:
                    tiny_boxes.append((img_path, b))
                if area > LARGE_THRESH:
                    large_boxes.append((img_path, b))

    # ── Pick sample images (from train split preferably) ──
    sample_split = "train" if "train" in all_images else list(all_images.keys())[0]
    sample_pool = list(all_images[sample_split])

    if SEED is not None:
        random.seed(SEED)
    random.shuffle(sample_pool)

    n = len(sample_pool) if N_SAMPLES == -1 else min(N_SAMPLES, len(sample_pool))
    samples = sample_pool[:n]

    # ── Draw & save annotated images ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDrawing GT boxes on {n} sample images → {OUT_DIR}/\n")

    for img_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read {img_path}")
            continue
        boxes = parse_label(label_path_for(img_path))
        annotated = draw_gt(img, boxes, class_names)
        out_path = OUT_DIR / f"{img_path.stem}_gt.jpg"
        cv2.imwrite(str(out_path), annotated)

    # ── Build report ──
    report_lines = [
        "DATASET QUALITY REPORT",
        "=" * 50,
        f"YAML           : {yaml_path}",
        f"Classes        : {len(class_names)}",
        f"Total boxes    : {total_boxes}",
        "",
    ]

    for split, imgs in all_images.items():
        report_lines.append(f"  {split:>6s}: {len(imgs)} images")

    report_lines += ["", "CLASS DISTRIBUTION", "-" * 40]
    if global_class_counts:
        max_count = max(global_class_counts.values())
        for name, count in global_class_counts.most_common():
            bar = "█" * int(40 * count / max_count)
            report_lines.append(f"  {name:>25s}  {count:5d}  {bar}")
    else:
        report_lines.append("  No class counts collected.")

    report_lines += ["", "POTENTIAL ISSUES", "-" * 40]
    report_lines.append(f"  Images with no label file  : {images_without_labels}")
    report_lines.append(f"  Label files with 0 boxes   : {empty_label_files}")
    report_lines.append(f"  Tiny boxes  (<{TINY_THRESH*100:.0f}% area)   : {len(tiny_boxes)}")
    report_lines.append(f"  Huge boxes  (>{LARGE_THRESH*100:.0f}% area)  : {len(large_boxes)}")
    report_lines.append(f"  Out-of-bounds class IDs    : {len(class_id_oob)}")

    if tiny_boxes:
        report_lines += ["", "  TINY BOX SAMPLES (first 10):"]
        for path, b in tiny_boxes[:10]:
            area_pct = b["w"] * b["h"] * 100
            report_lines.append(f"    {path.name}  cls={b['cls']}  area={area_pct:.2f}%")

    if large_boxes:
        report_lines += ["", "  HUGE BOX SAMPLES (first 10):"]
        for path, b in large_boxes[:10]:
            area_pct = b["w"] * b["h"] * 100
            report_lines.append(f"    {path.name}  cls={b['cls']}  area={area_pct:.2f}%")

    if class_id_oob:
        report_lines += ["", "  OUT-OF-BOUNDS CLASS IDS (first 10):"]
        for path, cid in class_id_oob[:10]:
            report_lines.append(f"    {path.name}  cls={cid}")

    if global_class_counts:
        most = global_class_counts.most_common(1)[0][1]
        least = global_class_counts.most_common()[-1][1]
        ratio = most / max(least, 1)
        if ratio > 5:
            report_lines += [
                "",
                f"  CLASS IMBALANCE: most-to-least ratio = {ratio:.1f}x",
                "  Consider oversampling the minority class or adding more data.",
            ]

    report_text = "\n".join(report_lines)
    report_path = OUT_DIR / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nReport saved  -> {report_path}")
    print(f"GT images     -> {OUT_DIR}/")
    print(f"\nOpen {OUT_DIR}/ in Finder and inspect the *_gt.jpg files.")
    print("Look for wrong labels, missing boxes, or wildly wrong box sizes.")


if __name__ == "__main__":
    main()