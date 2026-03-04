from __future__ import annotations

# from multiprocessing.pool import workers
from pathlib import Path
from ultralytics import YOLO # type: ignore

def find_yaml(dataset_dir: Path) -> Path:
    for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        p = dataset_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No dataset YAML found in {dataset_dir}.")

def main() -> None:
    dataset_dir = Path("../data/roboflow/study-desk-items")
    data_yaml = find_yaml(dataset_dir)

    model = YOLO("../yolo26n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=11,
        imgsz=640,
        batch=8,
        workers=11,
        device="mps",
        pretrained=True,
        project="runs",
        name="deskitems_baseline",
    )

    print("Training complete")
    print(results)

if __name__ == "__main__":
    main()
