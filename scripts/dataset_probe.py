from __future__ import annotations

from pathlib import Path
import yaml # type: ignore

def find_yaml(dataset_dir: Path) -> Path:
    candidates = [
       dataset_dir / "data.yaml",
       dataset_dir / "dataset.yaml",
       dataset_dir / "data.yml",
       dataset_dir / "dataset.yml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No dataset YAML found in {dataset_dir}. Expected data.yaml or dataset.yaml")

def main() -> None:
    dataset_dir = Path("../data/roboflow/study-desk-items")
    yaml_path = find_yaml(dataset_dir)

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    print("=== Dataset YAML ===")
    print(f"YAML: {yaml_path}")

    base = dataset_dir.resolve()
    train = cfg.get("train")
    val = cfg.get("val")
    test = cfg.get("test")

    print(f"Base: {base}")
    print(f"train: {train}")
    print(f"val: {val}")
    print(f"test: {test}")

    names = cfg.get("names")
    nc = cfg.get("nc")

    if isinstance(names, dict):
        class_list = [names[k] for k in sorted(names.keys())]
    elif isinstance(names, list):
        class_list = names
    else:
        class_list = None

    print(f"nc: {nc}")
    if class_list:
        print(f"classes ({len(class_list)}): {class_list[:30]}{'...' if len(class_list) > 30 else ''}")

    def check_split(split_value: str | None, label: str) -> None:
        if not split_value:
            print(f"[WARN] No {label} split specified in YAML.")
            return

        p = (base / split_value).resolve() if not Path(split_value).is_absolute() else Path(split_value)
        print(f"{label} path resolved -> {p} | exists={p.exists()}")

    check_split(train, "train")
    check_split(val, "test")
    check_split(test, "test")

if __name__ == "__main__":
    main()