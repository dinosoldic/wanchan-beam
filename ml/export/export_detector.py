"""Export the selected server detector checkpoint to ONNX."""

from pathlib import Path

from ultralytics import YOLO

ML_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL_PATH = ML_ROOT / "artifacts" / "detector" / "yolo26s.pt"

INPUT_SIZE = 960


def main() -> None:
    if not SOURCE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Detector not found: {SOURCE_MODEL_PATH}")

    model = YOLO(str(SOURCE_MODEL_PATH))

    exported = model.export(
        format="onnx",
        imgsz=INPUT_SIZE,
        batch=1,
        dynamic=False,
        simplify=True,
        nms=False,
        device="cpu",
    )

    if exported is None:
        raise RuntimeError("Ultralytics did not return an exported model path.")

    exported_path = Path(exported)
    print(f"Exported detector: {exported_path}")


if __name__ == "__main__":
    main()
