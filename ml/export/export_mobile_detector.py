import platform
from pathlib import Path

from ultralytics import YOLO

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent

SOURCE_MODEL_PATH = ML_ROOT / "artifacts" / "detector" / "yolo26n.pt"
DEPLOYMENT_MODEL_PATH = PROJECT_ROOT / "models" / "mobile" / "v1" / "detector.tflite"

INPUT_SIZE = 544


def main() -> None:
    if platform.system() not in {"Linux", "Darwin"}:
        raise RuntimeError(
            "Ultralytics LiteRT export requires Linux x86 or macOS. "
            "Run this script inside the Linux export environment."
        )

    if not SOURCE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Detector not found: {SOURCE_MODEL_PATH}")

    model = YOLO(str(SOURCE_MODEL_PATH))

    exported = model.export(
        format="litert",
        imgsz=INPUT_SIZE,
        batch=1,
        quantize=32,
        device="cpu",
    )

    if exported is None:
        raise RuntimeError("Ultralytics did not return an exported model path.")

    exported_path = Path(exported).resolve()
    DEPLOYMENT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    exported_path.replace(DEPLOYMENT_MODEL_PATH)

    print(f"Source checkpoint: {SOURCE_MODEL_PATH}")
    print(f"Input shape: [1, 3, {INPUT_SIZE}, {INPUT_SIZE}]")
    print("Precision: FP32")
    print(f"Exported mobile detector: {DEPLOYMENT_MODEL_PATH}")
    print(f"Model size: {DEPLOYMENT_MODEL_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
