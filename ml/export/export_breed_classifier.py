"""Load and validate the selected breed-classifier checkpoint."""

from pathlib import Path
import numpy as np

import torch

import onnx
import onnxruntime as ort

from datasets.tsinghua_dogs import (
    TRAIN_SPLIT_PATH,
    build_breed_mapping,
    load_split_paths,
)
from training.breed_classifier import build_breed_classifier

ML_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL_PATH = ML_ROOT / "artifacts" / "classifier" / "breed-classifier.pt"
OUTPUT_MODEL_PATH = SOURCE_MODEL_PATH.with_suffix(".onnx")

ONNX_OPSET_VERSION = 20
INPUT_SIZE = 256
NUMBER_OF_BREEDS = 130
CLASSIFIER_HIDDEN_FEATURES = 1280
DROPOUT_PROBABILITY = 0.4


def validate_onnx_export() -> None:
    """Validate the ONNX graph and its runtime input/output contract."""

    exported_model = onnx.load(str(OUTPUT_MODEL_PATH))
    onnx.checker.check_model(exported_model)

    session = ort.InferenceSession(
        str(OUTPUT_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1:
        raise ValueError(f"Expected one ONNX input, but found {len(inputs)}")

    if len(outputs) != 1:
        raise ValueError(f"Expected one ONNX output, but found {len(outputs)}")

    model_input = inputs[0]
    model_output = outputs[0]

    expected_input_shape = ["batch", 3, INPUT_SIZE, INPUT_SIZE]
    expected_output_shape = ["batch", NUMBER_OF_BREEDS]

    if model_input.name != "images":
        raise ValueError(
            f"Expected input name 'images', but found {model_input.name!r}"
        )

    if model_input.type != "tensor(float)":
        raise ValueError(f"Expected float32 input, but found {model_input.type!r}")

    if model_input.shape != expected_input_shape:
        raise ValueError(
            "Unexpected ONNX input shape: "
            f"expected {expected_input_shape}, got {model_input.shape}"
        )

    if model_output.name != "logits":
        raise ValueError(
            f"Expected output name 'logits', but found {model_output.name!r}"
        )

    if model_output.type != "tensor(float)":
        raise ValueError(f"Expected float32 output, but found {model_output.type!r}")

    if model_output.shape != expected_output_shape:
        raise ValueError(
            "Unexpected ONNX output shape: "
            f"expected {expected_output_shape}, got {model_output.shape}"
        )

    for batch_size in (1, 3):
        test_images = torch.rand(
            (batch_size, 3, INPUT_SIZE, INPUT_SIZE),
            dtype=torch.float32,
        ).numpy()

        runtime_outputs = session.run(
            ["logits"],
            {"images": test_images},
        )
        output_logits = runtime_outputs[0]

        if not isinstance(output_logits, np.ndarray):
            raise TypeError(
                "Expected dense NumPy logits, but ONNX Runtime returned "
                f"{type(output_logits).__name__}"
            )

        expected_runtime_shape = (batch_size, NUMBER_OF_BREEDS)

        if output_logits.shape != expected_runtime_shape:
            raise ValueError(
                "Unexpected runtime output shape: "
                f"expected {expected_runtime_shape}, "
                f"got {output_logits.shape}"
            )

    print("ONNX graph validation passed.")
    print("Dynamic batch validation passed for batch sizes 1 and 3.")


def main() -> None:
    if not SOURCE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {SOURCE_MODEL_PATH}")

    training_paths = load_split_paths(TRAIN_SPLIT_PATH)
    expected_breed_names, _ = build_breed_mapping(training_paths)

    checkpoint = torch.load(
        SOURCE_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint_breed_names = tuple(checkpoint["breed_names"])

    if checkpoint_breed_names != expected_breed_names:
        raise ValueError("Checkpoint breed ordering does not match the dataset")

    checkpoint_input_size = int(checkpoint["model_input_size"])

    if checkpoint_input_size != INPUT_SIZE:
        raise ValueError(
            "Unexpected checkpoint input size: "
            f"expected {INPUT_SIZE}, got {checkpoint_input_size}"
        )

    model = build_breed_classifier(
        number_of_breeds=NUMBER_OF_BREEDS,
        use_pretrained_weights=False,
        dropout_probability=DROPOUT_PROBABILITY,
        classifier_hidden_features=CLASSIFIER_HIDDEN_FEATURES,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    example_images = torch.zeros(
        (1, 3, INPUT_SIZE, INPUT_SIZE),
        dtype=torch.float32,
    )

    dynamic_batch = torch.export.Dim(
        "batch",
        min=1,
    )

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example_images,),
            OUTPUT_MODEL_PATH,
            input_names=["images"],
            output_names=["logits"],
            opset_version=ONNX_OPSET_VERSION,
            dynamo=True,
            dynamic_shapes=({0: dynamic_batch},),
            external_data=False,
        )

    if not OUTPUT_MODEL_PATH.is_file():
        raise RuntimeError(f"ONNX export was not created: {OUTPUT_MODEL_PATH}")

    validate_onnx_export()  # validate the onnx

    print(f"Loaded checkpoint: {SOURCE_MODEL_PATH}")
    print(f"Input shape: [batch, 3, {INPUT_SIZE}, {INPUT_SIZE}]")
    print(f"Output shape: [batch, {len(checkpoint_breed_names)}]")
    print("Checkpoint validation passed.")
    print(f"Exported ONNX model: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
