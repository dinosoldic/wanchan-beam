"""Export the trained breed classifier for on-device LiteRT inference."""

import platform
from pathlib import Path

# LiteRT Torch is installed only inside the Linux export container.
import litert_torch  # pyright: ignore[reportMissingImports]
import numpy as np
import torch

from datasets.tsinghua_dogs import (
    TRAIN_SPLIT_PATH,
    build_breed_mapping,
    load_split_paths,
)
from training.breed_classifier import build_breed_classifier

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent

SOURCE_MODEL_PATH = ML_ROOT / "artifacts" / "classifier" / "breed-classifier.pt"
OUTPUT_MODEL_PATH = PROJECT_ROOT / "models" / "v1" / "mobile-breed-classifier.tflite"

INPUT_SIZE = 256
NUMBER_OF_BREEDS = 130
CLASSIFIER_HIDDEN_FEATURES = 1280
DROPOUT_PROBABILITY = 0.4

PARITY_ABSOLUTE_TOLERANCE = 1e-3
PARITY_RELATIVE_TOLERANCE = 1e-3


def unwrap_edge_output(output: object) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        if len(output) != 1:
            raise ValueError(f"Expected one classifier output, received {len(output)}.")

        output = output[0]

    logits = np.asarray(output)

    if logits.shape != (1, NUMBER_OF_BREEDS):
        raise ValueError(
            "Unexpected LiteRT output shape: "
            f"expected {(1, NUMBER_OF_BREEDS)}, got {logits.shape}."
        )

    return logits


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError(
            "LiteRT Torch export must run inside the Linux export container."
        )

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
        raise ValueError("Checkpoint breed ordering does not match the dataset.")

    if len(checkpoint_breed_names) != NUMBER_OF_BREEDS:
        raise ValueError(
            "Unexpected breed count: "
            f"expected {NUMBER_OF_BREEDS}, "
            f"got {len(checkpoint_breed_names)}."
        )

    checkpoint_input_size = int(checkpoint["model_input_size"])

    if checkpoint_input_size != INPUT_SIZE:
        raise ValueError(
            "Unexpected checkpoint input size: "
            f"expected {INPUT_SIZE}, got {checkpoint_input_size}."
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

    torch.manual_seed(0)

    sample_inputs = (
        torch.rand(
            (1, 3, INPUT_SIZE, INPUT_SIZE),
            dtype=torch.float32,
        ),
    )

    with torch.inference_mode():
        pytorch_logits = model(*sample_inputs).numpy()

    edge_model = litert_torch.convert(model, sample_inputs)
    litert_logits = unwrap_edge_output(edge_model(*sample_inputs))

    maximum_logit_difference = float(np.max(np.abs(pytorch_logits - litert_logits)))

    if not np.allclose(
        pytorch_logits,
        litert_logits,
        atol=PARITY_ABSOLUTE_TOLERANCE,
        rtol=PARITY_RELATIVE_TOLERANCE,
    ):
        raise RuntimeError("PyTorch/LiteRT classifier logit parity tolerance exceeded.")

    pytorch_top_1 = int(np.argmax(pytorch_logits, axis=1)[0])
    litert_top_1 = int(np.argmax(litert_logits, axis=1)[0])

    if pytorch_top_1 != litert_top_1:
        raise RuntimeError("PyTorch and LiteRT produced different top-1 classes.")

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(OUTPUT_MODEL_PATH))

    if not OUTPUT_MODEL_PATH.is_file():
        raise RuntimeError(f"LiteRT export was not created: {OUTPUT_MODEL_PATH}")

    print(f"Source checkpoint: {SOURCE_MODEL_PATH}")
    print(f"Input shape: [1, 3, {INPUT_SIZE}, {INPUT_SIZE}]")
    print(f"Output shape: [1, {NUMBER_OF_BREEDS}]")
    print("Precision: FP32")
    print("Maximum sample logit difference: " f"{maximum_logit_difference:.10f}")
    print(f"Matching sample top-1 class: {pytorch_top_1}")
    print(f"Exported mobile classifier: {OUTPUT_MODEL_PATH}")
    print(f"Model size: {OUTPUT_MODEL_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
