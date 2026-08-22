"""Compare PyTorch and ONNX outputs on all real validation crops."""

from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from datasets.tsinghua_dogs import (
    TRAIN_SPLIT_PATH,
    VALIDATION_SPLIT_PATH,
    TsinghuaDogsDataset,
    build_breed_mapping,
    load_split_paths,
)
from evaluation.evaluate_breed_classifier import load_model_checkpoint
from export.export_breed_classifier import (
    INPUT_SIZE,
    NUMBER_OF_BREEDS,
    OUTPUT_MODEL_PATH,
    SOURCE_MODEL_PATH,
)

BATCH_SIZE = 64
NUM_WORKERS = 4
MAXIMUM_ALLOWED_ABSOLUTE_DIFFERENCE = 0.0001
MAXIMUM_ALLOWED_MEAN_DIFFERENCE = 0.00001


def main() -> None:
    """Compare the complete validation split across both runtimes."""

    training_paths = load_split_paths(TRAIN_SPLIT_PATH)
    breed_names, breed_to_id = build_breed_mapping(training_paths)

    model, _, classifier_hidden_features, model_input_size = load_model_checkpoint(
        checkpoint_path=SOURCE_MODEL_PATH,
        expected_breed_names=breed_names,
        device=torch.device("cpu"),
    )

    if model_input_size != INPUT_SIZE:
        raise ValueError(f"Expected input size {INPUT_SIZE}, got {model_input_size}")

    validation_paths = load_split_paths(VALIDATION_SPLIT_PATH)
    validation_dataset = TsinghuaDogsDataset(
        validation_paths,
        breed_to_id,
        model_input_size=model_input_size,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )

    session = ort.InferenceSession(
        str(OUTPUT_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    maximum_absolute_difference = 0.0
    total_absolute_difference = 0.0
    total_logit_values = 0
    processed_examples = 0
    top_one_matches = 0
    top_two_matches = 0

    started_at = perf_counter()

    with torch.inference_mode():
        for batch_index, (batch_images, _) in enumerate(
            validation_loader,
            start=1,
        ):
            pytorch_logits_tensor = model(batch_images)
            pytorch_logits = pytorch_logits_tensor.numpy()

            runtime_outputs = session.run(
                ["logits"],
                {"images": batch_images.numpy()},
            )
            onnx_logits = runtime_outputs[0]

            if not isinstance(onnx_logits, np.ndarray):
                raise TypeError(
                    "Expected dense NumPy logits, but ONNX Runtime returned "
                    f"{type(onnx_logits).__name__}"
                )

            batch_size = batch_images.shape[0]
            expected_shape = (batch_size, NUMBER_OF_BREEDS)

            if pytorch_logits.shape != expected_shape:
                raise ValueError(
                    "Unexpected PyTorch output shape: " f"{pytorch_logits.shape}"
                )

            if onnx_logits.shape != expected_shape:
                raise ValueError(f"Unexpected ONNX output shape: {onnx_logits.shape}")

            absolute_difference = np.abs(pytorch_logits - onnx_logits)

            maximum_absolute_difference = max(
                maximum_absolute_difference,
                float(absolute_difference.max()),
            )
            total_absolute_difference += float(
                absolute_difference.sum(dtype=np.float64)
            )
            total_logit_values += int(absolute_difference.size)

            pytorch_top_two = torch.topk(
                pytorch_logits_tensor,
                k=2,
                dim=1,
            ).indices.numpy()

            onnx_top_two = np.argsort(
                -onnx_logits,
                axis=1,
            )[:, :2]

            top_one_matches += int(
                np.count_nonzero(pytorch_top_two[:, 0] == onnx_top_two[:, 0])
            )
            top_two_matches += int(
                np.count_nonzero(
                    np.all(
                        pytorch_top_two == onnx_top_two,
                        axis=1,
                    )
                )
            )

            processed_examples += batch_size

            if batch_index % 10 == 0 or processed_examples == len(validation_dataset):
                print(
                    f"Compared {processed_examples}/"
                    f"{len(validation_dataset)} validation examples"
                )

    if processed_examples != len(validation_dataset):
        raise RuntimeError("Parity verification did not process the complete dataset")

    mean_absolute_difference = total_absolute_difference / total_logit_values
    elapsed_seconds = perf_counter() - started_at

    print()
    print(f"Validation examples compared: {processed_examples}")
    print(f"Classifier hidden features: {classifier_hidden_features}")
    print(f"Output shape per batch: [batch, {NUMBER_OF_BREEDS}]")
    print("Maximum absolute logit difference: " f"{maximum_absolute_difference:.10f}")
    print("Mean absolute logit difference: " f"{mean_absolute_difference:.10f}")
    print(f"Matching top-1 IDs: {top_one_matches}/{processed_examples}")
    print("Matching ordered top-2 IDs: " f"{top_two_matches}/{processed_examples}")
    print(f"Elapsed time: {elapsed_seconds:.1f} seconds")

    if maximum_absolute_difference > MAXIMUM_ALLOWED_ABSOLUTE_DIFFERENCE:
        raise AssertionError(
            "Maximum logit difference exceeded the allowed limit: "
            f"{maximum_absolute_difference:.10f}"
        )

    if mean_absolute_difference > MAXIMUM_ALLOWED_MEAN_DIFFERENCE:
        raise AssertionError(
            "Mean logit difference exceeded the allowed limit: "
            f"{mean_absolute_difference:.10f}"
        )

    if top_one_matches != processed_examples:
        raise AssertionError("PyTorch and ONNX top-1 predictions did not all match")

    if top_two_matches != processed_examples:
        raise AssertionError(
            "PyTorch and ONNX ordered top-2 predictions did not all match"
        )

    print("Full PyTorch/ONNX parity validation passed.")


if __name__ == "__main__":
    main()
