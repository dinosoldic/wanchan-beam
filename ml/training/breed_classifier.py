"""Construct and inspect the MobileNetV3 breed classifier."""

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)

NUMBER_OF_BREEDS = 130
MODEL_INPUT_SIZE = 224
EXAMPLE_BATCH_SIZE = 4


def build_breed_classifier(
    number_of_breeds: int = NUMBER_OF_BREEDS,
) -> nn.Module:
    """Build MobileNetV3 with a trainable dog-breed output layer."""

    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)

    # Preserve the pretrained features during the first training phase.
    for parameter in model.parameters():
        parameter.requires_grad = False

    original_output_layer = model.classifier[-1]

    if not isinstance(original_output_layer, nn.Linear):
        raise TypeError("Expected MobileNetV3's final classifier layer to be linear")

    model.classifier[-1] = nn.Linear(
        in_features=original_output_layer.in_features,
        out_features=number_of_breeds,
    )

    return model


def main() -> None:
    """Build the classifier and verify its trainable output shape."""

    model = build_breed_classifier()

    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    example_batch = torch.randn(
        EXAMPLE_BATCH_SIZE,
        3,
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
    )

    model.eval()

    with torch.inference_mode():
        output_logits = model(example_batch)

    print(f"Trainable parameters: {trainable_parameter_count:,}")
    print(f"Input shape: {example_batch.shape}")
    print(f"Output shape: {output_logits.shape}")


if __name__ == "__main__":
    main()
