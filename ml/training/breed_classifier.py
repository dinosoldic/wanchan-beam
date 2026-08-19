"""Construct the MobileNetV3 breed classifier."""

from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)

NUMBER_OF_BREEDS = 130


def build_breed_classifier(
    number_of_breeds: int = NUMBER_OF_BREEDS,
    use_pretrained_weights: bool = True,
) -> nn.Module:
    """Build MobileNetV3 for training or loading a saved checkpoint."""

    weights = MobileNet_V3_Large_Weights.DEFAULT if use_pretrained_weights else None
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
