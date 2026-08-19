"""Construct the MobileNetV3 breed classifier."""

from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)
from torchvision.models.mobilenetv3 import MobileNetV3

NUMBER_OF_BREEDS = 130


def build_breed_classifier(
    number_of_breeds: int = NUMBER_OF_BREEDS,
    use_pretrained_weights: bool = True,
) -> MobileNetV3:
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


def unfreeze_for_fine_tuning(
    model: MobileNetV3,
    number_of_feature_modules: int = 3,
) -> None:
    """Unfreeze the classifier and deepest feature modules."""

    feature_modules = list(model.features.children())

    if not 1 <= number_of_feature_modules <= len(feature_modules):
        raise ValueError("The number of feature modules must fit inside model.features")

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    for feature_module in feature_modules[-number_of_feature_modules:]:
        for parameter in feature_module.parameters():
            parameter.requires_grad = True
