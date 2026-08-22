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
    dropout_probability: float = 0.2,
    classifier_hidden_features: int | None = 1280,
) -> MobileNetV3:
    """Build MobileNetV3 with a configurable breed-classification head."""

    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError("Dropout probability must be between 0 and 1")

    if classifier_hidden_features is not None and classifier_hidden_features <= 0:
        raise ValueError("Classifier hidden features must be positive or None")

    weights = MobileNet_V3_Large_Weights.DEFAULT if use_pretrained_weights else None
    model = mobilenet_v3_large(weights=weights)

    original_input_layer = model.classifier[0]
    original_dropout_layer = model.classifier[2]
    original_output_layer = model.classifier[-1]

    if not isinstance(original_input_layer, nn.Linear):
        raise TypeError("Expected MobileNetV3 classifier[0] to be linear")

    if not isinstance(original_dropout_layer, nn.Dropout):
        raise TypeError("Expected MobileNetV3 classifier[2] to be dropout")

    if not isinstance(original_output_layer, nn.Linear):
        raise TypeError("Expected MobileNetV3's final classifier layer to be linear")

    # Preserve the pretrained features during the first training phase.
    for parameter in model.parameters():
        parameter.requires_grad = False

    if classifier_hidden_features is None:
        # Test whether the backbone features are already linearly separable.
        model.classifier = nn.Sequential(
            nn.Linear(
                in_features=original_input_layer.in_features,
                out_features=number_of_breeds,
            ),
        )
    elif classifier_hidden_features == original_input_layer.out_features:
        # Keep MobileNetV3's pretrained 960 -> 1280 projection.
        original_dropout_layer.p = dropout_probability

        model.classifier[-1] = nn.Linear(
            in_features=original_output_layer.in_features,
            out_features=number_of_breeds,
        )
    else:
        # A different hidden size requires a completely new classifier head.
        model.classifier = nn.Sequential(
            nn.Linear(
                in_features=original_input_layer.in_features,
                out_features=classifier_hidden_features,
            ),
            nn.Hardswish(inplace=True),
            nn.Dropout(
                p=dropout_probability,
                inplace=original_dropout_layer.inplace,
            ),
            nn.Linear(
                in_features=classifier_hidden_features,
                out_features=number_of_breeds,
            ),
        )

    return model


def unfreeze_for_fine_tuning(
    model: MobileNetV3,
    number_of_feature_modules: int = 3,
) -> None:
    """Unfreeze the classifier and selected deepest feature modules."""

    feature_modules = list(model.features.children())

    if not 0 <= number_of_feature_modules <= len(feature_modules):
        raise ValueError("The number of feature modules must fit inside model.features")

    # The complete classifier is always trained during fine-tuning.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    # Zero means classifier-only; avoid the [-0:] slice, which selects everything.
    if number_of_feature_modules == 0:
        return

    for feature_module in feature_modules[-number_of_feature_modules:]:
        for parameter in feature_module.parameters():
            parameter.requires_grad = True
