import re

import torch.nn as nn

from ifcb_classify.models.registry import MODELS


def _set_head(model: nn.Module, path: str, layer: nn.Module) -> None:
    """Set a nested attribute/index on a model using dot/bracket path notation.

    Supports paths like "fc", "classifier[6]", "heads[0]".
    """
    parts = re.split(r"\.", path)
    obj = model
    for part in parts[:-1]:
        obj = _resolve_part(obj, part)
    _assign_part(obj, parts[-1], layer)


def _resolve_part(obj, part: str):
    match = re.match(r"(\w+)\[(\d+)]", part)
    if match:
        attr, idx = match.group(1), int(match.group(2))
        return getattr(obj, attr)[idx]
    return getattr(obj, part)


def _assign_part(obj, part: str, value: nn.Module) -> None:
    match = re.match(r"(\w+)\[(\d+)]", part)
    if match:
        attr, idx = match.group(1), int(match.group(2))
        getattr(obj, attr)[idx] = value
    else:
        setattr(obj, part, value)


def get_model(name: str, num_classes: int) -> nn.Module:
    if name == "custom":
        return _build_custom(num_classes)

    spec = MODELS.get(name)
    if spec is None:
        raise ValueError(f"Unknown model: {name}. Available: {sorted(MODELS.keys())}")

    weights_arg = {"weights": spec.weights} if spec.weights else {"weights": None}
    model = spec.constructor(**weights_arg)

    head = nn.Linear(in_features=spec.in_features, out_features=num_classes, bias=spec.bias)
    _set_head(model, spec.head_path, head)

    return model


def _build_custom(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(6),
        nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(start_dim=1),
        nn.Linear(in_features=12 * 4 * 4, out_features=120),
        nn.ReLU(),
        nn.BatchNorm1d(120),
        nn.Linear(in_features=120, out_features=60),
        nn.ReLU(),
        nn.Linear(in_features=60, out_features=num_classes),
    )
