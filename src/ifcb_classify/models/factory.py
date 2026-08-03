"""Model instantiation: build a torchvision backbone with a fresh classifier head.

:func:`get_model` looks a name up in the :data:`MODELS` registry, constructs the
pretrained backbone, and swaps its final layer for a new ``nn.Linear`` sized to
``num_classes``. Because different architectures name their head differently
(``fc``, ``classifier[6]``, ``heads[0]`` …), the registry stores a *path string*
and the small ``_set_head`` helper resolves and replaces it. The special name
``custom`` builds a tiny from-scratch CNN instead.
"""

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
    """Read one path segment off ``obj`` — attribute, or ``attr[idx]`` for a list."""
    match = re.match(r"(\w+)\[(\d+)]", part)
    if match:
        attr, idx = match.group(1), int(match.group(2))
        return getattr(obj, attr)[idx]
    return getattr(obj, part)


def _assign_part(obj, part: str, value: nn.Module) -> None:
    """Assign ``value`` to one path segment of ``obj`` (attribute or ``attr[idx]``)."""
    match = re.match(r"(\w+)\[(\d+)]", part)
    if match:
        attr, idx = match.group(1), int(match.group(2))
        getattr(obj, attr)[idx] = value
    else:
        setattr(obj, part, value)


def get_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build the model ``name`` with a ``num_classes``-wide classification head.

    Looks ``name`` up in :data:`MODELS` (case-insensitively), instantiates the
    backbone with its default pretrained weights, and replaces the head. Pass
    ``pretrained=False`` to train from scratch instead; a spec that already sets
    ``weights=None`` (such as ``inception_v3_untrained``) trains from scratch
    either way. ``name="custom"`` returns the bundled small CNN, which is always
    from scratch. Raises ``ValueError`` for an unknown name.
    """
    if name == "custom":
        return _build_custom(num_classes)

    spec = MODELS.get(name) or MODELS.get(name.lower())
    if spec is None:
        raise ValueError(f"Unknown model: {name}. Available: {sorted(MODELS.keys())}")

    weights_arg = {"weights": spec.weights if pretrained else None}
    model = spec.constructor(**weights_arg)

    head = nn.Linear(in_features=spec.in_features, out_features=num_classes, bias=spec.bias)
    _set_head(model, spec.head_path, head)

    return model


def _build_custom(num_classes: int) -> nn.Module:
    """Build the bundled small from-scratch CNN (a LeNet-style baseline)."""
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
