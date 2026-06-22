"""Inference-time cell counting with per-taxon YOLO detectors.

:class:`ChainCounter` is constructed once per inference run from a
:class:`ChainCountingConfig` and used by :mod:`ifcb_classify.infer` to count
cells in ROIs whose (thresholded) class is configured for counting. Detectors
are loaded lazily and cached, and ``ultralytics`` is imported only on first use,
so the core classifier never pays for the chains dependency unless counting
actually runs.
"""

import logging
from pathlib import Path

from ifcb_classify.chains.config import ChainCountingConfig

logger = logging.getLogger(__name__)

_CHAINS_EXTRA_HINT = (
    "Chain counting requires the optional 'chains' extra. "
    'Install it with: uv pip install -e ".[chains]"'
)


class ChainCounter:
    """Counts cells in chain-forming ROIs using per-taxon YOLO detectors.

    Models are loaded lazily on first use and cached, so only detectors for
    classes actually encountered are ever loaded into memory.
    """

    def __init__(self, config: ChainCountingConfig):
        """Validate up front that every configured weights file exists.

        Failing here (rather than lazily on first use) means a misconfigured path
        is reported before any bins are processed. Raises ``FileNotFoundError``
        listing the missing detectors.
        """
        self._config = config
        self._models: dict[str, object] = {}

        missing = [name for name, spec in config.models.items() if not Path(spec.weights).exists()]
        if missing:
            details = ", ".join(f"{name} -> {config.models[name].weights}" for name in missing)
            raise FileNotFoundError(f"Chain detector weights not found: {details}")

    def handles(self, class_name: str) -> bool:
        """Whether a classifier label is configured for counting."""
        return class_name in self._config.models

    def models_metadata(self) -> dict:
        """Provenance mapping (class -> {weights, iou, conf}) for HDF5 output."""
        return {
            name: {"weights": spec.weights, "iou": spec.iou, "conf": spec.conf}
            for name, spec in self._config.models.items()
        }

    def _get_model(self, class_name: str):
        """Return the cached YOLO model for ``class_name``, loading it on first use.

        Imports ``ultralytics`` lazily and raises ``ImportError`` with an install
        hint if the ``chains`` extra is missing.
        """
        if class_name not in self._models:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(_CHAINS_EXTRA_HINT) from exc
            weights = self._config.models[class_name].weights
            logger.info("Loading chain detector for %s: %s", class_name, weights)
            self._models[class_name] = YOLO(weights)
        return self._models[class_name]

    def count(self, image, class_name: str) -> int:
        """Count cells in a single ROI image for a counted class.

        ``image`` is an RGB PIL image or numpy array. Raises KeyError if the
        class is not configured (callers should gate on ``handles`` first).
        """
        spec = self._config.models[class_name]
        model = self._get_model(class_name)
        results = model(image, iou=spec.iou, conf=spec.conf, verbose=False)
        return int(sum(len(r.boxes) for r in results))
