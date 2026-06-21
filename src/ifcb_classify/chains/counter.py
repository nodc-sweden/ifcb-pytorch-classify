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
