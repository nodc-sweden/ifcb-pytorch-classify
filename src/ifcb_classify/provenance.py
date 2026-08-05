"""What produced a set of class scores, recorded alongside them.

Two class-scores files that disagree are unresolvable unless each one says how it
was made. ``classifier_name`` alone does not: it is derived from the checkpoint's
parent directory name, so the same weights copied into two differently named
folders produce two differently labelled outputs, and a renamed folder silently
relabels everything.

:func:`build_provenance` collects the things that actually determine a score —
the code, the libraries whose RNG and kernels it runs on, the preprocessing, and
the weights identified by content rather than by path. Writers store it verbatim;
see :func:`ifcb_classify.hdf5_output.write_class_scores`.

Deliberately no timestamp. The filesystem already records one, and keeping the
output byte-identical across identical runs makes "did this change?" answerable
with a plain file comparison — worth more here than a field that guarantees every
run differs.
"""

import hashlib
import logging
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

_DIGEST_CHUNK = 1 << 20


def build_provenance(
    transform: str,
    model_architecture: str,
    checkpoint_path: str | Path | None = None,
) -> dict[str, str]:
    """Describe the software and weights behind a classification run.

    ``transform`` should be the transform actually applied, which is not
    necessarily the one named in the checkpoint — see
    :func:`ifcb_classify.data.datasets.eval_transform_name`.
    """
    import torch
    import torchvision

    # str() every value, not just for tidiness: torch.__version__ is a
    # TorchVersion, a str subclass that numpy renders as fixed-width unicode,
    # which h5py has no conversion path for. Writers get plain str or nothing.
    provenance = {
        "ifcb_classify_version": str(_package_version()),
        "python_version": str(platform.python_version()),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "transform": str(transform),
        "model_architecture": str(model_architecture),
    }

    if checkpoint_path is not None:
        digest = checkpoint_sha256(checkpoint_path)
        if digest is not None:
            provenance["checkpoint_sha256"] = digest

    return provenance


def _package_version() -> str:
    """The version of the code that is running.

    Read from the package rather than via ``importlib.metadata``, which reports
    what was *installed*: in an editable checkout that is whatever version the
    last ``pip install`` saw, so it silently goes stale and would stamp a wrong
    number into every output file.
    """
    from ifcb_classify import __version__

    return __version__


def checkpoint_sha256(path: str | Path) -> str | None:
    """SHA256 of the checkpoint file, or None if it cannot be read.

    Identifies the weights by content, so a checkpoint copied between directories
    or renamed still matches. Returns None rather than raising: provenance is
    metadata, and failing to record it must never fail a classification run.
    """
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(_DIGEST_CHUNK):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as err:
        logger.warning("Could not hash checkpoint %s for provenance: %s", path, err)
        return None
