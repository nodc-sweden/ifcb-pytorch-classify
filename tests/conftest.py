import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def tiny_imagefolder(tmp_dir):
    """Create a minimal ImageFolder dataset with 3 classes, 5 images each."""
    classes = ["ClassA", "ClassB", "ClassC"]
    for cls in classes:
        cls_dir = tmp_dir / cls
        cls_dir.mkdir()
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8))
            img.save(cls_dir / f"img_{i}.png")
    return tmp_dir, classes
