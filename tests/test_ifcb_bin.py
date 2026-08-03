from pathlib import Path

import pytest
from PIL import Image

from ifcb_classify.data.ifcb_bin import get_bin_lid, iter_bin_images, iter_directory_bins

FIXTURES = Path(__file__).parent / "fixtures"
BIN_PATH = FIXTURES / "bins" / "D20220519T124533_IFCB134.roi"


def test_get_bin_lid():
    assert get_bin_lid("/data/D20220519T124533_IFCB134.roi") == "D20220519T124533_IFCB134"
    assert get_bin_lid("D20220519T124533_IFCB134.adc") == "D20220519T124533_IFCB134"


def test_get_bin_lid_from_fixture():
    assert get_bin_lid(BIN_PATH) == "D20220519T124533_IFCB134"


def test_iter_bin_images():
    images = list(iter_bin_images(BIN_PATH))
    assert len(images) > 0
    target_num, img = images[0]
    assert isinstance(target_num, int)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_iter_bin_images_targets_ascending():
    targets = [t for t, _ in iter_bin_images(BIN_PATH)]
    assert targets == sorted(targets)


def test_iter_bin_images_accepts_any_fileset_extension():
    from_roi = [t for t, _ in iter_bin_images(BIN_PATH)]
    from_adc = [t for t, _ in iter_bin_images(BIN_PATH.with_suffix(".adc"))]
    assert from_roi == from_adc


def test_iter_bin_images_rejects_incomplete_fileset(tmp_path):
    lone = tmp_path / "D20220519T124533_IFCB134.roi"
    lone.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="Incomplete fileset"):
        list(iter_bin_images(lone))


def test_iter_directory_bins():
    bins = list(iter_directory_bins(FIXTURES / "bins"))
    assert [lid for lid, _ in bins] == ["D20220519T124533_IFCB134"]

    _, files = bins[0]
    assert files.adc_path.is_file()
    assert files.roi_path.is_file()


def test_iter_directory_bins_handles_round_trip():
    """The handle from iter_directory_bins feeds straight into iter_bin_images."""
    _, files = next(iter(iter_directory_bins(FIXTURES / "bins")))
    assert [t for t, _ in iter_bin_images(files)] == [t for t, _ in iter_bin_images(BIN_PATH)]
