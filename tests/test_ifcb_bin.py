from pathlib import Path

import pytest
from PIL import Image

from ifcb_classify.data.ifcb_bin import find_headerless_bins, get_bin_lid, iter_bin_images, iter_directory_bins

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


def _copy_bin(dest, *suffixes):
    """Copy the fixture bin's chosen fileset members into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        src = BIN_PATH.with_suffix(suffix)
        (dest / src.name).write_bytes(src.read_bytes())
    return dest


def test_find_headerless_bins_flags_filesets_without_a_header(tmp_path):
    """ifcbkit's discovery needs a .hdr; anything it drops must be reportable.

    Without this, a .roi/.adc pair is enumerated as pending work by one code
    path and never yielded by the other, so it is skipped in silence.
    """
    _copy_bin(tmp_path, ".roi", ".adc")

    assert list(iter_directory_bins(tmp_path)) == []
    assert [p.name for p in find_headerless_bins(tmp_path)] == ["D20220519T124533_IFCB134.roi"]


def test_find_headerless_bins_empty_for_complete_filesets():
    assert find_headerless_bins(FIXTURES / "bins") == []


def test_find_headerless_bins_empty_for_directory_without_bins(tmp_path):
    assert find_headerless_bins(tmp_path) == []


@pytest.mark.parametrize("folder", ["beads", "skip", "some_cruise_folder"])
def test_find_headerless_bins_ignores_deliberately_excluded_layouts(tmp_path, folder):
    """Only the missing-header case is reported, not every undiscovered bin.

    Discovery also skips beads/skip paths and any layout other than flat or
    ``Dyyyy/Dyyyymmdd/`` — the same filtering pyifcb applied. Reporting those
    would warn about every bin of a differently-organised archive on every run.
    """
    _copy_bin(tmp_path / folder, ".roi", ".adc", ".hdr")

    assert list(iter_directory_bins(tmp_path)) == []
    assert find_headerless_bins(tmp_path) == []


def test_find_headerless_bins_ignores_a_lone_roi(tmp_path):
    """A .roi with no .adc is not a usable bin, so it is not worth warning about."""
    _copy_bin(tmp_path, ".roi")

    assert find_headerless_bins(tmp_path) == []
