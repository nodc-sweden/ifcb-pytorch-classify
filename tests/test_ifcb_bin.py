from pathlib import Path

from PIL import Image

from ifcb_classify.data.ifcb_bin import get_bin_lid, iter_bin_images

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
