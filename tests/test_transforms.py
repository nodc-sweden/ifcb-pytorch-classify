import torch

from ifcb_classify.data.transforms import SquarePad, FullPad, ReflectPad


def test_squarepad_makes_square():
    image = torch.rand(3, 50, 100)  # C, H, W — wide image
    padded = SquarePad()(image)
    assert padded.size(-1) == padded.size(-2)


def test_fullpad_reaches_target():
    image = torch.rand(3, 50, 80)
    padded = FullPad(224, 224)(image)
    assert padded.size(-1) >= 224 or padded.size(-2) >= 224


def test_fullpad_returns_large_image_unchanged():
    image = torch.rand(3, 300, 300)
    padded = FullPad(224, 224)(image)
    assert padded.size() == image.size()


def test_reflectpad_reaches_target():
    image = torch.rand(3, 50, 80)
    padded = ReflectPad(224, 224)(image)
    assert padded.size(-1) == 224
    assert padded.size(-2) == 224


def test_reflectpad_returns_large_image_unchanged():
    image = torch.rand(3, 300, 300)
    padded = ReflectPad(224, 224)(image)
    assert padded.size() == image.size()


def test_reflectpad_preserves_channels():
    image = torch.rand(3, 40, 60)
    padded = ReflectPad(128, 128)(image)
    assert padded.size(0) == 3
