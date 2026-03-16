import torch

from ifcb_classify.device import get_device


def test_force_cpu():
    assert get_device("cpu") == torch.device("cpu")


def test_force_cuda_string():
    assert get_device("cuda") == torch.device("cuda")


def test_auto_returns_device():
    device = get_device("auto")
    assert device.type in ("cpu", "cuda", "mps")


def test_passthrough_device_string():
    assert get_device("cpu") == torch.device("cpu")
