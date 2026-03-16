import numpy as np
import torch

from ifcb_classify.seed import set_seed


def test_set_seed_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    t = torch.rand(5)

    set_seed(123)
    b = np.random.rand(5)
    u = torch.rand(5)

    np.testing.assert_array_equal(a, b)
    assert torch.equal(t, u)


def test_set_seed_different_seeds_differ():
    set_seed(1)
    a = np.random.rand(5)

    set_seed(2)
    b = np.random.rand(5)

    assert not np.array_equal(a, b)
