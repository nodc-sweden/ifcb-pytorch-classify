from collections import namedtuple
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ifcb_classify.config import TrainConfig
from ifcb_classify.train import _train_epoch, train_main

FIXTURES = Path(__file__).parent / "fixtures" / "training_data"


@pytest.mark.slow
def test_train_one_epoch(tmp_path):
    config = TrainConfig(
        data_dir=str(FIXTURES),
        model="resnet18",
        transform="dataset_squarepad",
        image_width=32,
        image_height=32,
        epochs=1,
        batch_size=8,
        lr=0.01,
        output_dir=str(tmp_path),
        tracker="csv",
        val_split=0.3,
        num_workers=0,
    )
    train_main(config)

    # Check that a checkpoint was saved
    checkpoints = list(tmp_path.glob("*.pt"))
    assert len(checkpoints) == 1

    # Check that CSV metrics were written
    csvs = list(tmp_path.glob("*.csv"))
    assert len(csvs) >= 1

    # Check that thresholds JSON was written
    jsons = list(tmp_path.glob("*thresholds*.json"))
    assert len(jsons) == 1

    # Check that classes.txt was written
    classes_files = list(tmp_path.glob("*classes.txt"))
    assert len(classes_files) == 1


# --- auxiliary-classifier outputs -------------------------------------------

GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
InceptionOutputs = namedtuple("InceptionOutputs", ["logits", "aux_logits"])


class _AuxModel(nn.Module):
    """Stands in for googlenet/inception_v3, which return a namedtuple in train mode."""

    def __init__(self, wrapper):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self._wrapper = wrapper

    def forward(self, x):
        logits = self.fc(x)
        if not self.training:
            return logits
        return self._wrapper(*([logits] * len(self._wrapper._fields)))


@pytest.mark.parametrize("wrapper", [GoogLeNetOutputs, InceptionOutputs])
def test_train_epoch_unwraps_auxiliary_outputs(wrapper):
    """Aux heads survive ``weights=None``, so from-scratch runs must not crash.

    The old code unwrapped only when the model was literally named
    ``inception_v3``, so ``googlenet`` and the ``inception_v3_untrained`` alias
    both died on the loss with ``must be Tensor, not GoogLeNetOutputs``.
    """
    model = _AuxModel(wrapper)
    batch = [(torch.rand(4, 4), torch.tensor([0, 1, 0, 1]))]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss, accuracy = _train_epoch(model, batch, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"))

    assert loss > 0
    assert 0.0 <= accuracy <= 1.0


def test_train_epoch_handles_plain_tensor_output():
    model = nn.Linear(4, 2)
    batch = [(torch.rand(4, 4), torch.tensor([0, 1, 0, 1]))]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss, accuracy = _train_epoch(model, batch, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"))

    assert loss > 0
    assert 0.0 <= accuracy <= 1.0
