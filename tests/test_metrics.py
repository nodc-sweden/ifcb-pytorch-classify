import torch

from ifcb_classify.metrics import MetricsCalculator


def test_metrics_calculator():
    calc = MetricsCalculator(num_classes=3)

    preds = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    labels = torch.tensor([0, 1, 2])
    calc.update(preds, labels)

    results = calc.compute()
    assert results.accuracy == 1.0
    assert results.weighted_f1 > 0.9
    assert results.confusion_matrix.shape == (3, 3)

    calc.reset()

    # After reset, feed new data to verify it works across epochs
    preds2 = torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.9, 0.05, 0.05]])
    labels2 = torch.tensor([1, 2, 0])
    calc.update(preds2, labels2)
    results2 = calc.compute()
    assert results2.accuracy == 1.0
