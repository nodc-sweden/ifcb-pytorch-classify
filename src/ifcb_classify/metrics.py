from dataclasses import dataclass

import torch
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassAUPRC,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


@dataclass(frozen=True)
class MetricsResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    weighted_f1: float
    auprc: float
    auroc: float
    confusion_matrix: torch.Tensor


class MetricsCalculator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self._accuracy = MulticlassAccuracy(average="micro", num_classes=num_classes)
        self._precision = MulticlassPrecision(num_classes=num_classes)
        self._recall = MulticlassRecall(num_classes=num_classes)
        self._f1 = MulticlassF1Score(average="macro", num_classes=num_classes)
        self._f1_weighted = MulticlassF1Score(average="weighted", num_classes=num_classes)
        self._auprc = MulticlassAUPRC(num_classes=num_classes)
        self._auroc = MulticlassAUROC(num_classes=num_classes)
        self._confusion = MulticlassConfusionMatrix(num_classes=num_classes)

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        pred_classes = preds.argmax(dim=1)
        self._accuracy.update(pred_classes, labels)
        self._precision.update(pred_classes, labels)
        self._recall.update(pred_classes, labels)
        self._f1.update(pred_classes, labels)
        self._f1_weighted.update(pred_classes, labels)
        self._confusion.update(pred_classes, labels)
        self._auprc.update(preds, labels)
        self._auroc.update(preds, labels)

    def compute(self) -> MetricsResult:
        return MetricsResult(
            accuracy=self._accuracy.compute().item(),
            precision=self._precision.compute().item(),
            recall=self._recall.compute().item(),
            f1=self._f1.compute().item(),
            weighted_f1=self._f1_weighted.compute().item(),
            auprc=self._auprc.compute().item(),
            auroc=self._auroc.compute().item(),
            confusion_matrix=self._confusion.compute(),
        )

    def reset(self) -> None:
        self._accuracy.reset()
        self._precision.reset()
        self._recall.reset()
        self._f1.reset()
        self._f1_weighted.reset()
        self._auprc.reset()
        self._auroc.reset()
        self._confusion.reset()
