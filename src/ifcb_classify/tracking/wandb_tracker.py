import numpy as np


class WandbTracker:
    def __init__(self, project: str = "ifcb-classify"):
        import wandb

        self._wandb = wandb
        self._project = project

    def begin_run(self, run_name: str, params: dict) -> None:
        self._wandb.init(project=self._project, name=run_name, config=params)

    def log_metrics(self, metrics: dict, step: int) -> None:
        self._wandb.log(metrics, step=step)

    def log_confusion_matrix(self, cm: np.ndarray, class_names: list[str], step: int) -> None:
        y_true = []
        y_pred = []
        for true_idx in range(cm.shape[0]):
            for pred_idx in range(cm.shape[1]):
                count = int(cm[true_idx, pred_idx])
                y_true.extend([true_idx] * count)
                y_pred.extend([pred_idx] * count)

        self._wandb.log({
            f"confusion_matrix_epoch_{step}": self._wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names,
            )
        })

    def end_run(self) -> None:
        self._wandb.finish()
