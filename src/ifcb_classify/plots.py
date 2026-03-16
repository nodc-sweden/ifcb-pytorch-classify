"""Evaluation plots generated after training.

Static plots use matplotlib (always available via scikit-learn).
Interactive plots use plotly (optional dependency).
"""

import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

_HAS_PLOTLY = False
try:
    import plotly.graph_objects as go  # noqa: F401

    _HAS_PLOTLY = True
except ImportError:
    pass


def _truncate(name: str, max_len: int = 40) -> str:
    """Shorten a label for static plots, keeping start and end."""
    if len(name) <= max_len:
        return name
    keep = (max_len - 3) // 2
    return f"{name[:keep]}...{name[-keep:]}"


def generate_evaluation_plots(
    output_dir: str | Path,
    run_name: str,
    epoch_metrics: list[dict],
    confusion_matrix: np.ndarray,
    class_names: list[str],
    class_metrics: dict | None = None,
) -> list[Path]:
    """Generate all evaluation plots and return paths to created files.

    Parameters
    ----------
    output_dir : path to output directory
    run_name : name of the training run
    epoch_metrics : list of per-epoch metric dicts (keys: train_loss, val_loss,
        train_accuracy, val_accuracy, f1, weighted_f1, etc.)
    confusion_matrix : (num_classes, num_classes) array
    class_names : ordered list of class names
    class_metrics : per-class metrics dict from compute_optimal_thresholds,
        keyed by class name with f1/precision/recall/support/threshold values.
        If None, per-class plots are skipped.
    """
    plots_dir = Path(output_dir) / "plots" / run_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    created.extend(_plot_training_curves(plots_dir, epoch_metrics))
    created.extend(_plot_top_confused_pairs(plots_dir, confusion_matrix, class_names))

    if class_metrics:
        created.extend(_plot_per_class_f1_bar(plots_dir, class_metrics))
        created.extend(_plot_precision_recall_scatter(plots_dir, class_metrics))
        created.extend(_plot_class_support_histogram(plots_dir, class_metrics))

    if _HAS_PLOTLY:
        created.extend(_plot_interactive_confusion_matrix(plots_dir, confusion_matrix, class_names))
        if class_metrics:
            created.extend(_plot_interactive_metrics_table(plots_dir, class_metrics))
    else:
        logger.info("Install plotly for interactive plots: pip install 'ifcb-classify[plots]'")

    logger.info("Saved %d evaluation plots to %s", len(created), plots_dir)
    return created


# ---------------------------------------------------------------------------
# Static plots (matplotlib)
# ---------------------------------------------------------------------------


def _plot_training_curves(plots_dir: Path, epoch_metrics: list[dict]) -> list[Path]:
    if not epoch_metrics:
        return []

    epochs = list(range(1, len(epoch_metrics) + 1))
    created: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    train_loss = [m["train_loss"] for m in epoch_metrics]
    val_loss = [m["val_loss"] for m in epoch_metrics]
    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    train_acc = [m["train_accuracy"] for m in epoch_metrics]
    val_acc = [m["val_accuracy"] for m in epoch_metrics]
    axes[1].plot(epochs, train_acc, label="Train")
    axes[1].plot(epochs, val_acc, label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    path = plots_dir / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path)

    return created


def _plot_per_class_f1_bar(
    plots_dir: Path,
    class_metrics: dict,
    tail_n: int = 20,
) -> list[Path]:
    sorted_items = sorted(class_metrics.items(), key=lambda x: x[1]["f1"])
    num_classes = len(sorted_items)

    # For small class counts, show all; for large, show worst + best tails
    if num_classes <= tail_n * 2:
        items_to_plot = sorted_items
        title = "Per-class F1 (sorted)"
    else:
        worst = sorted_items[:tail_n]
        best = sorted_items[-tail_n:]
        items_to_plot = worst + best
        title = f"Per-class F1 — worst {tail_n} and best {tail_n} of {num_classes}"

    names = [_truncate(name) for name, _ in items_to_plot]
    f1_scores = [m["f1"] for _, m in items_to_plot]

    show_n = len(names)
    fig_height = max(4, show_n * 0.28)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    y_pos = np.arange(show_n)

    ax.barh(y_pos, f1_scores, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)

    # Draw a separator line between worst and best groups
    if num_classes > tail_n * 2:
        ax.axhline(y=tail_n - 0.5, color="gray", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    path = plots_dir / "per_class_f1.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def _plot_precision_recall_scatter(plots_dir: Path, class_metrics: dict) -> list[Path]:
    precisions = []
    recalls = []
    supports = []
    for m in class_metrics.values():
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        supports.append(m["support"])

    supports_arr = np.array(supports, dtype=float)
    max_support = supports_arr.max() if supports_arr.max() > 0 else 1.0
    sizes = 10 + 200 * (supports_arr / max_support)

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(precisions, recalls, s=sizes, alpha=0.5, c=supports_arr, cmap="viridis")
    fig.colorbar(scatter, ax=ax, label="Support (sample count)")
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Per-class Precision vs. Recall")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.2)

    fig.tight_layout()
    path = plots_dir / "precision_recall_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def _plot_class_support_histogram(plots_dir: Path, class_metrics: dict) -> list[Path]:
    supports = [m["support"] for m in class_metrics.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(supports, bins=min(50, max(10, len(supports) // 4)), edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Number of classes")
    ax.set_title("Class Support Distribution")

    fig.tight_layout()
    path = plots_dir / "class_support_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def _plot_top_confused_pairs(
    plots_dir: Path,
    confusion_matrix: np.ndarray,
    class_names: list[str],
    top_k: int = 25,
) -> list[Path]:
    # Row-normalize so that imbalanced classes are compared fairly
    row_sums = confusion_matrix.sum(axis=1, keepdims=True).astype(float)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    cm_rate = confusion_matrix.astype(float) / row_sums
    np.fill_diagonal(cm_rate, 0)

    cm_raw = confusion_matrix.copy().astype(float)
    np.fill_diagonal(cm_raw, 0)

    num_pairs = min(top_k, int(cm_rate.size - cm_rate.shape[0]))
    if num_pairs <= 0:
        return []

    flat_indices = np.argsort(cm_rate.ravel())[::-1][:num_pairs]
    rows, cols = np.unravel_index(flat_indices, cm_rate.shape)

    labels = []
    rates = []
    annotations = []
    for r, c in zip(rows, cols):
        rate = cm_rate[r, c]
        if rate == 0:
            break
        count = int(cm_raw[r, c])
        support = int(row_sums[r, 0])
        labels.append(f"{_truncate(class_names[r], 35)} \u2192 {_truncate(class_names[c], 35)}")
        rates.append(rate * 100)
        annotations.append(f"{count}/{support}")

    if not labels:
        return []

    labels = list(reversed(labels))
    rates = list(reversed(rates))
    annotations = list(reversed(annotations))

    fig_height = max(4, len(labels) * 0.3)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, rates, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Confusion rate (%)")
    ax.set_title(f"Top {len(labels)} Most Confused Class Pairs (true \u2192 predicted)")

    # Annotate bars with raw count/support
    for bar, annotation in zip(bars, annotations):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            annotation,
            va="center",
            fontsize=6,
            color="gray",
        )

    fig.tight_layout()
    path = plots_dir / "top_confused_pairs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------------------
# Interactive plots (plotly, optional)
# ---------------------------------------------------------------------------


def _plot_interactive_confusion_matrix(
    plots_dir: Path,
    confusion_matrix: np.ndarray,
    class_names: list[str],
) -> list[Path]:
    import plotly.graph_objects as go

    # Row-normalize: each cell becomes fraction of true class support
    row_sums = confusion_matrix.sum(axis=1, keepdims=True).astype(float)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    cm_normalized = confusion_matrix.astype(float) / row_sums

    hover_text = []
    for i, true_name in enumerate(class_names):
        row = []
        for j, pred_name in enumerate(class_names):
            pct = cm_normalized[i, j] * 100
            count = int(confusion_matrix[i, j])
            support = int(row_sums[i, 0])
            row.append(
                f"True: {true_name}<br>"
                f"Predicted: {pred_name}<br>"
                f"Rate: {pct:.1f}%<br>"
                f"Count: {count} / {support}"
            )
        hover_text.append(row)

    num_classes = len(class_names)
    size = max(800, num_classes * 5)

    # Use integer indices for axes; full names only appear on hover
    indices = list(range(num_classes))

    # Normalized view (default) and raw count view as toggle buttons
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=cm_normalized * 100,
            x=indices,
            y=indices,
            hovertext=hover_text,
            hoverinfo="text",
            colorscale="Blues",
            colorbar=dict(title="%"),
            visible=True,
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=confusion_matrix,
            x=indices,
            y=indices,
            hovertext=hover_text,
            hoverinfo="text",
            colorscale="Blues",
            colorbar=dict(title="Count"),
            visible=False,
        )
    )
    fig.update_layout(
        title="Confusion Matrix (row-normalized % — hover for class names)",
        xaxis_title="Predicted class index",
        yaxis_title="True class index",
        width=size,
        height=size,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.12,
                buttons=[
                    dict(
                        label="Normalized (%)",
                        method="update",
                        args=[
                            {"visible": [True, False]},
                            {"title": "Confusion Matrix (row-normalized % — hover for class names)"},
                        ],
                    ),
                    dict(
                        label="Raw counts",
                        method="update",
                        args=[
                            {"visible": [False, True]},
                            {"title": "Confusion Matrix (raw counts — hover for class names)"},
                        ],
                    ),
                ],
            )
        ],
    )

    path = plots_dir / "confusion_matrix_interactive.html"
    fig.write_html(str(path))
    return [path]


def _plot_interactive_metrics_table(plots_dir: Path, class_metrics: dict) -> list[Path]:
    import plotly.graph_objects as go

    sorted_metrics = sorted(class_metrics.values(), key=lambda m: m["f1"], reverse=True)
    names = [m["class_name"] for m in sorted_metrics]
    f1s = [round(m["f1"], 4) for m in sorted_metrics]
    precisions = [round(m["precision"], 4) for m in sorted_metrics]
    recalls = [round(m["recall"], 4) for m in sorted_metrics]
    supports = [m["support"] for m in sorted_metrics]
    thresholds = [round(m["threshold"], 4) for m in sorted_metrics]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Class", "F1", "Precision", "Recall", "Support", "Threshold"],
                    fill_color="steelblue",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[names, f1s, precisions, recalls, supports, thresholds],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(title="Per-class Metrics (sortable)")

    path = plots_dir / "per_class_metrics_table.html"
    fig.write_html(str(path))
    return [path]
