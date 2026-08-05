"""Refit a checkpoint's per-class thresholds on a de-augmented validation split.

Thresholds fitted before the validation split stopped being augmented were
measured on randomly jittered and flipped images, so they no longer match the
operating point of the same model scored cleanly. This refits them. The weights
are untouched — only the decision thresholds change, so there is no retraining.

To tell whether a given file needs this, look for a ``"validation_transform"``
key in it: only releases that stopped augmenting the validation split write that
key, so a file without one predates the change and should be refit. A thresholds
JSON records no version of its own, which is why this is the check rather than a
version comparison.

The validation split is reconstructed from the checkpoint's own config (same
``data_dir``, ``val_split``, ``seed`` and class filtering), which is the only way
the refit lands on the images the model did not train on. If the training data
has changed since, the split no longer reproduces and the script refuses to
guess — see the class-list check below.

Usage:
    python recompute_thresholds.py --model path/to/weights.pt --out path/to/model_dir
    python recompute_thresholds.py --model path/to/weights.pt --out /tmp/x --device cpu
"""
import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from ifcb_classify.checkpoint import load_checkpoint
from ifcb_classify.data.datasets import create_training_datasets, eval_transform_name
from ifcb_classify.device import get_device
from ifcb_classify.models.factory import get_model
from ifcb_classify.thresholds import compute_optimal_thresholds, save_thresholds_and_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, type=Path, help="Checkpoint .pt to refit thresholds for")
    p.add_argument("--out", required=True, type=Path, help="Directory to write the thresholds JSON into")
    p.add_argument("--data-dir", type=Path, help="Override the training data dir recorded in the checkpoint")
    p.add_argument("--device", default="auto", help="'auto', 'cpu' or 'cuda'")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--run-name", help="Base name for the output JSON (default: the checkpoint's stem minus '_best')")
    args = p.parse_args()

    checkpoint = load_checkpoint(str(args.model), allow_unsafe=True)
    config, class_names = checkpoint["config"], checkpoint["class_names"]

    transform = config["transform"]
    eval_name = eval_transform_name(transform)
    print(f"checkpoint transform : {transform}")
    if eval_name == transform:
        print("  -> no augmentation in this transform; thresholds were already fitted cleanly.")
        print("     Refitting anyway is harmless but should reproduce the existing values.")
    else:
        print(f"  -> validation will use '{eval_name}'")

    data_dir = str(args.data_dir) if args.data_dir else config["data_dir"]
    print(f"training data        : {data_dir}")

    data = create_training_datasets(
        data_dir=data_dir,
        transform_name=transform,
        width=config["image_width"],
        height=config["image_height"],
        val_split=config["val_split"],
        mean=config.get("mean"),
        std=config.get("std"),
        seed=config.get("seed", 42),
        min_class_images=config.get("min_class_images"),
        manual_include_classes=config.get("manual_include_classes"),
    )

    # The split is reproducible only if the dataset is byte-for-byte what training
    # saw. A changed class list is the loud symptom; a changed image count within
    # the same classes is the quiet one, and would silently move images across the
    # train/val boundary — refitting on images the model trained on would produce
    # optimistic thresholds. The class check catches the common case.
    if data["class_names"] != class_names:
        raise SystemExit(
            f"Class list has changed since training ({len(data['class_names'])} classes in "
            f"{data_dir}, {len(class_names)} in the checkpoint). The validation split cannot "
            "be reproduced, so refitting here would use images the model trained on. Point "
            "--data-dir at the dataset this model was trained on."
        )

    device = get_device(args.device)
    model = get_model(config["model"], len(class_names), config.get("pretrained", True))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    print(f"device               : {device}")
    print(f"validation images    : {len(data['val'])}")

    val_loader = DataLoader(data["val"], batch_size=args.batch_size, num_workers=args.num_workers)
    thresholds, class_metrics = compute_optimal_thresholds(model, val_loader, device, class_names)

    run_name = args.run_name or args.model.stem.removesuffix("_best")
    json_path = save_thresholds_and_metrics(
        args.out, run_name, config.get("best_epoch", checkpoint.get("epoch", 0)),
        class_names, thresholds, class_metrics,
        validation_transform=eval_name,
    )

    new_weighted = float(np.average(
        [m["f1"] for m in class_metrics.values()],
        weights=[m["support"] for m in class_metrics.values()],
    ))
    print(f"\nwrote {json_path}")
    print(f"weighted F1 (clean validation): {new_weighted:.4f}")
    _report_change(args.model.parent, class_names, thresholds, new_weighted)


def _report_change(model_dir: Path, class_names, thresholds, new_weighted) -> None:
    """Print how far the refit moved things, against whatever thresholds shipped."""
    previous = None
    for candidate in (*sorted(model_dir.glob("*_thresholds_and_metrics.json")), model_dir / "thresholds.json"):
        if candidate.is_file():
            previous = candidate
            break
    if previous is None:
        print("(no previous thresholds file found to compare against)")
        return

    old = json.loads(previous.read_text())
    old_metrics = old.get("class_metrics", {})
    old_values = np.array([old_metrics.get(c, {}).get("threshold", np.nan) for c in class_names], dtype=np.float64)
    delta = np.abs(old_values - thresholds)

    print(f"\ncompared with {previous.name}:")
    if "weighted_F1" in old:
        print(f"  weighted F1 {old['weighted_F1']:.4f} -> {new_weighted:.4f}")
    print(f"  thresholds changed by: median {np.nanmedian(delta):.4f}, max {np.nanmax(delta):.4f}")
    moved = int(np.nansum(delta > 0.05))
    print(f"  {moved}/{len(class_names)} classes moved by more than 0.05")
    print("\nThis file is NOT installed automatically. Point inference at it with")
    print("--thresholds, or replace the model directory's thresholds.json once you")
    print("are happy with the numbers above.")


if __name__ == "__main__":
    main()
