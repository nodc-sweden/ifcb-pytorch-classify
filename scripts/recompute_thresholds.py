"""Refit a checkpoint's per-class thresholds on a de-augmented validation split.

Thresholds fitted before the validation split stopped being augmented were
measured on randomly jittered and flipped images, so they no longer match the
operating point of the same model scored cleanly. This refits them. The weights
are untouched — only the decision thresholds change, so there is no retraining.

To tell whether a given file needs this, look for a ``"validation_transform"``
key with a non-null value in it: only releases that stopped augmenting the
validation split write that key, so a file without one predates the change and
should be refit. A thresholds JSON records no version of its own, which is why
this is the check rather than a version comparison.

The validation split is reconstructed from the checkpoint's own config (same
``data_dir``, ``val_split``, ``seed`` and class filtering), which is the only way
the refit lands on the images the model did not train on. If the training data
has changed since, the split no longer reproduces and the script refuses to
guess — see the class-list check below.

Usage:
    python recompute_thresholds.py --model path/to/weights.pt --out /tmp/refit
    python recompute_thresholds.py --model path/to/weights.pt --out /tmp/refit --device cpu

The output goes somewhere new by default, and the refit is then compared against
whatever thresholds shipped with the model. Writing into the checkpoint's own
directory instead replaces the file inference loads by default and destroys the
only record of the previous operating point, so that needs ``--in-place`` on top
of ``--out``.
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

THRESHOLDS_SUFFIX = "_thresholds_and_metrics.json"


def main() -> None:
    args = _parse_args()

    checkpoint = load_checkpoint(str(args.model), allow_unsafe=args.allow_unsafe)
    config, class_names = checkpoint["config"], checkpoint["class_names"]
    model_dir = args.model.parent
    run_name = args.run_name or args.model.stem.removesuffix("_best")
    _check_destination(args.out, model_dir, run_name, args.in_place)
    _require_training_config(config, args.model, args.data_dir)

    eval_name = _announce_transform(config["transform"])
    data = _rebuild_validation_split(config, args.data_dir, class_names)

    device = get_device(args.device)
    model = get_model(config["model"], len(class_names), config.get("pretrained", True))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    print(f"device               : {device}")
    print(f"validation images    : {len(data['val'])}")

    # Read the shipped thresholds before writing anything. With --in-place the
    # write lands on the very file being compared against, so reading afterwards
    # would compare the refit with itself and report no change at all.
    previous = _read_previous(model_dir, run_name)

    val_loader = DataLoader(data["val"], batch_size=args.batch_size, num_workers=args.num_workers)
    thresholds, class_metrics = compute_optimal_thresholds(model, val_loader, device, class_names)

    if args.in_place:
        _back_up(previous)
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
    _report_change(previous, class_names, thresholds, new_weighted)
    _report_destination(args.in_place)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, type=Path, help="Checkpoint .pt to refit thresholds for")
    p.add_argument("--out", required=True, type=Path, help="Directory to write the thresholds JSON into")
    p.add_argument("--data-dir", type=Path, help="Override the training data dir recorded in the checkpoint")
    p.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda' or 'mps'")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--run-name", help="Base name for the output JSON (default: the checkpoint's stem minus '_best')")
    p.add_argument(
        "--in-place", action="store_true",
        help="Allow --out to be the checkpoint's own directory, replacing the thresholds that shipped with it",
    )
    p.add_argument(
        "--allow-unsafe", action="store_true",
        help="Allow unsafe checkpoint loading for legacy .pt files (unpickling can execute arbitrary code)",
    )
    return p.parse_args()


def _check_destination(out: Path, model_dir: Path, run_name: str, in_place: bool) -> None:
    """Refuse to overwrite the shipped thresholds unless that was asked for.

    ``{run_name}_thresholds_and_metrics.json`` in the checkpoint's own directory
    is the first file ``infer`` looks for, so writing there is not "somewhere to
    put the output" — it installs the refit for every subsequent run and destroys
    the record of the operating point the model shipped with. Worth an explicit
    flag, especially since the numbers this script prints are what decides
    whether the refit should be installed at all.
    """
    if in_place or out.resolve() != model_dir.resolve():
        return
    raise SystemExit(
        f"--out is the checkpoint's own directory, so this would overwrite "
        f"{model_dir / (run_name + THRESHOLDS_SUFFIX)} — the file inference loads by default, and "
        "the only record of the thresholds this model shipped with. Write somewhere new and "
        "compare the numbers first, or pass --in-place if replacing them is what you want."
    )


def _require_training_config(config: dict, model_path: Path, data_dir_override: Path | None) -> None:
    """Fail early and clearly on a checkpoint that cannot describe its own split.

    Legacy bare state-dicts get a synthesised config (see
    :func:`ifcb_classify.checkpoint.load_checkpoint`) carrying only the model
    name, image size and transform. Reconstructing the validation split needs
    ``val_split`` and ``data_dir`` as well, and neither can be guessed — the
    split would silently include images the model trained on.
    """
    missing = [key for key in ("val_split", "data_dir") if key not in config]
    if data_dir_override is not None and "data_dir" in missing:
        missing.remove("data_dir")
    if not missing:
        return
    raise SystemExit(
        f"{model_path} records no {' or '.join(missing)}, which usually means it is a legacy "
        "checkpoint holding only weights. The validation split cannot be reconstructed from it, "
        "and guessing would refit on images the model trained on. Refit from a pipeline "
        "checkpoint, or retrain."
    )


def _announce_transform(transform: str) -> str:
    """Print which transform the refit will score through, and return it."""
    eval_name = eval_transform_name(transform)
    print(f"checkpoint transform : {transform}")
    if eval_name == transform:
        print("  -> no augmentation in this transform; thresholds were already fitted cleanly.")
        print("     Refitting anyway is harmless but should reproduce the existing values.")
    else:
        print(f"  -> validation will use '{eval_name}'")
    return eval_name


def _rebuild_validation_split(config: dict, data_dir_override: Path | None, class_names: list[str]) -> dict:
    """Reconstruct the split training used, or refuse if the dataset has moved on."""
    data_dir = str(data_dir_override) if data_dir_override else config["data_dir"]
    print(f"training data        : {data_dir}")

    data = create_training_datasets(
        data_dir=data_dir,
        transform_name=config["transform"],
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
    return data


def _read_previous(model_dir: Path, run_name: str) -> tuple[Path, dict] | None:
    """Load the thresholds that shipped with the checkpoint, if they can be identified.

    Resolution mirrors :func:`ifcb_classify.infer._find_thresholds_file`, so the
    comparison is against the file inference would actually load: the run name
    first, then a hand-placed ``thresholds.json``, then an unambiguous single
    match. Several unmatched candidates resolve to nothing rather than to the
    alphabetically first one — comparing against an unrelated run's thresholds
    would put a plausible but meaningless number in front of the decision this
    script exists to inform.
    """
    exact = model_dir / f"{run_name}{THRESHOLDS_SUFFIX}"
    plain = model_dir / "thresholds.json"
    candidates = sorted(model_dir.glob(f"*{THRESHOLDS_SUFFIX}"))

    if exact.is_file():
        previous = exact
    elif plain.is_file():
        previous = plain
    elif len(candidates) == 1:
        previous = candidates[0]
    else:
        if candidates:
            print(
                f"\n(found {len(candidates)} thresholds files in {model_dir} and none matches the run "
                f"name {run_name!r}; not guessing between them, so there is nothing to compare against)"
            )
        else:
            print(f"\n(no previous thresholds file in {model_dir} to compare against)")
        return None

    try:
        return previous, json.loads(previous.read_text())
    except (OSError, json.JSONDecodeError) as err:
        print(f"\n(could not read {previous.name} to compare against: {err})")
        return None


def _back_up(previous: tuple[Path, dict] | None) -> None:
    """Copy the thresholds ``--in-place`` is about to replace, keeping them recoverable.

    How far the refit moved is only printed *after* the write, so consenting to
    --in-place is not the same as having seen the numbers and accepted them. The
    ``.bak`` suffix keeps the copy out of the ``*_thresholds_and_metrics.json``
    glob that inference resolves against, so it cannot be loaded by accident.
    """
    if previous is None:
        return
    path, _ = previous
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(path.read_text())
    print(f"backed up {path.name} -> {backup.name}")


def _report_change(previous: tuple[Path, dict] | None, class_names, thresholds, new_weighted) -> None:
    """Print how far the refit moved things, against whatever thresholds shipped."""
    if previous is None:
        return
    path, old = previous

    old_metrics = old.get("class_metrics", {})
    old_values = np.array([old_metrics.get(c, {}).get("threshold", np.nan) for c in class_names], dtype=np.float64)
    delta = np.abs(old_values - thresholds)

    print(f"\ncompared with {path.name}:")
    if "weighted_F1" in old:
        print(f"  weighted F1 {old['weighted_F1']:.4f} -> {new_weighted:.4f}")
    if np.isnan(delta).all():
        # Every class missing from the old file: it describes a different model,
        # so a summary statistic over nothing would only mislead.
        print(f"  none of the {len(class_names)} classes appear in it, so the thresholds are not comparable")
        return
    print(f"  thresholds changed by: median {np.nanmedian(delta):.4f}, max {np.nanmax(delta):.4f}")
    moved = int(np.sum(delta > 0.05))
    print(f"  {moved}/{len(class_names)} classes moved by more than 0.05")


def _report_destination(in_place: bool) -> None:
    """Say plainly whether the refit is now live, since that turns on --in-place."""
    if in_place:
        print("\nThis replaced the thresholds in the checkpoint's directory, so inference will")
        print("load the refitted values from here on.")
    else:
        print("\nThis file is NOT installed automatically. Point inference at it with")
        print("--thresholds, or copy it over the model directory's thresholds file once")
        print("you are happy with the numbers above.")


if __name__ == "__main__":
    main()
