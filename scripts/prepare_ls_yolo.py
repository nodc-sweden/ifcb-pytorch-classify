"""Turn a Label Studio YOLO export into a chains-train dataset.

Label Studio exports a flat set (images/ + labels/ + classes.txt). This pairs
labels to images by filename, splits train/val, and writes a data.yaml ready
for `ifcb-classify chains-train --data <out>`.

Usage:
    python prepare_ls_yolo.py --labels export/labels --images export/images \
        --out datasets/thalassionema --class-name thalassionema_nitzschioides

If the export's images/ is empty (common with synced storage), point --images
at the original image folder instead; pairing is by filename stem.
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        for cand in (images_dir / f"{stem}{ext}", images_dir / f"{stem}{ext.upper()}"):
            if cand.exists():
                return cand
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="Directory of YOLO .txt label files")
    p.add_argument("--images", required=True, help="Directory of source images")
    p.add_argument("--out", required=True, help="Output dataset directory")
    p.add_argument("--class-name", required=True, help="Taxon name (single class)")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symlink", action="store_true", help="Symlink instead of copy")
    args = p.parse_args()

    labels_dir = Path(args.labels)
    images_dir = Path(args.images)
    out = Path(args.out)

    label_files = sorted(labels_dir.glob("*.txt"))
    label_files = [f for f in label_files if f.stem.lower() != "classes"]
    if not label_files:
        raise SystemExit(f"No .txt label files found in {labels_dir}")

    pairs = []
    missing = []
    for lf in label_files:
        img = _find_image(images_dir, lf.stem)
        if img is None:
            missing.append(lf.stem)
        else:
            pairs.append((img, lf))

    if missing:
        print(f"WARNING: {len(missing)} labels had no matching image (skipped), e.g. {missing[:3]}")
    if not pairs:
        raise SystemExit("No label/image pairs found — check --images path and filenames")

    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, round(len(pairs) * args.val_split)) if len(pairs) > 1 else 0
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    # Wipe any previous contents so a re-run with a changed export (or a flipped
    # train/val split) can't leave stale files behind or duplicate an image into
    # both splits.
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)

    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lf in items:
            _place(img, out / "images" / split / img.name, args.symlink)
            _place(lf, out / "labels" / split / lf.name, args.symlink)

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: {args.class_name}\n"
    )

    print(f"Dataset written to {out}")
    print(f"  train: {len(splits['train'])}  val: {len(splits['val'])}")
    print(f"  data.yaml: {data_yaml}")


def _place(src: Path, dst: Path, symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
