"""Trial existing chain detectors on a new taxon to pick a bootstrap model.

Runs every trained detector under --models-root over a small image sample, saves
annotated previews per model, and prints box-count stats so you can eyeball which
transfers best. Pick the bootstrap detector by *cell morphology*, not taxonomy.

Usage:
    python compare_bootstrap_models.py --src /path/to/new_taxon_images \
        --models-root models/chains --out /tmp/bootstrap_trial -n 24
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def sample(src: Path, n: int):
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() == ".png")
    if len(imgs) <= n:
        return imgs
    step = len(imgs) / n
    return [imgs[int(i * step)] for i in range(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path, help="Directory of images for the new taxon")
    p.add_argument("--models-root", required=True, type=Path, help="Directory of trained detectors (<name>/best.pt)")
    p.add_argument("--out", required=True, type=Path, help="Output directory for annotated previews")
    p.add_argument("-n", "--num", type=int, default=24, help="Number of images to sample")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.30)
    args = p.parse_args()

    imgs = sample(args.src, args.num)
    print(f"sample: {len(imgs)} images from {args.src.name}\n")
    models = sorted(d.name for d in args.models_root.iterdir() if (d / "best.pt").exists())
    if not models:
        raise SystemExit(f"No detectors (<name>/best.pt) found under {args.models_root}")

    rows = []
    for name in models:
        model = YOLO(str(args.models_root / name / "best.pt"))
        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        nonzero = 0
        for img in imgs:
            r = model(str(img), imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
            n_boxes = len(r.boxes)
            total += n_boxes
            nonzero += n_boxes > 0
            cv2.imwrite(str(out_dir / img.name), r.plot())
        rows.append((name, total / len(imgs), 100 * nonzero / len(imgs)))
        print(f"  {name:32s} mean {total/len(imgs):4.2f} cells/img   {100*nonzero/len(imgs):3.0f}% with detections")

    print("\n=== ranked by detection rate ===")
    for name, mean, rate in sorted(rows, key=lambda r: (-r[2], -r[1])):
        print(f"  {name:32s} {rate:3.0f}%   mean {mean:.2f}")
    print(f"\nPreviews: {args.out}")


if __name__ == "__main__":
    main()
