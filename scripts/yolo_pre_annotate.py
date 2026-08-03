"""Pre-annotate images with a trained detector for review in Label Studio.

Runs a chain detector over a folder of images and emits Label Studio task JSON
with the predicted boxes attached as PREDICTIONS. Import this into your project
so each task opens with boxes already drawn — you just correct and submit.

Usage:
    python yolo_pre_annotate.py --weights best.pt --images /pool/thalassionema \
        --out preann.json --image-root-url "/data/local-files/?d=thalassionema/"

`--image-root-url` is prefixed to each filename to build the task's image URL;
it must match how Label Studio serves your images (e.g. the local-files path).
`--from-name`/`--to-name`/`--label` must match your labeling config
(<RectangleLabels name="label" toName="image"> with <Label value="cell">).
"""
import argparse
import json
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True, help="Directory of images to pre-annotate")
    p.add_argument("--out", required=True, help="Output Label Studio JSON")
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--label", default="cell")
    p.add_argument("--from-name", dest="from_name", default="label")
    p.add_argument("--to-name", dest="to_name", default="image")
    p.add_argument("--image-root-url", dest="image_root_url", default="")
    p.add_argument("--model-version", dest="model_version", default="bootstrap")
    args = p.parse_args()

    from ultralytics import YOLO

    images_dir = Path(args.images)
    files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"No images found in {images_dir}")

    model = YOLO(args.weights)

    tasks = []
    total_boxes = 0
    for i in range(0, len(files), 64):
        batch = [str(f) for f in files[i : i + 64]]
        for path, result in zip(
            files[i : i + 64], model(batch, iou=args.iou, conf=args.conf, verbose=False), strict=True
        ):
            h, w = result.orig_shape
            xyxyn = result.boxes.xyxyn.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            results = []
            for (x1, y1, x2, y2), c in zip(xyxyn, confs, strict=True):
                results.append({
                    "type": "rectanglelabels",
                    "from_name": args.from_name,
                    "to_name": args.to_name,
                    "original_width": int(w),
                    "original_height": int(h),
                    "image_rotation": 0,
                    "value": {
                        "x": float(x1) * 100.0,
                        "y": float(y1) * 100.0,
                        "width": float(x2 - x1) * 100.0,
                        "height": float(y2 - y1) * 100.0,
                        "rotation": 0,
                        "rectanglelabels": [args.label],
                    },
                    "score": float(c),
                })
            total_boxes += len(results)
            tasks.append({
                "data": {"image": f"{args.image_root_url}{path.name}"},
                "predictions": [{
                    "model_version": args.model_version,
                    "score": float(confs.mean()) if len(confs) else 0.0,
                    "result": results,
                }],
            })

    Path(args.out).write_text(json.dumps(tasks, indent=2))
    print(f"Wrote {len(tasks)} tasks ({total_boxes} predicted boxes) to {args.out}")
    print("Import into Label Studio: Project > Import > select this JSON file.")


if __name__ == "__main__":
    main()
