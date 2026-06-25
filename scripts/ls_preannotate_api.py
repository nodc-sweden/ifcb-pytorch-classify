"""Pre-annotate existing Label Studio tasks in place via the API.

For each un-annotated task in a project, find the image locally, run a trained
detector, and POST the predicted boxes as a *prediction* on that task. The
boxes then appear pre-drawn when you open the task — correct and submit.

No duplicate tasks are created (predictions attach to existing tasks).

Token (never pass it on the command line in a shared shell): put it in a file
and use --token-file, or export LABEL_STUDIO_API_TOKEN.

Usage:
    python ls_preannotate_api.py --url http://localhost:8080 --project 1 \
        --token-file ~/.ls_token \
        --weights best.pt --images /path/to/source_images --limit 50
"""
import argparse
import os
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def get_token(args) -> str:
    if args.token:
        return args.token
    if args.token_file:
        return Path(args.token_file).expanduser().read_text().strip()
    env = os.environ.get("LABEL_STUDIO_API_TOKEN")
    if env:
        return env.strip()
    raise SystemExit("Provide a token via --token-file, --token, or LABEL_STUDIO_API_TOKEN")


def authenticate(session, url, token) -> None:
    """Set the right auth header on the session.

    Label Studio 1.23 personal access tokens are JWTs that must be exchanged
    for a short-lived access token; legacy tokens are used directly.
    """
    if token.count(".") == 2:  # JWT personal access token
        r = session.post(f"{url}/api/token/refresh", json={"refresh": token})
        r.raise_for_status()
        session.headers["Authorization"] = f"Bearer {r.json()['access']}"
    else:
        session.headers["Authorization"] = f"Token {token}"


def request(session, method, url, reauth, _retries=5, _backoff=2.0, **kwargs):
    """Make a request, re-authenticating once on 401 and retrying on 5xx.

    Label Studio's SQLite backend returns transient 500s when the DB is briefly
    locked (e.g. a concurrent writer); retry with exponential backoff instead of
    aborting a long import.
    """
    import time
    for attempt in range(_retries + 1):
        resp = session.request(method, url, **kwargs)
        if resp.status_code == 401:
            reauth()
            resp = session.request(method, url, **kwargs)
        if resp.status_code < 500 or attempt == _retries:
            return resp
        time.sleep(_backoff * (2 ** attempt))
    return resp


def detect_config(project: dict):
    """Return (from_name, to_name, image_key, label) from the project's parsed config."""
    cfg = project.get("parsed_label_config") or {}
    for name, ctrl in cfg.items():
        if ctrl.get("type") == "RectangleLabels":
            to_name = (ctrl.get("to_name") or ["image"])[0]
            inputs = ctrl.get("inputs") or [{}]
            image_key = inputs[0].get("value", "image")
            labels = ctrl.get("labels") or ["cell"]
            return name, to_name, image_key, labels[0]
    raise SystemExit("No RectangleLabels control found in the project's labeling config")


def image_filename(task_data: dict, image_key: str) -> str | None:
    value = task_data.get(image_key)
    if not value:
        # Fall back to any string value that looks like a path/url
        value = next((v for v in task_data.values() if isinstance(v, str)), None)
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.query:
        q = parse_qs(parsed.query)
        if "d" in q:  # /data/local-files/?d=<path>
            return Path(unquote(q["d"][0])).name
    return Path(unquote(parsed.path or value)).name


def iter_tasks(session, url, project, reauth, page_size=100):
    page = 1
    while True:
        r = request(session, "GET", f"{url}/api/tasks", reauth,
                    params={"project": project, "page": page, "page_size": page_size})
        if r.status_code == 404:
            break
        r.raise_for_status()
        data = r.json()
        tasks = data["tasks"] if isinstance(data, dict) and "tasks" in data else data
        if not tasks:
            break
        yield from tasks
        if len(tasks) < page_size:
            break
        page += 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--project", type=int, required=True)
    p.add_argument("--token")
    p.add_argument("--token-file", dest="token_file")
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True, help="Local folder holding the source images")
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (match training imgsz)")
    p.add_argument("--limit", type=int, default=0, help="Max tasks to pre-annotate (0 = all)")
    p.add_argument("--overwrite", action="store_true", help="Also (re)predict tasks that already have predictions")
    p.add_argument("--dry-run", action="store_true", help="Don't POST; just report what would happen")
    args = p.parse_args()

    token = get_token(args)
    session = requests.Session()

    def reauth():
        authenticate(session, args.url, token)

    reauth()

    project = request(session, "GET", f"{args.url}/api/projects/{args.project}", reauth).json()
    from_name, to_name, image_key, label = detect_config(project)
    print(f"Config: from_name={from_name} to_name={to_name} image_key={image_key} label={label}")

    images_dir = Path(args.images)
    local = {f.name: f for f in images_dir.iterdir() if f.suffix.lower() in IMG_EXTS}

    from ultralytics import YOLO
    model = YOLO(args.weights)

    n_done = n_skip_annotated = n_skip_haspred = n_missing = 0
    for task in iter_tasks(session, args.url, args.project, reauth):
        if args.limit and n_done >= args.limit:
            break
        if task.get("total_annotations") or task.get("is_labeled") or task.get("annotations"):
            n_skip_annotated += 1
            continue
        if not args.overwrite and (task.get("total_predictions") or task.get("predictions")):
            n_skip_haspred += 1
            continue

        fname = image_filename(task.get("data", {}), image_key)
        img = local.get(fname) if fname else None
        if img is None:
            n_missing += 1
            continue

        result = model(str(img), imgsz=args.imgsz, iou=args.iou, conf=args.conf, verbose=False)[0]
        h, w = result.orig_shape
        xyxyn = result.boxes.xyxyn.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        boxes = [{
            "type": "rectanglelabels",
            "from_name": from_name,
            "to_name": to_name,
            "original_width": int(w),
            "original_height": int(h),
            "image_rotation": 0,
            "value": {
                "x": float(x1) * 100, "y": float(y1) * 100,
                "width": float(x2 - x1) * 100, "height": float(y2 - y1) * 100,
                "rotation": 0, "rectanglelabels": [label],
            },
        } for (x1, y1, x2, y2) in xyxyn]

        if args.dry_run:
            print(f"  [dry-run] task {task['id']} {fname}: {len(boxes)} boxes")
        else:
            resp = request(session, "POST", f"{args.url}/api/predictions", reauth, json={
                "task": task["id"], "result": boxes,
                "model_version": "bootstrap",
                "score": float(confs.mean()) if len(confs) else 0.0,
            })
            resp.raise_for_status()
        n_done += 1

    print(f"\nPre-annotated: {n_done}")
    print(f"Skipped (already annotated): {n_skip_annotated}")
    print(f"Skipped (already had prediction): {n_skip_haspred}")
    print(f"Skipped (image not found locally): {n_missing}")


if __name__ == "__main__":
    main()
