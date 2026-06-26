# Chain-counting: annotation & training workflow

End-to-end workflow for building per-taxon YOLO cell-counting detectors for IFCB
chain-forming plankton, from labelling in Label Studio to a final GPU-trained
model. The detector counts cells in an ROI (count = number of detected boxes);
the `ifcb-classify` classifier decides *what* the ROI is, the detector decides
*how many* cells it contains.

The `chains-train` / `chains-eval` / `infer` commands are part of the package;
the helper scripts referenced below live in [`scripts/`](../scripts) and cover
the data side. They need the `chains` extra (`uv pip install -e ".[chains]"`).

## Helper scripts

| Script | Purpose |
|---|---|
| [`scripts/prepare_ls_yolo.py`](../scripts/prepare_ls_yolo.py) | Turn a Label Studio **YOLO export** into a `chains-train` dataset (pairs labels↔images, splits train/val, writes `data.yaml`). |
| [`scripts/ls_preannotate_api.py`](../scripts/ls_preannotate_api.py) | **Pre-annotate existing Label Studio tasks in place** via the API — runs a trained model and attaches predicted boxes as predictions (no duplicate tasks). |
| [`scripts/yolo_pre_annotate.py`](../scripts/yolo_pre_annotate.py) | Alternative: emit Label Studio **import JSON** with predictions, for images that are *not* yet tasks. |
| [`scripts/compare_bootstrap_models.py`](../scripts/compare_bootstrap_models.py) | Trial every trained detector on a new taxon and rank them, to pick a bootstrap model by **cell morphology** (not taxonomy). |

## The iterative bootstrap loop

Manual bounding-box annotation is the slow part. Don't draw every box from
scratch — bootstrap a model and *correct* its boxes, improving each round:

1. **Label a small batch** in Label Studio (~30–80 ROIs).
2. **Export → YOLO**, then build a dataset:
   ```bash
   python scripts/prepare_ls_yolo.py --labels <export>/labels --images <source_images> \
       --out datasets/<taxon> --class-name <taxon>
   ```
3. **Train** a bootstrap model:
   ```bash
   ifcb-classify chains-train --class-name <taxon> --data datasets/<taxon> \
       --model yolo11s.pt --imgsz 1024 --device cpu     # or --device 0 on GPU
   ```
4. **Pre-annotate** the next batch in place (predictions appear pre-drawn):
   ```bash
   python scripts/ls_preannotate_api.py --url http://localhost:8080 --project <id> \
       --token-file ~/.ls_token --weights <best.pt> --images <source_images> \
       --imgsz 1024 --conf 0.12 --limit 250
   ```
   Use a **low `--conf`** (0.10–0.15) so faint cells in long chains surface as
   candidate boxes — deleting extras is faster than drawing missed cells.
   Match `--imgsz` to the training resolution.
5. **Correct** the pre-annotated tasks, re-export, and retrain. Each round the
   pre-labels improve and you do less drawing.

Skeletonema reached **98% exact count match** (vs manual) this way; a 78-image
Thalassionema model reached recall **0.99**.

When approaching a **new** taxon with no detector yet, run
`scripts/compare_bootstrap_models.py --src <new_taxon_images> --models-root
models/chains --out /tmp/trial` first: it ranks your existing detectors on a
sample so you can pick the one whose cell shape transfers best as a starting
point (e.g. a rounded-centric detector bootstraps another rounded-centric taxon).

## Label Studio setup

- **Labelling config** (Settings → Labeling Interface), single class:
  ```xml
  <View>
    <Image name="image" value="$image" zoom="true" zoomControl="true"/>
    <RectangleLabels name="label" toName="image">
      <Label value="cell" background="#1f9d55"/>
    </RectangleLabels>
  </View>
  ```
  Select the `cell` label (or press `1`) before drawing, or the canvas won't draw.
- **One project per taxon** (each export → one detector).
- **Auth (LS 1.23):** personal access tokens are JWTs. Save the token to
  `~/.ls_token` (chmod 600); `ls_preannotate_api.py` exchanges it via
  `/api/token/refresh` automatically. (Legacy 40-char tokens also work.)
- **Source images:** with synced storage the export's `images/` is often empty —
  that's fine, the scripts pair the exported `labels/` to your local images by
  filename.

### Annotation conventions (decide once, apply uniformly)

Consistency matters more than pixel-perfection — the model learns the
distribution of your boxes.

- **One box per cell**, including cells that touch/overlap in a chain. Overlap is
  expected (NMS IoU is tuned for it via `chains-eval`).
- **Box partially-occluded cells** too (estimate their extent) — skipping them
  undercounts.
- Box the **cell body only**, excluding connecting threads / setae.
- Pick a rule for **cells cut off at the ROI border** (e.g. box if ≳50% visible)
  and for **faint/blurry** cells, and stick to it.
- Never box debris, detritus, or other taxa.

## Validating counts (`chains-eval`)

`chains-eval` compares a detector's predicted counts against a CSV of manual
counts (`file_name,cell_count` columns by default) and sweeps the NMS IoU
threshold so you can pick the best value per taxon:

```bash
ifcb-classify chains-eval --weights <best.pt> \
    --images datasets/<taxon>/images/val \
    --counts-csv counts.csv --ious 0.3,0.5,0.7
```

You don't need to re-count by hand: the ground-truth count for an annotated
image **is** the number of boxes in its label file, so the CSV can be generated
straight from the YOLO labels — e.g. each row is `<image>, $(wc -l < labels/<image>.txt)`.

## Resolution (`--imgsz`): match it to the ROI sizes

`chains-train` defaults to **640**. Whether to go higher depends on how large
the ROIs are — long chains are physically wider, so they live in the large-ROI
tail that 640 downsamples (smearing thin cells together). Check the distribution:

```python
from PIL import Image; import glob, numpy as np
longest = np.array([max(Image.open(f).size) for f in glob.glob("<images>/*.png")])
for thr in (640, 1024):
    print(thr, f"{100*(longest>thr).mean():.1f}% of ROIs exceed this")
```

**Worked example — Thalassionema nitzschioides** (3,818 ROIs, long thin stellate
chains): longest-side median 384 px, 95th pct 952 px, max 1359 px. **16% of ROIs
exceed 640 px** (these are the long chains — the hard cases), but only **1.2%
exceed 1024 px**. So:

- **640** is fine for short chains (small ROIs) and trains faster.
- **1024** rescues the long/blurry chains by not downsampling that 16% tail — the
  right default for this taxon. (`v2` at 1024 reached recall 0.99.)
- **1280+** isn't worth it here — almost no ROIs exceed 1024.

Rule of thumb: set `--imgsz` to roughly the ~95th-percentile longest side, capped
where the tail flattens. Use `chains-eval` on a manual-count set to confirm.

## One detector per genus

A detector is single-class ("cell vs not-cell"), so one genus-level model
typically serves all species of that genus plus the genus-level class — map
several classifier labels to the same weights in the inference config. Verify per
species with `chains-eval`; only train a species-specific detector if one species
shows high count error.

## Final training on GPU

CPU is fine for bootstrapping but slow. For production models, train on a CUDA
box with a larger model and high resolution:

```bash
ifcb-classify chains-train --class-name <taxon> --data datasets/<taxon> \
    --model yolo11x.pt --imgsz 1024 --device 0 --batch -1 --epochs 200
```

- `--batch -1` lets ultralytics auto-pick the largest safe batch (yolo11x @ 1024
  is VRAM-heavy; set a smaller `--batch` if you hit OOM).
- `--imgsz 1024` must be passed explicitly (it is **not** the default).

### Dataset portability

Datasets are standard YOLO folders (`images/{train,val}` + `labels/{train,val}` +
`data.yaml`). The `data.yaml`/`data.local.yaml` `path:` is **absolute** to this
machine, so to train on another box either (a) place the dataset at the same
path, or (b) edit `path:` to the new location. ROIs are tiny PNGs, so the folders
are small to copy. Pretrained weights (`yolo11x.pt`) auto-download on the GPU box.
