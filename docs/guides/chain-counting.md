# Chain counting

Many plankton grow as multi-celled colonies in a single ROI — chains (e.g.
*Skeletonema*), but also ribbons, fans, and branched or spherical colonies. For
these taxa, classification alone tells you *what* the ROI is but not *how many*
cells it contains. This optional feature trains a small
[YOLO](https://docs.ultralytics.com/) object detector **per taxon** that counts
the individual cells in each ROI, and (during inference) stores that **cell
count** alongside the classification result.

!!! note "It counts cells, not chains"
    The feature and its CLI commands keep the "chain" name (`chains-train`,
    `chains-count`, `chains-eval`) for historical reasons, but the value it
    produces is a per-ROI **cell count** — the number of cells in a colony of
    *any* form, not a tally of chains. The stored dataset is named `cell_count`
    accordingly.

This approach follows Groves et al. (2026), who demonstrated automatic
enumeration of marine diatom chains with YOLO:

> Groves, G. J. J., Arthur, G., Bresnan, E., Whyte, C., Arce, P., & Davidson, K.
> (2026). Automatic enumeration of chains of marine diatoms using "You Only Look
> Once"—a machine learning approach. *Journal of Plankton Research*, 48(2),
> fbaf064. https://doi.org/10.1093/plankt/fbaf064

Requires the `chains` extra:

```bash
uv pip install -e ".[chains]"
```

!!! tip "Annotating efficiently"
    Drawing every bounding box by hand is slow. See
    [Chain-counting annotation](chain-counting-annotation.md) for the full
    workflow — Label Studio setup, annotation conventions, the bootstrap loop
    (pre-annotate with a model, then *correct* its boxes), `--imgsz` guidance,
    and the helper scripts that support it.

## Training a detector for any chain-forming taxon

Train one detector per taxon you want to count. This works for any chain-forming
species — bring your own annotated data. Some annotated chain-count images are
available from
[EuropeanIFCBGroup/IFCBChainCounts](https://github.com/EuropeanIFCBGroup/IFCBChainCounts).

```bash
python -m ifcb_classify chains-train --config configs/chains_train_default.yaml
```

With CLI overrides (e.g. a larger model on a GPU):

```bash
python -m ifcb_classify chains-train \
    --class-name Skeletonema --data /path/to/datasets/skeletonema \
    --model yolo11x.pt --epochs 200 --device 0
```

The best checkpoint is written to `<project>/<name>/weights/best.pt`.

### Dataset layout

Object detection needs *bounding boxes* around individual cells (the class-folder
data used for classification has none), so ROIs must be annotated first (e.g.
with [Label Studio](https://labelstud.io/), [CVAT](https://www.cvat.ai/), or
[Roboflow](https://roboflow.com/)). Export in YOLO format:

```
datasets/skeletonema/
  data.yaml                 # names + train/val image dirs
  images/train/*.png        labels/train/*.txt   # one .txt of boxes per image
  images/val/*.png          labels/val/*.txt
```

Each label `.txt` holds one line per cell: `class_id cx cy w h` (normalised
0–1). With a single taxon per detector, `class_id` is always `0`. A `data.yaml`:

```yaml
path: /abs/path/to/datasets/skeletonema
train: images/train
val: images/val
names:
  0: skeletonema
```

`--data` accepts either a `data.yaml` file or a directory containing one
(`data.local.yaml` is preferred over `data.yaml` when both exist). A Label Studio
YOLO export doesn't match this layout directly —
[`scripts/prepare_ls_yolo.py`](https://github.com/nodc-sweden/ifcb-pytorch-classify/blob/main/scripts/prepare_ls_yolo.py)
pairs the exported labels to images, splits train/val, and writes the `data.yaml`
for you.

### Compute

`yolo11n.pt` (nano) trains in ~hours on CPU and is a good starting point; use a
larger model (`yolo11x.pt`) on a GPU (`--device 0`) for best accuracy. CUDA
requires a CUDA build of PyTorch (see [Installation](../installation.md)).

See `configs/chains_train_default.yaml` for all options.

## Counting during inference

Add a `chain_counting` block to your inference config to count cells while
classifying. Only ROIs whose **thresholded `class_name`** matches a configured
key are counted; all other ROIs get `cell_count = -1`.

```yaml
chain_counting:
  enabled: true
  conf: 0.25            # default; per-model override allowed
  iou: 0.30             # default; per-model override allowed
  models:
    Skeletonema_marinoi:
      weights: /models/chains/chains_skeletonema_yolo11n/weights/best.pt
      iou: 0.30
    # Several labels may share one detector (e.g. species + genus-level class):
    # Thalassiosira_spp: { weights: /models/chains/thalassiosira_best.pt }
```

!!! note "Keys must match the classifier's output labels exactly"
    A detector is a single-class "cell vs. not-cell" model, so one detector
    typically serves all species of a genus plus the genus-level class — map each
    label to the same weights.

!!! warning "Security"
    Detector `weights` are loaded with ultralytics' `YOLO(...)`, which unpickles
    the checkpoint and can execute arbitrary code. Only point `chain_counting` at
    weights you trained or otherwise trust — the same caution that applies to the
    classifier checkpoint loaded with `--allow-unsafe`.

```bash
python -m ifcb_classify infer --config configs/infer_with_chains.yaml
python -m ifcb_classify infer --config configs/infer_with_chains.yaml --no-count  # disable
```

The output `_class.h5` gains a `cell_count` dataset (int32, one per ROI; `-1`
where not counted) and a `cell_counter_models` JSON attribute recording the
weights/IoU/conf used. Existing consumers ignore the extra dataset. See
`configs/infer_with_chains.yaml` for a full example.

!!! info "What `cell_count` stores"
    The **number of cells in that ROI** — i.e. the number of boxes the detector
    found in the image. Each ROI is one colony (a chain, ribbon, fan, branched or
    spherical colony), and `cell_count` is how many cells it contains; it is
    *not* a tally of colonies. `-1` means the ROI was not counted (not a counted
    taxon, or below its classifier threshold).

## Counting on already-classified bins

If you already have `_class.h5` files and only want to add (or refresh) counts —
e.g. after training a new detector — use `chains-count` instead of re-running
`infer`. It reuses the stored `class_name` to decide which ROIs to count, so it
**skips the classifier entirely** and only runs the detector on the matching
ROIs, reading their pixels from the raw bins:

```bash
# Reuse the same inference config (input_path = raw bins, output_dir = the
# directory of existing *_class.h5 files, plus the chain_counting block):
python -m ifcb_classify chains-count --config configs/infer_with_chains.yaml

# Or point at the two directories directly:
python -m ifcb_classify chains-count \
    --input /path/to/raw/bins \
    --output output/class_scores \
    --config configs/infer_with_chains.yaml   # still needed for the detector block
```

Each file's `cell_count` dataset is written in place. Files that already carry
counts are skipped unless you pass `--overwrite`. The raw bins are still required
(the `.h5` stores scores, not pixels), but the expensive ResNet pass is avoided.

## Validating count accuracy

`chains-eval` compares a detector's predicted counts against manual counts and
sweeps the NMS IoU so you can pick the best value per taxon. Provide a directory
of test images and a CSV with a filename column and an integer count column
(`file_name,cell_count`):

```bash
python -m ifcb_classify chains-eval \
    --weights output/chains/chains_skeletonema_yolo11n/weights/best.pt \
    --images /path/to/test_images \
    --counts-csv /path/to/test_image_counts.csv \
    --ious 0.3,0.5,0.7
```

It reports MAE, mean bias, exact-match and within-±1 accuracy, and total counts
per IoU. Add `--output results.csv` for per-image predictions.

**Checking one detector across species** — to verify that a single genus-level
detector generalises (rather than training per species), run the *same*
`--weights` against each species' test set and compare the metrics. Train a
dedicated detector only if a particular species shows high error. See
`configs/chains_eval_default.yaml`.
