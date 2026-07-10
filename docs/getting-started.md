# Getting started

This walks through the happy path: install, train a classifier on a folder of
labelled images, and classify raw IFCB bins. For CUDA and the optional extras,
see [Installation](installation.md). New to the terminology (fine-tuning,
checkpoint, ROI, bin)? See [Concepts & glossary](concepts.md) first.

!!! tip "Two ways to run every command"
    After installing, the tool is available both as `ifcb-classify <command>`
    (a console script) and as `python -m ifcb_classify <command>`. They are
    identical — this guide uses the `python -m` form, but you can swap in
    `ifcb-classify` anywhere.

## 1. Install (CPU)

Requires Python 3.11–3.12 and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv pip install -e .
```

## 2. Get some training data

Training expects **one folder per class**, each folder full of the ROI images
for that class:

```
example_data/plankton/
  Guinardia_delicatula/
  Cryptomonadales/
  Leptocylindrus_danicus_minimus/
  ...
```

This repo bundles a small [example dataset](https://github.com/nodc-sweden/ifcb-pytorch-classify/tree/main/example_data)
(`example_data/plankton/`, six classes, 40 images each) so you can complete this
walkthrough without building a dataset first. It's enough to *see the pipeline
run*, not to train an accurate model — for that you need many more images per
class and more classes. See [Training](guides/training.md#labelling-images) for
how to build your own labelled dataset.

## 3. Train a model

```bash
python -m ifcb_classify train --config configs/train_default.yaml \
    --data-dir example_data/plankton --dataset-version example
```

Training on CPU is slow (tens of minutes to hours, depending on your machine); a
GPU is much faster. See [Training](guides/training.md#hardware-and-training-time)
for expectations and how to speed up a quick demo.

When it finishes, the best checkpoint is written to your output directory
(`output/` by default). **The filename is built from the run settings**, not a
fixed name — with the command above it is:

```
output/example-resnet50_dataset_squarepad_augmented_b64_lr0.0001_e20_best.pt
```

Alongside it you'll find `..._classes.txt` (the class list) and
`..._thresholds_and_metrics.json`. See
[Training](guides/training.md#what-training-produces) for the full list and how
to read the results. Use whatever `*_best.pt` file appears in `output/` in the
next step.

## 4. Run inference on raw bins

Inference runs on **raw IFCB bins** (`.roi/.adc/.hdr` triples), not on ROI
images. This repo bundles one bin in `example_data/bins/`, so you can run this
step directly:

```bash
python -m ifcb_classify infer \
    --input example_data/bins \
    --model output/example-resnet50_dataset_squarepad_augmented_b64_lr0.0001_e20_best.pt \
    --output output/class_scores
```

(For your own data, point `--input` at a directory of bins from your IFCB
instrument. The toy model above won't classify a real bin *accurately* — it only
knows six classes — but the run produces a valid output file so you see the
pipeline work end to end.)

This writes one `{sample}_class.h5` per bin, compatible with the IFCB Dashboard,
[iRfcb](https://europeanifcbgroup.github.io/iRfcb/) and
[ClassiPyR](https://europeanifcbgroup.github.io/ClassiPyR/). See
[Inference](guides/inference.md) for config files, legacy checkpoints, and
output details.

## Next steps

- [Concepts & glossary](concepts.md) — what the terms mean if any were unfamiliar
- [Configuration](configuration.md) — all training/inference options
- [Chain counting](guides/chain-counting.md) — count cells in chain-forming taxa
- [API reference](reference/ifcb_classify/index.md) — the Python API
