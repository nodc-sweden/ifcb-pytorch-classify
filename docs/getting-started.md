# Getting started

This walks through the happy path: install, train a classifier on a folder of
labelled images, and classify a directory of raw IFCB bins. For CUDA and the
optional extras, see [Installation](installation.md).

## 1. Install (CPU)

Requires Python 3.11–3.14 and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv pip install -e .
```

## 2. Organise training data

Training expects one folder per class:

```
training_data/V1/
  Asterionellopsis_glacialis/
  Dactyliosolen_fragilissimus/
  Dinophysis_acuminata/
  ...
```

## 3. Train a model

```bash
python -m ifcb_classify train --config configs/train_default.yaml
```

The best checkpoint is written to your configured output directory. See
[Training](guides/training.md) for CLI overrides, evaluation plots, and
normalisation.

## 4. Run inference on raw bins

Point inference at a directory of raw IFCB bins (`.roi/.adc/.hdr`):

```bash
python -m ifcb_classify infer \
    --input /path/to/bins \
    --model output/model_best.pt \
    --output /path/to/class_scores
```

This writes one `{sample}_class.h5` per bin, compatible with the IFCB Dashboard,
[iRfcb](https://europeanifcbgroup.github.io/iRfcb/) and
[ClassiPyR](https://europeanifcbgroup.github.io/ClassiPyR/). See
[Inference](guides/inference.md) for config files, legacy checkpoints, and
output details.

## Next steps

- [Configuration](configuration.md) — all training/inference options
- [Chain counting](guides/chain-counting.md) — count cells in chain-forming taxa
- [API reference](reference/ifcb_classify/index.md) — the Python API
