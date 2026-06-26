# Training

Fine-tune a pretrained image classifier on a class-folder dataset.

## Basic run

```bash
python -m ifcb_classify train --config configs/train_default.yaml
```

With CLI overrides:

```bash
python -m ifcb_classify train --config configs/train_default.yaml \
    --model convnext_tiny --lr 0.001 --epochs 30
```

See [Configuration](../configuration.md) for the full list of training
parameters and their defaults.

## Training data layout

Organise images in one folder per class:

```
training_data/V1/
  Asterionellopsis_glacialis/
  Dactyliosolen_fragilissimus/
  Dinophysis_acuminata/
  ...
```

## Evaluation plots

Add `--plots` to generate evaluation plots after training:

```bash
python -m ifcb_classify train --config configs/train_default.yaml --plots
```

This produces static PNG plots (training curves, per-class F1, precision vs.
recall scatter, class support distribution, top confused pairs) saved to
`<output_dir>/plots/<run_name>/`. Interactive HTML plots are also generated
(zoomable confusion matrix with row-normalized percentages, sortable per-class
metrics table).

## Dataset normalisation

Normalised transforms need the dataset mean and standard deviation. Compute them
with:

```bash
python -m ifcb_classify normalise --data-dir training_data/V1
```
