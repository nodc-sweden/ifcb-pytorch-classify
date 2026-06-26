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

## Labelling images

Building this dataset means sorting IFCB ROIs into per-class folders. This
pipeline does not include an annotation tool, but several dedicated IFCB image
annotators exist — pick whichever fits your stack:

| Tool | Language | Link |
|---|---|---|
| ClassiPyR | R | <https://github.com/EuropeanIFCBGroup/ClassiPyR> |
| SAMS IFCB Annotator | Python | <https://github.com/EuropeanIFCBGroup/SAMS_IFCBAnnotator> |
| ifcb-analysis | MATLAB | <https://github.com/hsosik/ifcb-analysis> |

Export the labelled ROIs as one folder per class (as above), then point
`train` at that directory.

!!! note
    This is image-level labelling for **classification** (what each ROI is),
    which is separate from the bounding-box annotation used for
    [chain counting](chain-counting.md) (how many cells are in an ROI).

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
