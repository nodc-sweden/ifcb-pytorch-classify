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

New to terms like *epoch*, *learning rate*, or *fine-tuning*? See
[Concepts & glossary](../concepts.md).

## Configuration files

Training is driven by a YAML config file, passed with `--config` (required
for `train`).
[`configs/train_default.yaml`](https://github.com/nodc-sweden/ifcb-pytorch-classify/blob/main/configs/train_default.yaml)
is a commented template covering every option. Copy it and edit it
for your own runs (data directory, model, learning rate, and so on).

Most settings can also be overridden on the command line, as with `--model`,
`--lr` and `--epochs` above; CLI values take precedence over the file.
`pretrained`, `sweep_params` and `manual_include_classes` have no CLI flag and
must be set in the config file. A common workflow is to keep one config per
dataset or experiment and tweak the odd parameter on the CLI.

See [Configuration](../configuration.md) for the full parameter reference and
defaults.

## Supported models

`--model` (or `model:` in the config) accepts any of 52 torchvision
architectures across these families, all fine-tuned from ImageNet-pretrained
weights except `inception_v3_untrained`, which trains from scratch:

| Family | Examples |
|---|---|
| ResNet / Wide ResNet / ResNeXt | `resnet18`, `resnet50` (default), `resnet152`, `wide_resnet50_2`, `resnext50_32x4d` |
| ConvNeXt | `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large` |
| EfficientNetV2 | `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l` |
| Vision Transformers | `vit_b_16`, `vit_l_16`, `swin_v2_t`, `maxvit_t` |
| VGG / DenseNet / MobileNet / others | `vgg16`, `densenet121`, `mobilenet_v3_large`, `mnasnet1_0`, `alexnet`, … |

To see the exact list of accepted names, run:

```bash
ifcb-classify list-models
```

The full registry (with the classifier-head details for each) is in the
[`models.registry`](../reference/ifcb_classify/models/registry.md) API reference.
Bigger models are generally more accurate but slower and more memory-hungry; the
default `resnet50` is a reasonable starting point.

## Hardware and training time

Training is much faster on an NVIDIA GPU (`cuda`) than on CPU. The device
is selected automatically (GPU if available), and the default `resnet50` is a
fairly heavy architecture:

- **GPU**: a real dataset typically trains in minutes to a couple of hours.
- **CPU**: slow. Tens of minutes for a tiny demo, many hours (or
  impractical) for a large dataset. If you don't have a GPU, keep runs small.

To speed up a quick CPU demo (e.g. on the bundled example data), use a
lighter model and fewer epochs. Accuracy will suffer, but the pipeline will
run:

```bash
python -m ifcb_classify train --config configs/train_default.yaml \
    --data-dir example_data/plankton --dataset-version example \
    --model resnet18 --epochs 5
```

For CUDA setup, see [Installation](../installation.md#with-cuda).

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
annotators exist. Pick whichever fits your stack:

| Tool | Language | Link |
|---|---|---|
| ClassiPyR | R | <https://github.com/EuropeanIFCBGroup/ClassiPyR> |
| SAMS IFCB Annotator | Python | <https://github.com/EuropeanIFCBGroup/SAMS_IFCBAnnotator> |
| ifcb-analysis | MATLAB | <https://github.com/hsosik/ifcb-analysis> |

Export the labelled ROIs as one folder per class (as above), then point
`train` at that directory.

!!! note
    This is image-level labelling for classification (what each ROI is),
    which is separate from the bounding-box annotation used for
    [chain counting](chain-counting.md) (how many cells are in an ROI).

## What training produces

Everything lands in your `output_dir` (`output/` by default). Files are named
after the run rather than with a fixed name. The run name is built from the
settings (`{dataset_version}-{model}_{transform}_b{batch_size}_lr{lr}_e{epochs}`),
so a default run on data tagged `example` produces:

(Sweep runs are named differently, one per grid combination:
`{dataset_version}-{param}{value}-{param}{value}…`. See
[Hyperparameter sweeps](../configuration.md#hyperparameter-sweeps).)

| File | What it is |
|---|---|
| `example-resnet50_..._e20_best.pt` | The best checkpoint. Pass this to `infer --model`. |
| `example-resnet50_..._e20_classes.txt` | The class list, in the model's label order. |
| `example-resnet50_..._e20_thresholds_and_metrics.json` | Per-class decision thresholds and metrics. |
| `example-resnet50_..._e20.csv` | Per-epoch metrics (with the default `csv` tracker). |
| `confusion_matrix/<run_name>/` | Per-epoch confusion matrices (`csv` tracker). |
| `plots/<run_name>/` | Evaluation plots, if you passed `--plots` or set `plots: true`. |

Only one `*_best.pt` is kept; it's overwritten whenever a later epoch scores
higher on the `checkpoint_metric` (weighted F1 by default). There is **no**
`model_best.pt`; use whatever `*_best.pt` file appears in `output/`.

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

## Reading the results

A single overall accuracy number hides most of what matters. Look at:

- **Training curves**: training and validation metrics per epoch. If validation
  F1 stops improving (or falls) while training F1 keeps climbing, the model is
  overfitting; if both are still rising at the last epoch, you likely
  stopped too early. Adjust `epochs` accordingly.
- **Per-class F1**: near-zero for some classes usually means too few images
  in those classes. Collect more, or exclude tiny classes with
  `min_class_images` (see [Configuration](../configuration.md)).
- **Confusion matrix**: shows *which* classes get mistaken for which.
  Off-diagonal hotspots between two visually similar taxa point to genuinely hard
  pairs (more labelled examples help most here).
- **Class support distribution**: how many images each class has. Large
  imbalance (some classes hundreds, others a handful) drags down the rare
  classes; `weighted_f1` accounts for size, but more balanced data is better.

There's no universal "good" F1; it depends on the number of classes and how
separable they are. Practical levers, in rough order of impact: more labelled
data (especially for weak classes), a longer or shorter run (from the
curves), a different model or transform, then the learning rate.

!!! note "The example dataset won't score well"
    Six classes with 40 images each is enough to watch the pipeline run, not to
    train an accurate model. Expect modest, noisy metrics. The cause is the
    data size, not a bug.

## Dataset normalisation

Normalised transforms need the dataset mean and standard deviation. Compute them
with:

```bash
python -m ifcb_classify normalise --data-dir training_data/V1
```
