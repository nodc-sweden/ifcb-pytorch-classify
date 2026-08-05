# Concepts & glossary

The docs use vocabulary from two fields: machine learning (PyTorch) and IFCB
plankton imaging. If a term in the guides was unfamiliar, look it up here. If
you're brand new to PyTorch, the official
[PyTorch tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html)
are a good companion.

## How the pipeline works (the short version)

An IFCB instrument photographs particles in a water sample, producing thousands
of small images called **ROIs**. This tool does two separate jobs:

1. **Classification**: decide *what* each ROI is (which plankton class). You
   first **train** a model on ROIs you've already sorted into per-class folders;
   the model then labels new ROIs automatically.
2. **Chain counting** (optional): for colony-forming taxa, count *how many*
   cells are in an ROI, using a separate object-detection model.

Training and inference are distinct phases: training learns from labelled
examples and produces a model file (a *checkpoint*); inference loads that
checkpoint and applies it to new, unlabelled data. You train once, then run
inference many times.

## Machine-learning terms

**Model / architecture**: the neural network. `resnet50`, `convnext_tiny`, etc.
are architectures with different size/accuracy trade-offs. See the
[`models.registry`](reference/ifcb_classify/models/registry.md) module for the
full list.

**Pretrained**: the model starts from weights learned on a large general image
dataset (ImageNet) rather than from scratch, which means you're fine-tuning.
`pretrained: true` (the default) does this. Set `pretrained: false` to train from
scratch instead. `inception_v3_untrained` always trains from scratch, whatever
the setting.

**Fine-tuning**: continuing to train a pretrained model on your own data.
Because the model already "knows" generic visual features, fine-tuning needs far
less data and time than training from scratch.

**Checkpoint**: a saved model file (`.pt`). This pipeline keeps the
best-scoring one as `{run_name}_best.pt` (see
[What training produces](guides/training.md#what-training-produces)). Inference
loads a checkpoint with `--model`.

**Epoch**: one full pass over the training data. `epochs: 20` means the model
sees every training image 20 times. Too few and the model underfits; too many
and it may **overfit** (memorise the training set instead of generalising).

**Learning rate (`lr`)**: how big a step the model takes when updating its
weights each batch. Too high and training is unstable; too low and it learns
slowly. `0.0001` is a sensible default for fine-tuning.

**Batch size**: how many images are processed together before the model
updates. Larger batches are faster but use more memory (RAM or GPU VRAM). Lower
it if you run out of memory.

**Transform**: the preprocessing applied to each image before it reaches the
model. It resizes/pads the image to a fixed square and, for augmented
transforms, adds random flips and brightness/contrast jitter that make the
model more robust. This project's transform names are listed in
[`data.datasets`](reference/ifcb_classify/data/datasets.md).

The random part belongs to training only. Anywhere images are *scored* rather
than trained on — inference, the validation split, dataset statistics — the name
is mapped to its augmentation-free counterpart first, so a result depends only
on the image. Applying the random operations while scoring would make every
prediction one draw of many, and the same image would classify differently from
one run to the next.

**Augmentation**: deliberately varying training images (random horizontal and
vertical flips, plus brightness/contrast jitter) so the model doesn't overfit to
exact pixel positions.

**Normalisation (mean/std)**: rescaling pixel values so they're centred around
zero, which helps training. Some transforms need your dataset's mean and standard
deviation; compute them with `ifcb-classify normalise` (see
[Training](guides/training.md#dataset-normalisation)).

**Validation split (`val_split`)**: the fraction of data held back from training
and used only to measure how well the model generalises. `0.2` means 20% is used
for validation.

**Metrics**: numbers summarising model quality.

- **Precision**: of the ROIs the model *called* class X, how many really were X.
- **Recall**: of the ROIs that really *are* X, how many the model found.
- **F1**: the harmonic mean of precision and recall (one number balancing both).
- **Weighted F1**: F1 averaged across classes, weighted by class size; the
  default `checkpoint_metric`.
- **Confusion matrix**: a grid showing which classes get mistaken for which.

**Threshold**: the minimum confidence score for the model to commit to a label.
Below it, an ROI is left unclassified. This pipeline can use a per-class
threshold (tuned during training) so rare or hard classes aren't over-predicted.

**Device**: where computation runs, either `cpu` or `cuda` (an NVIDIA GPU).
GPUs are much faster for training. See
[Installation](installation.md#with-cuda).

## IFCB / domain terms

**IFCB**: Imaging FlowCytobot, an instrument that images individual particles
(plankton, detritus) as water flows past a camera.

**ROI**: Region Of Interest. One cropped image of a single particle, the unit
this pipeline classifies and counts.

**Bin**: one IFCB sample, stored as a triple of raw files with the same name.

- **`.roi`**: the raw image pixels for every ROI in the sample.
- **`.adc`**: per-ROI metadata (position, size, etc.).
- **`.hdr`**: sample-level header/settings.

Inference reads bins directly (via [ifcbkit](https://github.com/WHOIGit/ifcbkit));
you don't extract the ROIs to PNGs yourself.

**Chain / colony**: many plankton grow as multi-celled units in a single ROI
(chains, ribbons, fans, branched or spherical colonies). Classification says
what the colony is; [chain counting](guides/chain-counting.md) counts the cells
in it.

**class_scores (v3)**: the HDF5 output format inference writes (one
`{sample}_class.h5` per bin), compatible with the IFCB Dashboard,
[iRfcb](https://europeanifcbgroup.github.io/iRfcb/) and
[ClassiPyR](https://europeanifcbgroup.github.io/ClassiPyR/).
