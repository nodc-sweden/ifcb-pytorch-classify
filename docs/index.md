# IFCB Classify

Pipeline for training and running inference on IFCB (Imaging FlowCytobot)
plankton images using PyTorch.

## Capabilities

- **Training** — Fine-tune 40+ pretrained architectures (ResNet, EfficientNet,
  ConvNeXt, Vision Transformers, etc.) on class-folder organised image datasets,
  with optional evaluation plots (static PNG + interactive HTML)
- **Inference** — Batch-classify raw IFCB bins (`.roi/.adc/.hdr`) via
  [pyifcb](https://github.com/joefutrelle/pyifcb) into HDF5 files in IFCB
  Dashboard class_scores v3 format, with per-class decision thresholds; output
  works with the IFCB Dashboard,
  [iRfcb](https://europeanifcbgroup.github.io/iRfcb/) and
  [ClassiPyR](https://europeanifcbgroup.github.io/ClassiPyR/)
- **Chain / cell counting** — Optionally train per-taxon
  [YOLO](https://docs.ultralytics.com/) detectors that count individual cells in
  colony ROIs and store the count alongside each classification
- **Experiment tracking** — CSV (default), MLflow, or Weights & Biases
- **Built for pipelines** — Date-placeholder paths for date-organised continuous
  inference, and automatic device selection (GPU for training, CPU by default for
  inference)

## Where to next

| If you want to… | Go to |
|---|---|
| Get up and running quickly | [Getting started](getting-started.md) |
| Install with CUDA or optional extras | [Installation](installation.md) |
| Train a classifier | [Training](guides/training.md) |
| Classify raw bins | [Inference](guides/inference.md) |
| Count cells in chains/colonies | [Chain counting](guides/chain-counting.md) |
| Look up config options | [Configuration](configuration.md) |
| Browse the code API | [API reference](reference/ifcb_classify/index.md) |

The project lives on
[GitHub](https://github.com/nodc-sweden/ifcb-pytorch-classify).
