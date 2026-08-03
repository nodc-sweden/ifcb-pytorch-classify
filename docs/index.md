# IFCB Classify

Pipeline for training and running inference on IFCB (Imaging FlowCytobot)
plankton images using PyTorch.

## Capabilities

- **Training.** Fine-tune 52 pretrained architectures (ResNet, EfficientNet,
  ConvNeXt, Vision Transformers, etc.) on class-folder organised image datasets,
  with optional evaluation plots (static PNG + interactive HTML). Run
  `ifcb-classify list-models` to see them all
- **Inference.** Batch-classify raw IFCB bins (`.roi/.adc/.hdr`) via
  [ifcbkit](https://github.com/WHOIGit/ifcbkit) into IFCB Dashboard class_scores
  v3 HDF5 by default, or into `csv`, `mat` and `csv-labels` via `--format`, with
  per-class decision thresholds; output works with the IFCB Dashboard,
  [iRfcb](https://europeanifcbgroup.github.io/iRfcb/) and
  [ClassiPyR](https://europeanifcbgroup.github.io/ClassiPyR/)
- **Chain / cell counting.** Optionally train per-taxon
  [YOLO](https://docs.ultralytics.com/) detectors that count individual cells in
  colony ROIs and store the count alongside each classification
- **Experiment tracking.** CSV (default), MLflow, or Weights & Biases
- **Built for pipelines.** Date-placeholder paths for date-organised continuous
  inference, and automatic device selection (CUDA, then Apple MPS, then CPU) for
  both training and inference, overridable per run with `infer --device`

## Where to next

| If you want to… | Go to |
|---|---|
| Get up and running quickly | [Getting started](getting-started.md) |
| Install with CUDA or optional extras | [Installation](installation.md) |
| Understand the terminology | [Concepts & glossary](concepts.md) |
| Train a classifier | [Training](guides/training.md) |
| Classify raw bins | [Inference](guides/inference.md) |
| Count cells in chains/colonies | [Chain counting](guides/chain-counting.md) |
| Look up config options | [Configuration](configuration.md) |
| Fix a common error | [Troubleshooting](troubleshooting.md) |
| Browse the code API | [API reference](reference/ifcb_classify/index.md) |

The project lives on
[GitHub](https://github.com/nodc-sweden/ifcb-pytorch-classify).

## Citation

If you use this software, please cite it. On GitHub, use the **"Cite this
repository"** button (which reads
[`CITATION.cff`](https://github.com/nodc-sweden/ifcb-pytorch-classify/blob/main/CITATION.cff)),
or use the following BibTeX:

```bibtex
@software{torstensson_ifcb_classify,
  author  = {Torstensson, Anders},
  title   = {ifcb-classify: a PyTorch pipeline for IFCB plankton image classification},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/nodc-sweden/ifcb-pytorch-classify}
}
```
