# Changelog

This file records notable changes to the project. It follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-03

### Added

- Inference can write class scores in more than one format, selected with
  `--format` or `output_format`. The options are `h5` (the default), `csv`,
  `mat`, `csv-labels`, any comma-separated combination, or `all`.
  - `csv` is the IFCB Dashboard's scores-only export: pid plus per-class scores.
  - `mat` is class_scores v1 MATLAB, which the dashboard can read and
    [iRfcb](https://europeanifcbgroup.github.io/iRfcb/) can process.
  - `csv-labels` is the ClassiPyR and iRfcb per-ROI label CSV, with `file_name`,
    `class_name`, `class_name_auto`, `score`, and `cell_count` where counts
    exist.

  Inference writes each format only when that file is missing, or when you pass
  `--overwrite`, so adding a format to already-processed bins leaves the
  existing files alone. The `h5`, `mat` and `csv-labels` outputs carry cell
  counts. The scores CSV does not.
- `ifcb-classify list-models`, which prints the architectures that
  `train --model` accepts.
- A MkDocs documentation site on GitHub Pages, with guides for training,
  inference and chain counting, a configuration reference, and an API reference
  generated from the docstrings.
- `CITATION.cff`, so GitHub's "Cite this repository" button produces APA and
  BibTeX metadata.

### Changed

- Raw IFCB bins now go through [ifcbkit](https://github.com/WHOIGit/ifcbkit)
  instead of [pyifcb](https://github.com/joefutrelle/pyifcb). pyifcb pinned its
  dependencies with `==` (scipy, scikit-image, pandas, Pillow, h5py), which
  froze those packages for the whole project and capped which Python versions
  would install. ifcbkit needs only Pillow and aiofiles.
- Python 3.11 to 3.14 is now supported, up from 3.11 and 3.12. The CI matrix
  covers all four.
- `iter_directory_bins` now yields a frozen `BinFiles` handle, the LID plus the
  resolved `.adc` and `.roi` paths, rather than an open pyifcb bin object, since
  ifcbkit has no equivalent. Code that imported it and used the old
  context-manager handle needs updating. The CLI is unaffected.
- A bin given as a direct file path now reads its `.adc` and `.roi` siblings
  itself instead of going through ifcbkit's directory discovery. That discovery
  only finds filesets that have a `.hdr`, which would have broken single-file
  inference on a fileset missing its header.
- `pyproject.toml` now names the Ruff rule set explicitly and CI pins the Ruff
  version, so a change to Ruff's defaults upstream no longer silently changes
  which rules run.

### Fixed

- `pretrained: false` now actually trains from scratch. The setting was accepted
  but never read, so every run loaded ImageNet weights regardless. Anyone who
  believed they had trained a from-scratch baseline on an earlier version did
  not, and should retrain to get one.

### Upgrading

Existing HDF5 output is unchanged. ROI reads match pyifcb byte for byte across
the test fixture, and directory enumeration behaves as before on flat, nested
`Dyyyy/Dyyyymmdd/` and incomplete-fileset layouts. Recreate or re-sync your
environment when you upgrade, since pyifcb is no longer a dependency.

## [0.2.0] - 2026-06-26

### Added

- Optional per-taxon [YOLO](https://docs.ultralytics.com/) detectors that count
  individual cells in colony ROIs. `chains-train` trains a detector,
  `chains-count` counts on pre-classified bins, and `chains-eval` checks count
  accuracy. Inference can also count as it classifies and store the count next
  to each classification.
- Label Studio annotation scripts and a guide to the annotation workflow.
- A production chain-counting inference config, which reuses detector weights
  across diatoms of similar shape.

### Changed

- The chain-counting HDF5 dataset is now called `cell_count` rather than
  `chain_count`, because it reports per-ROI cell counts for colonies of any
  form, not only chains.
- `chains-train` now defaults to `yolo11s` at image size 640 and batch size 16.

## [0.1.0] - 2026-03-19

### Added

- Training: fine-tune 40+ pretrained architectures (ResNet, EfficientNet,
  ConvNeXt, Vision Transformers and others) on class-folder image datasets, with
  evaluation plots and hyperparameter sweeps.
- Inference: batch-classify raw IFCB bins (`.roi`, `.adc`, `.hdr`) through
  pyifcb into IFCB Dashboard class_scores v3 HDF5, with per-class decision
  thresholds, in the format iRfcb reads. Legacy checkpoints also load.
- Experiment tracking through CSV (the default), MLflow or Weights & Biases.
- Automatic device selection: GPU for training, CPU by default for inference.

[0.3.0]: https://github.com/nodc-sweden/ifcb-pytorch-classify/compare/v.0.2.0...v0.3.0
[0.2.0]: https://github.com/nodc-sweden/ifcb-pytorch-classify/compare/v0.1.0...v.0.2.0
[0.1.0]: https://github.com/nodc-sweden/ifcb-pytorch-classify/releases/tag/v0.1.0
