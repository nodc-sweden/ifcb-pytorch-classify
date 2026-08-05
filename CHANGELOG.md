# Changelog

This file records notable changes to the project. It follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Inference no longer applies the training augmentation. A transform name bundles
  preprocessing (padding, resize, normalisation) with augmentation (random flips
  and brightness/contrast jitter), and everything that scored images rebuilt the
  whole bundle from the checkpoint's `transform`. With an `_augmented` checkpoint
  that made every score a single random draw: because one transform is built per
  run and the RNG advances across ROIs, the same bin scored differently depending
  on which bins preceded it, on the torchvision version, and on anything else
  that shifted the stream. Re-running the *same* command stayed reproducible,
  because inference reseeds on every invocation; it was runs that differed in
  where the bin sat, or in the installed libraries, that diverged. Two such runs
  of one real bin, with identical weights, disagreed on 49 of 712 classifications
  and moved the sample's cell count by about 1%. Scoring now goes through
  `eval_transform_name`, which keeps the preprocessing and drops the random
  operations, and three call sites are fixed:

  - `infer` scores with the de-augmented transform and logs a warning when the
    checkpoint's name had augmentation in it.
  - The validation split in `create_training_datasets` is no longer augmented.
    Reported validation metrics, the checkpoint-selection metric and the
    per-class thresholds were all measured on randomly jittered and flipped
    images.
  - `normalise` computes dataset mean and standard deviation without
    augmentation, so brightness and contrast jitter no longer inflate them.

### Added

- The `h5` class-scores output now records what produced it: root attributes
  carrying the `ifcb-classify`, Python, torch and torchvision versions, the
  transform actually applied, the model architecture, and the checkpoint's
  SHA256.

  Until now the only provenance was `classifier_name`, derived from the
  checkpoint's *parent directory name* — so the same weights in two differently
  named folders produced differently labelled outputs, and nothing recorded the
  code or libraries involved. Given two files that disagreed, neither said why.

  HDF5 readers ignore unknown attributes, so this is additive there. No timestamp
  is recorded, so two identical runs still produce comparable files. The `mat`,
  `csv` and `csv-labels` outputs are unchanged: the `mat` is an interchange
  format for iRfcb and the Dashboard whose native reader aborts the whole file on
  any variable it cannot represent (a struct among them), so provenance stays in
  the `h5`; the csv column layouts are a contract with iRfcb and ClassiPyR.

  Note that `ifcb_classify_version` is the *installed* version. An editable
  checkout keeps whatever its last `pip install -e .` saw, so run that after
  pulling or the recorded version lags behind the code. The checkpoint hash and
  library versions are unaffected.

- `scripts/recompute_thresholds.py`, which refits an existing checkpoint's
  per-class thresholds against a de-augmented validation split. It reconstructs
  the split from the checkpoint's own config and refuses to run if the class list
  no longer matches, since a changed dataset would refit on images the model
  trained on. No retraining is involved — only the thresholds change.

  **Existing checkpoints need their thresholds recomputed.** The thresholds
  shipped alongside a model trained before this release were fitted against the
  augmented validation split, so they no longer match the operating point of the
  de-augmented model. Bins classified with an `_augmented` checkpoint each hold
  one arbitrary draw and should be reclassified for comparable results.

  **To tell whether a thresholds file is affected, look for a
  `validation_transform` key in it.** This release adds that key, and only
  releases from this one onward write it, so a file without one was fitted
  against the augmented split and should be refit. The file records no version of
  its own, so the key is the check — and it also covers hand-written thresholds
  files, which a version comparison could not. Inference warns once per run when
  it loads a file that lacks it.

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
- The README and the docs now record that the optional `chains` extra installs
  Ultralytics YOLO under AGPL-3.0, while the rest of the project is MIT. The
  dependency is not new, but the licence difference was documented nowhere.

### Fixed

- `pretrained: false` now actually trains from scratch. The setting was accepted
  but never read, so every run loaded ImageNet weights regardless. Anyone who
  believed they had trained a from-scratch baseline on an earlier version did
  not, and should retrain to get one.

  Inference reads the setting too, and builds the architecture the way training
  built it. That matters because torchvision's `weights=` argument also reshapes
  some models: `inception_v3` forces `transform_input=True`. A from-scratch
  `inception_v3` checkpoint was therefore run under input rescaling it had never
  been trained with, and `load_state_dict` still reported that every key matched.
  Inference on a from-scratch checkpoint also stops downloading ImageNet weights
  it then throws away, so it no longer needs network access.
- Per-class thresholds are now found automatically. Training writes them as
  `{run_name}_thresholds_and_metrics.json`, but inference only ever looked for a
  file named exactly `thresholds.json`. Unless you passed `--thresholds`, it fell
  back to `threshold_default` (0.0) and accepted every ROI, without saying so.

  This changes output. Every classification produced by an earlier version
  without an explicit `--thresholds` was un-thresholded: `class_name` always
  equalled `class_name_auto`, and no ROI was ever `unclassified`. Re-running
  those bins with `--overwrite` will now mark low-confidence ROIs
  `unclassified`. Inference logs which thresholds file it picked, and logs the
  fallback when it finds none. Where an output directory holds several runs, it
  uses the one matching the checkpoint's name, and falls back to the default
  rather than guess between them.
- Bins whose fileset has no `.hdr` are no longer skipped silently in directory
  mode. Directory discovery only finds filesets with a header, but the
  pending-work check globbed for `.roi` files. A headerless bin was therefore
  counted as work to do and never processed, and it came up pending again on
  every later run. The run printed no warning and exited zero. Both checks now
  use the same enumeration, and inference names any otherwise-complete fileset it
  had to skip for a missing header. To classify one anyway, pass its path
  directly.

  Only the missing-header case is reported. Directory discovery also ignores
  `skip` and `beads` paths, and any layout other than flat or
  `Dyyyy/Dyyyymmdd/`, exactly as pyifcb's did. Those exclusions are deliberate,
  so they pass without a warning. Reporting them would mean a warning for every
  bin of an archive organised some other way. If a directory run finds fewer bins
  than you expect, check the layout.
- `overwrite` and `allow_unsafe` can now be set in an inference YAML config, and
  `overwrite` in a `chains-count` config. Their command-line flags defaulted to
  `false` rather than "unset", so they overwrote whatever the file said and the
  config keys could never take effect.
- `googlenet` and `inception_v3_untrained` can now be trained. Both keep their
  auxiliary classifier heads when built from scratch, and the training loop only
  unwrapped the auxiliary output for the model named exactly `inception_v3`, so
  both died on the first batch with a type error. `inception_v3_untrained` is the
  from-scratch route the model registry advertises, and it had never worked.
- A checkpoint that is missing, unreadable or truncated now says so. Every
  failure to load used to come back as "Safe load failed, re-run with
  `--allow-unsafe`", including a typo'd path and a half-finished download, and
  taking that advice then failed anyway. A missing path or a directory raises
  `FileNotFoundError`, an unreadable archive is reported as truncated or corrupt,
  and only a genuine unpickling failure suggests `--allow-unsafe`. If the unsafe
  load then fails too, the file is reported as corrupt rather than raising a bare
  pickle error.
- Writing a `.mat` for a bin with more than 65535 ROIs now raises instead of
  wrapping. The v1 format stores `roinum` as uint16 and the value was cast
  without a range check, so ROI 70000 was written as 4464 and every score and
  count in the file was attributed to the wrong ROI, silently. Such a bin can
  still be written to `h5` or `csv-labels`, which use int32.

### Upgrading

Existing HDF5 output is unchanged in layout. Resolved labels will differ, though:
inference now applies the per-class thresholds training produced, so unless you
were already passing `--thresholds`, earlier runs and new ones will disagree. ROI
reads match pyifcb byte for byte across the test fixture, which is a D-style bin.
ifcbkit stitches I-style bins differently, and that comparison does not cover
them. Directory enumeration behaves as before on flat and nested
`Dyyyy/Dyyyymmdd/` layouts, and filesets missing a `.hdr` are skipped with a
warning, as pyifcb's discovery skipped them. Recreate or re-sync your environment
when you upgrade, since pyifcb is no longer a dependency.

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

[Unreleased]: https://github.com/nodc-sweden/ifcb-pytorch-classify/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/nodc-sweden/ifcb-pytorch-classify/compare/v.0.2.0...v0.3.0
[0.2.0]: https://github.com/nodc-sweden/ifcb-pytorch-classify/compare/v0.1.0...v.0.2.0
[0.1.0]: https://github.com/nodc-sweden/ifcb-pytorch-classify/releases/tag/v0.1.0
