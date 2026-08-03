# Configuration

`train` is driven by a YAML config file; `infer` can use one, or run without a
config from `--input` and `--model` alone. Most values can also be overridden on
the command line (CLI wins over the file), the exceptions being `pretrained`,
`sweep_params`, `manual_include_classes` and `chain_counting`, which are YAML
only. Copy one of the ready-to-use templates,
[`configs/train_default.yaml`](https://github.com/nodc-sweden/ifcb-pytorch-classify/blob/main/configs/train_default.yaml)
or
[`configs/infer_default.yaml`](https://github.com/nodc-sweden/ifcb-pytorch-classify/blob/main/configs/infer_default.yaml),
and edit it. The tables below are the full parameter reference; the
defaults come from the config dataclasses in
[`config`](reference/ifcb_classify/config.md).

New to terms like *epoch*, *learning rate*, or *validation split*? See
[Concepts & glossary](concepts.md).

## Training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | `training_data/V1` | Dataset root: one folder per class (`--data-dir`). |
| `dataset_version` | `V1` | Tag used in the run name and output filenames (`--dataset-version`). |
| `val_split` | `0.2` | Fraction of data held out for validation (0 to 1, exclusive). |
| `image_width` | `224` | Input width the images are resized/padded to. |
| `image_height` | `224` | Input height the images are resized/padded to. |
| `mean` | *(unset)* | Dataset pixel mean for normalised transforms; compute with `normalise`. |
| `std` | *(unset)* | Dataset pixel std for normalised transforms; compute with `normalise`. |
| `transform` | `dataset_squarepad_augmented` | Preprocessing pipeline (see [`data.datasets`](reference/ifcb_classify/data/datasets.md)). |
| `model` | `resnet50` | Architecture (see [Training → Supported models](guides/training.md#supported-models)). |
| `pretrained` | `true` | Start from ImageNet-pretrained weights (fine-tune) rather than from scratch. `inception_v3_untrained` trains from scratch either way. YAML only. |
| `lr` | `0.0001` | Learning rate (must be > 0). |
| `batch_size` | `64` | Images per update; lower it if you run out of memory. |
| `epochs` | `20` | Number of full passes over the training data. |
| `num_workers` | `0` | Data-loading worker processes (0 = load in the main process). |
| `seed` | `42` | Random seed for reproducible splits/shuffling. |
| `output_dir` | `output` | Where checkpoints, metrics, and plots are written. |
| `checkpoint_metric` | `weighted_f1` | Metric that decides which epoch is kept as `*_best.pt`. |
| `tracker` | `csv` | Experiment tracker: `csv`, `mlflow`, `wandb`, or `none`. |
| `mlflow_uri` | *(unset)* | MLflow tracking server URI (with `tracker: mlflow`). |
| `wandb_project` | *(unset)* | Weights & Biases project name (with `tracker: wandb`). |
| `experiment_name` | `ifcb-classify` | Experiment name passed to the tracker. |
| `min_class_images` | *(unset)* | Drop classes with fewer than this many images. |
| `manual_include_classes` | *(unset)* | Class names to keep even if below `min_class_images`. |
| `sweep_params` | *(unset)* | Map of field → list of values to grid-search over (see below). |
| `plots` | `false` | Generate evaluation plots after training (`--plots`). |

### Hyperparameter sweeps

`sweep_params` grids over any combination of training fields, running one job per
combination:

```yaml
sweep_params:
  lr: [0.001, 0.0001]
  batch_size: [64]
  model: [resnet50]
  transform: [dataset_squarepad, dataset_fullpad]
  epochs: [5]
```

## Inference parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_path` | *(required)* | Bin file or directory of raw bins (`--input`). |
| `model_checkpoint` | *(required)* | Path to the `*_best.pt` checkpoint (`--model`). |
| `output_dir` | `output/class_scores` | Where the per-bin class-scores files are written; the suffix depends on `output_format` (`--output`). |
| `batch_size` | `64` | Images per forward pass. |
| `num_workers` | `0` | Data-loading worker processes. |
| `thresholds_path` | *(auto)* | Per-class thresholds (`.json` from training, or `.yaml`); auto-detected next to the checkpoint. |
| `threshold_default` | `0.0` | Per-class decision threshold used when no thresholds file is supplied or auto-detected. |
| `device` | `auto` | `auto`, `cpu`, or `cuda`. |
| `classifier_name` | *(unset)* | Name recorded in the output metadata. |
| `overwrite` | `false` | Re-generate bins whose output already exists (`--overwrite`). |
| `classes_path` | *(auto)* | Class list; auto-detected next to the checkpoint, or supply for legacy checkpoints. |
| `model_name` | *(unset)* | Architecture name for legacy checkpoints (e.g. `resnet50`). |
| `num_threads` | *(all cores)* | Cap CPU threads for inference. |
| `allow_unsafe` | `false` | Permit loading legacy raw-state-dict checkpoints (`--allow-unsafe`). |
| `chain_counting` | *(unset)* | Chain-counting block; see [Chain counting](guides/chain-counting.md#counting-during-inference). YAML only. |
| `output_format` | `h5` | Class-scores file format(s): `h5`, `csv`, `mat`, `csv-labels`, a comma-separated list (`h5,csv-labels`), a YAML list, or `all` (`--format`). See [Inference → Output formats](guides/inference.md#output-formats). |

## Date placeholders

String values in YAML configs, and CLI overrides, support date placeholders that
are expanded at load time (UTC). This is useful for continuous inference
pipelines where input/output directories are organised by date.

| Placeholder | Example value | Description |
|-------------|---------------|-------------|
| `{year}` | `2026` | Four-digit year |
| `{month}` | `03` | Zero-padded month |
| `{day}` | `14` | Zero-padded day |
| `{date}` | `20260314` | Combined `YYYYMMDD` |

Example `infer.yaml`:

```yaml
input_path: /ifcb/data/{year}
output_dir: /ifcb/output/{year}
```

The chain-counting configs (`chains_train_default.yaml`, `chains_eval_default.yaml`)
have their own options. See [Chain counting](guides/chain-counting.md).
