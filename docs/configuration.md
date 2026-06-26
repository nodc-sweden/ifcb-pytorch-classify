# Configuration

See `configs/train_default.yaml` and `configs/infer_default.yaml` for all
available options.

## Key training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `resnet50` | Model architecture (see the [`models.registry`](reference/ifcb_classify/models/registry.md) module for the full list) |
| `transform` | `dataset_squarepad_augmented` | Image preprocessing pipeline |
| `lr` | `0.0001` | Learning rate |
| `batch_size` | `64` | Batch size |
| `epochs` | `20` | Number of training epochs |
| `checkpoint_metric` | `weighted_f1` | Metric used for best-model checkpointing |
| `tracker` | `csv` | Experiment tracker (`csv`, `mlflow`, `wandb`, `none`) |
| `plots` | `false` | Generate evaluation plots after training |

## Date placeholders

Path values in YAML configs support date placeholders that are expanded at load
time (UTC). This is useful for continuous inference pipelines where input/output
directories are organised by date.

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
