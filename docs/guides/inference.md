# Inference

Batch-classify raw IFCB bins and write HDF5 class-score files.

## On a directory of bins

Point inference at a directory of raw IFCB bins (`.roi/.adc/.hdr`):

```bash
python -m ifcb_classify infer \
    --input /path/to/bins \
    --model output/model_best.pt \
    --output /path/to/class_scores
```

Or with a config file:

```bash
python -m ifcb_classify infer --config configs/infer_default.yaml
```

## Legacy checkpoints

Legacy checkpoints (raw state dicts saved outside this pipeline) require unsafe
pickle loading. Add `--allow-unsafe` to permit this, and supply the class list:

```bash
python -m ifcb_classify infer \
    --input /path/to/bins \
    --model /path/to/legacy_model.pt \
    --classes /path/to/classes.txt \
    --allow-unsafe
```

## Output

Output is one `{sample}_class.h5` file per bin, in IFCB Dashboard class_scores v3
format — compatible with the IFCB Dashboard,
[iRfcb](https://europeanifcbgroup.github.io/iRfcb/) and
[ClassiPyR](https://europeanifcbgroup.github.io/ClassiPyR/).

To also count cells in chain-forming taxa during inference, see
[Chain counting](chain-counting.md).
