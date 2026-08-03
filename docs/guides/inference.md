# Inference

Batch-classify raw IFCB bins and write class-score files (HDF5 by default; see
[Output formats](#output-formats)).

## On a directory of bins

Point inference at a directory of raw IFCB bins (`.roi/.adc/.hdr`):

```bash
python -m ifcb_classify infer \
    --input /path/to/bins \
    --model output/<run>_best.pt \
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

The architecture is guessed from the checkpoint's layer shapes, but only
`resnet50` and `efficientnet_v2_s` are recognised and anything else falls back to
`resnet50`. If loading then fails with missing or unexpected state-dict keys, name
the architecture explicitly with `--model-name` (e.g. `--model-name resnet18`).

## Output

By default, output is one `{sample}_class.h5` file per bin, in IFCB Dashboard
class_scores v3 format, compatible with the IFCB Dashboard,
[iRfcb](https://europeanifcbgroup.github.io/iRfcb/) and
[ClassiPyR](https://europeanifcbgroup.github.io/ClassiPyR/).

### Output formats

`--format` (or `output_format:` in the config) selects which class-scores file(s)
to write per bin. HDF5 is the default; you can also write CSV, a `.mat` file, or
several at once:

| Value | File | Notes |
|---|---|---|
| `h5` (default) | `{sample}_class.h5` | class_scores **v3** HDF5. The format the IFCB Dashboard accessions. Carries scores, per-class thresholds, resolved labels, and, when counting, a `cell_count` dataset (see [chain counting](chain-counting.md)). |
| `csv` | `{sample}_class.csv` | The dashboard's per-ROI class-scores export format: one row per ROI, indexed by `pid` (`{sample}_{roi:05d}`), one column per class holding the score. Scores only, with no resolved label. |
| `mat` | `{sample}_class_v1.mat` | class_scores **v1** MATLAB file, ingestible by the dashboard (pyifcb's v1 reader) *and* processable by [iRfcb](https://europeanifcbgroup.github.io/iRfcb/) (`ifcb_extract_biovolumes`, `ifcb_summarize_class_counts`, `ifcb_summarize_cell_counts`). Carries scores, resolved/thresholded classes (`TBclass`, `TBclass_above_threshold`), the classifier name, and, when counting, `cell_count`. |
| `csv-labels` | `{sample}.csv` | The ClassiPyR/iRfcb per-ROI resolved-label CSV: columns `file_name`, `class_name` (thresholded), `class_name_auto` (argmax), `score` (winning confidence), and, when counting, `cell_count`. Read by iRfcb's summarisers. Named `{sample}.csv` (no `_class` suffix) so iRfcb resolves the sample correctly. |

```bash
# a single alternative format
python -m ifcb_classify infer --input /path/to/bins --model output/<run>_best.pt \
    --output /path/to/class_scores --format csv

# several at once (comma-separated, or "all")
python -m ifcb_classify infer --config configs/infer_default.yaml --format h5,csv-labels
```

```yaml
# in an inference config
output_format: csv-labels   # h5 (default) | csv | mat | csv-labels | h5,csv-labels | all
```

!!! note "Two different CSVs: pick the right one"
    `csv` is the dashboard scores export: `pid` + one score column per class,
    no resolved label (iRfcb cannot read it). `csv-labels` is the
    ClassiPyR/iRfcb CSV: the resolved `class_name`, `class_name_auto`, winning
    `score`, and `cell_count`. Use this one if you want per-ROI labels
    or to feed iRfcb from CSV.

!!! note "Where each field lives"
    Resolved/thresholded classes and the classifier name are in `h5`, `mat`, and
    `csv-labels`; the plain `csv` is scores-only. **Chain counts** are written to
    `h5`, `mat`, and `csv-labels` (not the scores `csv`), so include one of those
    in the format list when counting.

!!! warning "One format per directory for downstream tools"
    Writing several formats sends multiple class-scores files for the *same* bin
    into one output directory (e.g. `{sample}_class.h5` and
    `{sample}_class_v1.mat`). Tools that scan a directory for class files reject
    this: iRfcb's summarisers, for example, abort with *"samples resolve to more
    than one classification file … supply a single file format per sample to avoid
    double-counting."* If you plan to point such a tool at the output, write a
    single format there, or move the extra formats into separate directories
    first.

!!! info "What the dashboard actually ingests"
    The IFCB Dashboard accessions HDF5/MAT class-scores files, not CSV, which is
    a download/interchange format it exports. So keep `h5` (or `mat`) as the file
    you load into the dashboard; `csv` is for spreadsheets, R/pandas, and the like.
    The dashboard discovers `.mat` files via pyifcb's v1 reader, which prefers a
    `class{year}_v1/` subdirectory before falling back to an exhaustive search, so
    point its class-scores directory at these files accordingly (or arrange them in
    the year subfolder).

!!! note "Why the `.mat` fields are named `TB…`"
    The v1 `.mat` field names (`class2useTB`, `TBscores`, `TBclass`,
    `TBclass_above_threshold`) carry the `TB` for TreeBagger, the MATLAB
    random-forest classifier from Sosik & Olson's
    [`ifcb-analysis`](https://github.com/hsosik/ifcb-analysis) pipeline, where this
    format originated. This classifier is a PyTorch CNN, not a TreeBagger, so the
    names are historical, but pyifcb, the IFCB Dashboard, and iRfcb all read these
    exact names, so they are kept verbatim as a compatibility contract (the `h5`
    and `csv-labels` formats use their own, non-`TB` names). Renaming them would
    stop the data being picked up downstream.

To also count cells in chain-forming taxa during inference, see
[Chain counting](chain-counting.md).
