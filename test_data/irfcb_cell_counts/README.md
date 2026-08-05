# Chain-count test fixture for iRfcb `ifcb_summarize_cell_counts` (.mat support)

A curated **subset of real inference output**, for developing/testing `.mat`
support in iRfcb's `ifcb_summarize_cell_counts`. Two bins from
`D20230314 / IFCB134` were classified by **`SMHI-NIVA-SYKE-SAMS-SZN-ResNet50-V6`**
with per-class YOLO chain counting; the rows kept here carry that inference's own
ROI numbers, scores, thresholded classes and `cell_count` values **verbatim**.
Nothing is hand-authored.

> This matters. An earlier version of this fixture used invented class labels and
> `cell_count` values. Because the numbers looked biological, they were reasoned
> from during the iRfcb 0.10.0 work — a per-cell carbon calculation was scoped
> partly on them and had to be reversed once real classifications were checked.
> The fixture is now derived from real files precisely so that can't recur: the
> generator can only carry real rows across, never manufacture them.

Each bin is written in two formats with **identical underlying data**:

| File | Purpose |
|---|---|
| `{lid}_class_v1.mat` | The file under test — carries `cell_count` (int32, `-1` sentinel) alongside `class2useTB`, `TBscores`, `roinum`, `TBclass`, `TBclass_above_threshold`, `classifierName`. |
| `{lid}_class.h5` | The class_scores v3 HDF5 with the same `cell_count` dataset, plus the full 176-class score matrix, thresholds, and the `cell_counter_models` provenance attribute. It is the source of truth the `.mat` is derived from. |

CSV is intentionally not included: this pipeline's CSV is the dashboard's
scores-only export and carries no `cell_count`.

> **Note:** this folder holds both `.mat` and `.h5` for the *same* bins, so pass a
> single format's files (e.g. `list.files(..., pattern = "_class_v1.mat$")`) to
> `ifcb_summarize_cell_counts()` — pointing it at the whole folder errors with
> *"samples resolve to more than one classification file"*.

## Why a subset, and how it was chosen

The full bins are 1218 and 1122 ROIs (~1.8 MB each). This fixture keeps 15 and 18
ROIs — every row real, each `roinum` unchanged so it still resolves to an actual
image in the raw `.roi`. The selection was made to retain the fixture's full
discriminating power, covering each case the summariser must handle:

- a long chain (15 cells), mid chains, and single cells;
- a counted-but-empty `0` box (only `003836` has one in the real data — the
  counter never returned an empty box in `001205`, so this bin has none);
- `-1` not-counted ROIs across several taxa with no configured detector;
- threshold-demoted `unclassified` rows (their argmax winner is a real class —
  `Unicells`, `Heterosigma-like` — pushed below its threshold), all with
  `cell_count == -1`.

No automated test asserts specific ROI numbers, so the subset breaks nothing:
iRfcb's `test-ifcb_summarize_cell_counts.R` builds its own throwaway `.h5`/`.mat`
inline, and this pipeline has no test that reads this folder. The fixture is here
for interactive/format verification.

## Per-ROI data

`cell_count` semantics: number of cells the chain counter found in the ROI;
`-1` = ROI not chain-counted (class not configured for counting); `0` = counted
but no cells detected.

### `D20230314T001205_IFCB134` (15 ROIs)

| roi | class (`TBclass_above_threshold`) | cell_count | note |
|----:|---|----:|---|
| 2 | Pyramimonas_spp_small | -1 | not chain-counted (no detector for this class) |
| 3 | Guinardia_delicatula | 1 | single cell |
| 4 | Guinardia_delicatula | 2 | 2-cell chain |
| 5 | Unicells | -1 | not chain-counted (no detector for this class) |
| 8 | Thalassiosira_nordenskioeldii | 2 | 2-cell chain |
| 10 | unclassified | -1 | winner (Unicells) below its threshold → unclassified |
| 11 | Cryptophyta | -1 | not chain-counted (no detector for this class) |
| 24 | unclassified | -1 | winner (Unicells) below its threshold → unclassified |
| 122 | Guinardia_delicatula | 4 | 4-cell chain |
| 135 | Guinardia_delicatula | 3 | 3-cell chain |
| 143 | Thalassiosira_levanderi | 1 | single cell |
| 447 | Thalassiosira_nordenskioeldii | 15 | 15-cell chain |
| 601 | Guinardia_delicatula | 6 | 6-cell chain |
| 660 | Thalassiosira_nordenskioeldii | 8 | 8-cell chain |
| 689 | Guinardia_delicatula | 5 | 5-cell chain |

### `D20230314T003836_IFCB134` (18 ROIs)

| roi | class (`TBclass_above_threshold`) | cell_count | note |
|----:|---|----:|---|
| 2 | Dinophyceae_smaller_than_30 | -1 | not chain-counted (no detector for this class) |
| 3 | Unicells | -1 | not chain-counted (no detector for this class) |
| 6 | Cryptophyta | -1 | not chain-counted (no detector for this class) |
| 9 | Guinardia_delicatula | 1 | single cell |
| 16 | Guinardia_delicatula | 2 | 2-cell chain |
| 19 | unclassified | -1 | winner (Unicells) below its threshold → unclassified |
| 24 | unclassified | -1 | winner (Heterosigma-like) below its threshold → unclassified |
| 62 | Guinardia_delicatula | 3 | 3-cell chain |
| 86 | Thalassiosira_levanderi | 1 | single cell |
| 93 | Guinardia_delicatula | 8 | 8-cell chain |
| 100 | Skeletonema_spp | 0 | counted, none detected |
| 194 | Guinardia_delicatula | 5 | 5-cell chain |
| 201 | Guinardia_delicatula | 4 | 4-cell chain |
| 337 | Chaetoceros_spp_chain | 2 | 2-cell chain |
| 504 | Guinardia_delicatula | 6 | 6-cell chain |
| 644 | Chaetoceros_tenuissimus-like | 0 | counted, none detected |
| 873 | Guinardia_delicatula | 7 | 7-cell chain |
| 1025 | Chaetoceros_spp_chain | 4 | 4-cell chain |

The `unclassified` rows exercise the threshold path: `TBclass` (the argmax winner)
is a real class, but `TBclass_above_threshold` is `unclassified` because the score
fell below that class's threshold. In real output every such ROI is `-1`
(the counter keys on the thresholded name, so a counted `unclassified` is
impossible) — the generator asserts this invariant.

## Golden expected output

This is the **actual** output of `ifcb_summarize_cell_counts()` (iRfcb 0.10.0)
run on the `.h5` files, with default `single_cell_values = c(-1, 0)` (each
resolved to 1 cell) and grouping by class. Running the function on the `.mat`
files reproduces it exactly (`all.equal()` on every column but `classifier`).

| sample | class | counts | cell_counts | n_chains | mean_chain_length | median | max |
|---|---|--:|--:|--:|--:|--:|--:|
| D20230314T001205_IFCB134 | Cryptophyta | 1 | 1 | 0 | NA | NA | NA |
| D20230314T001205_IFCB134 | Guinardia_delicatula | 6 | 21 | 6 | 3.500 | 3.5 | 6 |
| D20230314T001205_IFCB134 | Pyramimonas_spp_small | 1 | 1 | 0 | NA | NA | NA |
| D20230314T001205_IFCB134 | Thalassiosira_levanderi | 1 | 1 | 1 | 1.000 | 1.0 | 1 |
| D20230314T001205_IFCB134 | Thalassiosira_nordenskioeldii | 3 | 25 | 3 | 8.333 | 8.0 | 15 |
| D20230314T001205_IFCB134 | Unicells | 1 | 1 | 0 | NA | NA | NA |
| D20230314T001205_IFCB134 | unclassified | 2 | 2 | 0 | NA | NA | NA |
| D20230314T003836_IFCB134 | Chaetoceros_spp_chain | 2 | 6 | 2 | 3.000 | 3.0 | 4 |
| D20230314T003836_IFCB134 | Chaetoceros_tenuissimus-like | 1 | 1 | 0 | NA | NA | NA |
| D20230314T003836_IFCB134 | Cryptophyta | 1 | 1 | 0 | NA | NA | NA |
| D20230314T003836_IFCB134 | Dinophyceae_smaller_than_30 | 1 | 1 | 0 | NA | NA | NA |
| D20230314T003836_IFCB134 | Guinardia_delicatula | 8 | 36 | 8 | 4.500 | 4.5 | 8 |
| D20230314T003836_IFCB134 | Skeletonema_spp | 1 | 1 | 0 | NA | NA | NA |
| D20230314T003836_IFCB134 | Thalassiosira_levanderi | 1 | 1 | 1 | 1.000 | 1.0 | 1 |
| D20230314T003836_IFCB134 | Unicells | 1 | 1 | 0 | NA | NA | NA |
| D20230314T003836_IFCB134 | unclassified | 2 | 2 | 0 | NA | NA | NA |

`counts` = ROIs in the class; `cell_counts` = total resolved cells (`-1`/`0` →
1); `n_chains` = ROIs with `cell_count >= 1`, over which the chain-length stats
are computed.

## Regenerate

The subset `.h5` files are **committed** (real, ~50 KB each) as the fixture's
source of truth — the fixture is self-contained in the repo. The rest of
`test_data/` is git-ignored; these files are whitelisted in `.gitignore`.
`scripts/make_irfcb_cell_count_fixture.py` regenerates the `.mat` from the `.h5`.

```sh
# default: rebuild the .mat from the subset .h5 already in this folder
python scripts/make_irfcb_cell_count_fixture.py

# re-curate the subset from the full real bins (only to change which ROIs are kept)
python scripts/make_irfcb_cell_count_fixture.py --source /path/to/full/bins
```

With no `--source` it reads the subset `{lid}_class.h5` here and just refreshes the
`.mat` — no multi-MB originals needed, `.h5` left untouched. `--source` (a caller's
argument, never hard-coded) points at the full bins and re-selects the ROIs in the
script's `KEEP`. In neither mode can it fabricate data. Run with a Python env that
has this package installed.
