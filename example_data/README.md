# Example data

A small, curated subset of IFCB plankton ROI images, provided so you can run the
[Getting started](../docs/getting-started.md) walkthrough end to end without
having to build your own labelled dataset first.

`plankton/` holds six morphologically distinct classes with 40 images each (240
images, ~1.2 MB), organised as one folder per class — the exact layout the
`train` command expects:

```
example_data/plankton/
  Guinardia_delicatula/
  Cryptomonadales/
  Leptocylindrus_danicus_minimus/
  Scrippsiella_group/
  Heterocapsa_rotundata/
  Cylindrotheca_Nitzschia_longissima/
```

This is a **demo** dataset — enough to see the pipeline train and classify, not
enough to build an accurate model. For a real model you want many more images
per class (hundreds to thousands) and more classes.

## Raw bin (`bins/`)

`bins/` holds one raw IFCB **bin** — the `.roi/.adc/.hdr` triple
`D20230314T003836_IFCB134` — so the inference step of the walkthrough is runnable
too. Classification and chain counting read bins directly (not the ROI PNGs
above). A toy model trained on six classes won't classify a real bin
*accurately*, but it will produce a valid `class_scores` HDF5 file so you can see
inference work end to end.

## Source and citation

These images are a subset of the SMHI IFCB plankton image reference library.
If you use them, please cite:

> Torstensson, Anders; Skjevik, Ann-Turi; Mohlin, Malin; Karlberg, Maria;
> Karlson, Bengt (2024). *SMHI IFCB plankton image reference library.* Swedish
> Meteorological and Hydrological Institute (SMHI). Dataset.
> <https://doi.org/10.17044/scilifelab.25883455.v3>

The full library (all classes, many more images) is available at the DOI above.
