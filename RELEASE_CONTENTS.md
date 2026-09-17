# Release contents and disclosure boundary

## Version

`v1.1.0`, released 2026-09-17.

## Included

- One locked Python analysis script, updated with a symmetric exact-visit-pair sensitivity and two-sided randomization sensitivity.
- Twenty-three nonidentifying derived CSV tables, including profile- and patient-level matched-distance quantities, bootstrap distributions, and randomization null distributions.
- Sixteen result CSV/JSON summary files.
- Eight generated analysis figures, including sampling architecture, matched functional-potential contrast, patient-identity correction, sensitivity/comparator availability, and supporting diagnostics.
- Software environment, citation, Zenodo metadata, data-provenance, and licensing files.

## Excluded

- Original `SEED.sub1.map.csv` and `metamap.csv` source tables.
- Raw sequence data, clinical free text, direct identifiers, author correspondence beyond the public repository contact, manuscript submission files, peer-review materials, and local absolute paths.

## Why the original source tables are excluded

They originate from a prior public study and are accessed through the original BioProject and associated supplementary material. They are not necessary for readers to inspect the released derived outputs, and this repository does not purport to redistribute or relicense them.

## Interpretation boundary

The released outputs document patient-specific structure of broad, DNA-level functional potential relative to visit-stage-matched external comparators in one public cohort. They include a symmetric exact-visit-pair sensitivity but do not demonstrate functional activity, molecular mechanism, healing relevance, treatment effects, clinical utility, or external validity.
