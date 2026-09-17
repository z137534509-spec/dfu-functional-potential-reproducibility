# v1.1.0 - targeted IJLEW revision release

This version retains the locked primary analysis and all previously reported numerical results. It adds a reviewer-requested symmetric exact-visit-pair sensitivity and a two-sided design-preserving randomization sensitivity.

## Additions

- Exact unordered visit-pair matching: each within-patient profile pair is compared with external profile pairs at the identical visit-ordinal pair, with external pair members from two different nonfocal patients.
- A positive patient-level median contrast of 0.01426 (95% patient-level bootstrap confidence interval, 0.00554 to 0.02760); 33 of 44 patient-level contrasts are positive.
- A 9,999-replicate, two-sided design-preserving randomization sensitivity with P=0.0001.
- Updated analysis script, derived tables, results summaries, and Figures 2 and 4.

## Unchanged primary analysis

The primary same-or-adjacent-visit Bray-Curtis estimand remains 0.01535 (95% full-recomputation patient-cluster-bootstrap confidence interval, 0.00757 to 0.02828); 36 of 44 patient-level contrasts are positive, and both the directional primary and two-sided randomization sensitivity give P=0.0001.

## Interpretation boundary

The release documents a patient-specific structural observation for broad DNA-level microbial functional potential in one public cohort. It does not establish functional activity, biological mechanism, healing relevance, treatment effects, clinical utility, or external validity.
