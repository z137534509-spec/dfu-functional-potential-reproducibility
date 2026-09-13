#!/usr/bin/env python
"""Formal, locked analysis of patient-specific DFU functional-potential structure."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon


if len(sys.argv) != 3:
    raise SystemExit("Usage: 03_run_functional_stability_full_analysis.py <source_data_dir> <output_root>")

SOURCE = Path(sys.argv[1])
OUTROOT = Path(sys.argv[2])
DERIVED = OUTROOT / "derived"
RESULTS = OUTROOT / "results"
FIGURES = OUTROOT / "figures"
for directory in (DERIVED, RESULTS, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260909
# The primary randomization test explicitly preserves the observed visit-stage
# sampling structure. Covariate-label checks remain supplementary diagnostics.
N_DESIGN_PERMUTATIONS = 9999
N_COVARIATE_PERMUTATIONS = 999
N_BOOTSTRAP = 2000
N_CLUSTER_BOOTSTRAP = 4999
RNG = np.random.default_rng(SEED)


def read_csv(name: str, usecols=None) -> pd.DataFrame:
    frame = pd.read_csv(SOURCE / name, usecols=usecols, low_memory=False)
    return frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]


def boolean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def normalise(matrix: pd.DataFrame) -> pd.DataFrame:
    matrix = matrix.astype(float).clip(lower=0)
    matrix = matrix.loc[matrix.sum(axis=1) > 0].copy()
    return matrix.div(matrix.sum(axis=1), axis=0)


def clr_pseudocount(matrix: pd.DataFrame) -> float:
    values = matrix.to_numpy(float)
    positive = values[values > 0]
    if not len(positive):
        raise ValueError("CLR requires at least one positive abundance.")
    return max(float(np.min(positive)) * 0.5, 1e-10)


def clr(matrix: pd.DataFrame, pseudocount: float) -> np.ndarray:
    """CLR after a fixed additive pseudocount for zero-containing profiles."""
    values = matrix.to_numpy(float)
    logged = np.log(values + pseudocount)
    return logged - logged.mean(axis=1, keepdims=True)


def safe_wilcoxon(values: pd.Series | np.ndarray) -> float:
    values = pd.Series(values, dtype=float).dropna()
    nonzero = values.loc[values != 0]
    if len(nonzero) == 0:
        return 1.0 if len(values) else np.nan
    return float(wilcoxon(values, alternative="two-sided", method="auto").pvalue)


def bootstrap_median_ci(values: pd.Series | np.ndarray) -> tuple[float, float]:
    values = pd.Series(values, dtype=float).dropna().to_numpy()
    if len(values) == 0:
        return np.nan, np.nan
    medians = np.median(RNG.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True), axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def summarize_patient_gaps(patient_metrics: pd.DataFrame, label: str) -> dict:
    values = patient_metrics["between_minus_within_distance"].dropna().astype(float)
    low, high = bootstrap_median_ci(values)
    return {
        "analysis": label,
        "n_patients": int(len(values)),
        "n_samples_contributing": int(patient_metrics["n_samples_contributing"].sum()),
        "median_between_minus_within_distance": float(values.median()) if len(values) else np.nan,
        "ci95_low": low,
        "ci95_high": high,
        "directional_proportion_positive": float((values > 0).mean()) if len(values) else np.nan,
        "wilcoxon_two_sided_p": safe_wilcoxon(values),
    }


def matched_individuality(
    distance_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    label: str,
    max_visit_delta: int = 1,
    eligible_mask: pd.Series | None = None,
    same_antibiotic_state: bool = False,
    depth_tolerance_cm: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Derive one matched-between-minus-within median distance per patient."""
    meta = metadata.loc[distance_matrix.index].copy()
    if eligible_mask is None:
        eligible_mask = pd.Series(True, index=meta.index)
    eligible_ids = meta.index[eligible_mask.reindex(meta.index).fillna(False)].tolist()
    meta = meta.loc[eligible_ids]
    sample_rows: list[dict] = []
    for sample_id, row in meta.iterrows():
        own = meta.index[(meta.patient_id == row.patient_id) & (meta.index != sample_id)]
        external = meta.loc[(meta.patient_id != row.patient_id) & ((meta.visit - row.visit).abs() <= max_visit_delta)]
        if same_antibiotic_state:
            own = meta.index[(meta.patient_id == row.patient_id) & (meta.index != sample_id) & (meta.antibiotic_exposed == row.antibiotic_exposed)]
            external = external.loc[external.antibiotic_exposed == row.antibiotic_exposed]
        if depth_tolerance_cm is not None:
            external = external.loc[(external.depth_cm - row.depth_cm).abs() <= depth_tolerance_cm]
        if not len(own) or external.empty:
            continue
        within = distance_matrix.loc[sample_id, own].to_numpy(float)
        between = distance_matrix.loc[sample_id, external.index].to_numpy(float)
        sample_rows.append({
            "analysis": label, "patient_id": int(row.patient_id), "sample_id": sample_id,
            "visit": int(row.visit), "antibiotic_exposed": bool(row.antibiotic_exposed), "depth_cm": float(row.depth_cm),
            "n_within_candidates": int(len(within)), "n_matched_between_candidates": int(len(between)),
            "n_matched_between_patients": int(external.patient_id.nunique()),
            "median_within_distance": float(np.median(within)), "median_matched_between_distance": float(np.median(between)),
            "between_minus_within_distance": float(np.median(between) - np.median(within)),
        })
    sample_metrics = pd.DataFrame(sample_rows)
    if sample_metrics.empty:
        return sample_metrics, pd.DataFrame(), {"analysis": label, "n_patients": 0}
    patient_metrics = sample_metrics.groupby("patient_id", as_index=False).agg(
        n_samples_contributing=("sample_id", "size"),
        median_within_distance=("median_within_distance", "median"),
        median_matched_between_distance=("median_matched_between_distance", "median"),
        between_minus_within_distance=("between_minus_within_distance", "median"),
    )
    patient_metrics.insert(0, "analysis", label)
    return sample_metrics, patient_metrics, summarize_patient_gaps(patient_metrics, label)


def lopo_matched(distance_matrix: pd.DataFrame, metadata: pd.DataFrame, label: str, **kwargs) -> pd.DataFrame:
    rows = []
    for patient in sorted(metadata.patient_id.unique()):
        keep = metadata.index[metadata.patient_id != patient]
        _, _, summary = matched_individuality(distance_matrix.loc[keep, keep], metadata.loc[keep], label, **kwargs)
        rows.append({"analysis": label, "omitted_patient": int(patient), **summary})
    return pd.DataFrame(rows)


def stage_preserving_patient_labels(labels: np.ndarray, visits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Break longitudinal identity while retaining every visit's observed label set."""
    permuted = labels.copy()
    for visit in np.unique(visits):
        indices = np.flatnonzero(visits == visit)
        permuted[indices] = rng.permutation(labels[indices])
    return permuted


def matched_primary_statistic_from_labels(
    distance: np.ndarray,
    visits: np.ndarray,
    labels: np.ndarray,
    max_visit_delta: int = 1,
    source_patient_labels: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Recompute the complete patient-level primary statistic for a label assignment.

    ``source_patient_labels`` is only needed for patient-cluster bootstrap
    replicates. It preserves source-patient exclusion when one original patient
    is sampled more than once: separate bootstrap copies remain separate
    clusters for their within-patient contrast, but cannot become one another's
    external comparators.
    """
    if source_patient_labels is not None and len(source_patient_labels) != len(labels):
        raise ValueError("source_patient_labels must align with labels.")
    per_patient: dict[int, list[float]] = {}
    stage_masks = [np.abs(visits - visit) <= max_visit_delta for visit in visits]
    for sample_index, label in enumerate(labels):
        own = labels == label
        own[sample_index] = False
        external = (labels != label) & stage_masks[sample_index]
        if source_patient_labels is not None:
            external &= source_patient_labels != source_patient_labels[sample_index]
        if not own.any() or not external.any():
            continue
        gap = float(np.median(distance[sample_index, external]) - np.median(distance[sample_index, own]))
        per_patient.setdefault(int(label), []).append(gap)
    patient_medians = np.asarray([np.median(values) for values in per_patient.values()], dtype=float)
    return float(np.median(patient_medians)), patient_medians


def primary_design_preserving_permutation(
    distance_matrix: pd.DataFrame, metadata: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Monte Carlo inference that keeps visit-stage label sets but breaks patient trajectories."""
    meta = metadata.loc[distance_matrix.index]
    if meta.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Within-visit label permutation requires at most one profile per patient and visit.")
    distance = distance_matrix.to_numpy(float)
    visits = meta.visit.to_numpy(int)
    labels = meta.patient_id.to_numpy(int)
    observed, observed_patient_medians = matched_primary_statistic_from_labels(distance, visits, labels)
    null = np.empty(N_DESIGN_PERMUTATIONS, dtype=float)
    rng = np.random.default_rng(SEED + 101)
    for index in range(N_DESIGN_PERMUTATIONS):
        permuted_labels = stage_preserving_patient_labels(labels, visits, rng)
        null[index], permuted_patient_medians = matched_primary_statistic_from_labels(distance, visits, permuted_labels)
        if len(permuted_patient_medians) != len(observed_patient_medians):
            raise RuntimeError("Stage-preserving permutation changed the number of contributing patient labels.")
    p_upper = float((1 + np.sum(null >= observed)) / (1 + len(null)))
    summary = {
        "analysis": "primary_bray_same_or_adjacent_visit_design_preserving_permutation",
        "n_design_permutations": int(N_DESIGN_PERMUTATIONS),
        "n_contributing_patients": int(len(observed_patient_medians)),
        "observed_median_between_minus_within_distance": observed,
        "permuted_null_median": float(np.median(null)),
        "permuted_null_q025": float(np.quantile(null, 0.025)),
        "permuted_null_q975": float(np.quantile(null, 0.975)),
        "observed_minus_permuted_null_median": float(observed - np.median(null)),
        "upper_tail_randomization_p": p_upper,
        "permutation_scheme": "patient labels independently permuted within visit ordinal; each visit retains its observed patient-label set and sample count",
    }
    return summary, pd.DataFrame({"permutation_index": np.arange(1, N_DESIGN_PERMUTATIONS + 1), "null_statistic": null})


def primary_full_recomputation_cluster_bootstrap(
    distance_matrix: pd.DataFrame, metadata: pd.DataFrame, seed_offset: int = 211,
    analysis_label: str = "primary_bray_same_or_adjacent_visit_full_recomputation_cluster_bootstrap",
    inferential_role: str = "uncertainty interval only; primary null-hypothesis inference uses the separate within-visit patient-label permutation test",
) -> tuple[dict, pd.DataFrame]:
    """Estimate the primary-estimand interval by resampling patient clusters.

    Each sampled patient copy receives a distinct bootstrap cluster label. The
    complete matched statistic, including comparator pools and patient medians,
    is recalculated in every replicate. When an original patient is sampled
    more than once, copies of that original patient are excluded from one
    another's external-comparator pools. This interval quantifies empirical
    resampling uncertainty; the visit-preserving permutation remains the sole
    primary null-hypothesis test.
    """
    meta = metadata.loc[distance_matrix.index]
    original_labels = meta.patient_id.to_numpy(int)
    visits = meta.visit.to_numpy(int)
    distance = distance_matrix.to_numpy(float)
    patients = np.unique(original_labels)
    observed, observed_patient_medians = matched_primary_statistic_from_labels(distance, visits, original_labels)
    sample_indices_by_patient = {
        patient: np.flatnonzero(original_labels == patient) for patient in patients
    }
    rng = np.random.default_rng(SEED + seed_offset)
    statistics = np.empty(N_CLUSTER_BOOTSTRAP, dtype=float)
    contributing_clusters = np.empty(N_CLUSTER_BOOTSTRAP, dtype=int)
    same_original_copy_audit_rows: list[dict] = []

    for replicate in range(N_CLUSTER_BOOTSTRAP):
        selected_patients = rng.choice(patients, size=len(patients), replace=True)
        source_indices = np.concatenate([sample_indices_by_patient[patient] for patient in selected_patients])
        bootstrap_labels = np.concatenate([
            np.full(len(sample_indices_by_patient[patient]), copy_index, dtype=int)
            for copy_index, patient in enumerate(selected_patients)
        ])
        bootstrap_visits = visits[source_indices]
        bootstrap_source_patient_labels = original_labels[source_indices]
        bootstrap_distance = distance[np.ix_(source_indices, source_indices)]
        statistics[replicate], replicate_patient_medians = matched_primary_statistic_from_labels(
            bootstrap_distance,
            bootstrap_visits,
            bootstrap_labels,
            source_patient_labels=bootstrap_source_patient_labels,
        )
        contributing_clusters[replicate] = len(replicate_patient_medians)
        sampled_originals, original_copy_counts = np.unique(selected_patients, return_counts=True)
        repeated_originals = original_copy_counts > 1
        repeated_patient_ids = sampled_originals[repeated_originals]
        copies_from_repeated_originals = int(original_copy_counts[repeated_originals].sum())
        profiles_from_repeated_original_copies = int(sum(
            len(sample_indices_by_patient[patient]) * count
            for patient, count in zip(sampled_originals[repeated_originals], original_copy_counts[repeated_originals])
        ))
        # Before this exclusion, every duplicated focal profile would have had
        # an exact same-profile, zero-distance external copy at the same visit.
        pre_exclusion_zero_distance_external_pairs = int(sum(
            len(sample_indices_by_patient[patient]) * count * (count - 1)
            for patient, count in zip(sampled_originals[repeated_originals], original_copy_counts[repeated_originals])
        ))
        same_original_copy_audit_rows.append({
            "bootstrap_replicate": int(replicate + 1),
            "n_unique_source_patients": int(len(sampled_originals)),
            "n_source_patients_selected_more_than_once": int(repeated_originals.sum()),
            "n_bootstrap_copies_from_repeated_source_patients": copies_from_repeated_originals,
            "n_profiles_from_repeated_source_patient_copies": profiles_from_repeated_original_copies,
            "n_pre_exclusion_zero_distance_external_candidates": pre_exclusion_zero_distance_external_pairs,
        })

    same_original_copy_audit = pd.DataFrame(same_original_copy_audit_rows)

    summary = {
        "analysis": analysis_label,
        "n_cluster_bootstrap_replicates": int(N_CLUSTER_BOOTSTRAP),
        "original_n_patient_clusters": int(len(patients)),
        "original_n_contributing_patient_clusters": int(len(observed_patient_medians)),
        "observed_median_between_minus_within_distance": float(observed),
        "ci95_low": float(np.quantile(statistics, 0.025)),
        "ci95_high": float(np.quantile(statistics, 0.975)),
        "bootstrap_median": float(np.median(statistics)),
        "bootstrap_contributing_clusters_median": float(np.median(contributing_clusters)),
        "bootstrap_contributing_clusters_min": int(np.min(contributing_clusters)),
        "bootstrap_contributing_clusters_max": int(np.max(contributing_clusters)),
        "bootstrap_scheme": f"resample {len(patients)} patient clusters with replacement; assign every selected copy a distinct bootstrap cluster label; rebuild the full within-patient and visit-stage-matched between-patient comparator pools; exclude copies sharing a source patient from one another's external pools; and recompute patient medians",
        "same_original_copy_external_exclusion": True,
        "bootstrap_replicates_with_repeated_source_patient": int((same_original_copy_audit.n_source_patients_selected_more_than_once > 0).sum()),
        "bootstrap_proportion_with_repeated_source_patient": float((same_original_copy_audit.n_source_patients_selected_more_than_once > 0).mean()),
        "bootstrap_median_pre_exclusion_zero_distance_external_candidates": float(same_original_copy_audit.n_pre_exclusion_zero_distance_external_candidates.median()),
        "bootstrap_max_pre_exclusion_zero_distance_external_candidates": int(same_original_copy_audit.n_pre_exclusion_zero_distance_external_candidates.max()),
        "inferential_role": inferential_role,
    }
    distribution = pd.DataFrame({
        "bootstrap_replicate": np.arange(1, N_CLUSTER_BOOTSTRAP + 1),
        "bootstrap_statistic": statistics,
        "n_contributing_bootstrap_clusters": contributing_clusters,
    }).merge(same_original_copy_audit, on="bootstrap_replicate", how="left", validate="one_to_one")
    return summary, distribution


def primary_distance_scale_and_comparator_audit(
    sample_metrics: pd.DataFrame, patient_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report absolute distance scale and visit-specific external-comparator availability."""
    within = patient_metrics.median_within_distance.astype(float)
    between = patient_metrics.median_matched_between_distance.astype(float)
    gap = patient_metrics.between_minus_within_distance.astype(float)
    scale = pd.DataFrame([{
        "analysis": "primary_bray_same_or_adjacent_visit",
        "n_contributing_patients": int(len(patient_metrics)),
        "n_contributing_profiles": int(len(sample_metrics)),
        "patient_median_within_distance": float(within.median()),
        "patient_median_matched_between_distance": float(between.median()),
        "patient_median_between_minus_within_distance": float(gap.median()),
        "relative_gap_vs_patient_median_within_distance": float(gap.median() / within.median()),
        "between_to_within_distance_ratio": float(between.median() / within.median()),
    }])

    rows: list[dict] = []
    for visit, group in sample_metrics.groupby("visit", sort=True):
        external_profiles = group.n_matched_between_candidates.astype(float)
        external_patients = group.n_matched_between_patients.astype(float)
        rows.append({
            "visit": int(visit),
            "n_focal_profiles": int(len(group)),
            "n_focal_patients": int(group.patient_id.nunique()),
            "external_profiles_min": int(external_profiles.min()),
            "external_profiles_q25": float(external_profiles.quantile(0.25)),
            "external_profiles_median": float(external_profiles.median()),
            "external_profiles_q75": float(external_profiles.quantile(0.75)),
            "external_profiles_max": int(external_profiles.max()),
            "external_patients_min": int(external_patients.min()),
            "external_patients_q25": float(external_patients.quantile(0.25)),
            "external_patients_median": float(external_patients.median()),
            "external_patients_q75": float(external_patients.quantile(0.75)),
            "external_patients_max": int(external_patients.max()),
        })
    audit = pd.DataFrame(rows)
    return scale, audit


def grouped_pseudo_r2(gower: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    """Pseudo-R2 for a categorical grouping without repeatedly fitting a dense design matrix."""
    codes, unique_labels = pd.factorize(labels, sort=False)
    total_ss = float(np.trace(gower))
    explained_ss = 0.0
    for code in range(len(unique_labels)):
        indices = np.flatnonzero(codes == code)
        explained_ss += float(gower[np.ix_(indices, indices)].sum()) / len(indices)
    return max(explained_ss / total_ss, 0.0), int(len(unique_labels))


def patient_identity_resolution_adjustment(distance_matrix: pd.DataFrame, metadata: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Quantify the degree-of-freedom and visit-structure contribution to patient pseudo-R2."""
    meta = metadata.loc[distance_matrix.index]
    if meta.duplicated(["patient_id", "visit"]).any():
        raise ValueError("Within-visit label permutation requires at most one profile per patient and visit.")
    gower = gower_center(distance_matrix.to_numpy(float))
    labels = meta.patient_id.to_numpy(int)
    visits = meta.visit.to_numpy(int)
    observed_r2, factor_levels = grouped_pseudo_r2(gower, labels)
    n_samples = len(labels)
    factor_df = factor_levels - 1
    adjusted_r2 = 1 - (1 - observed_r2) * (n_samples - 1) / (n_samples - factor_df - 1)
    rng = np.random.default_rng(SEED + 202)
    null = np.empty(N_DESIGN_PERMUTATIONS, dtype=float)
    for index in range(N_DESIGN_PERMUTATIONS):
        null[index], null_levels = grouped_pseudo_r2(gower, stage_preserving_patient_labels(labels, visits, rng))
        if null_levels != factor_levels:
            raise RuntimeError("Stage-preserving permutation changed the patient-factor degrees of freedom.")
    summary = {
        "analysis": "patient_identity_pseudo_r2_stage_preserving_null",
        "n_samples": int(n_samples),
        "patient_factor_levels": int(factor_levels),
        "patient_factor_df": int(factor_df),
        "observed_raw_pseudo_r2": observed_r2,
        "observed_df_adjusted_pseudo_r2": float(adjusted_r2),
        "permuted_null_median_raw_pseudo_r2": float(np.median(null)),
        "permuted_null_q025_raw_pseudo_r2": float(np.quantile(null, 0.025)),
        "permuted_null_q975_raw_pseudo_r2": float(np.quantile(null, 0.975)),
        "raw_pseudo_r2_excess_over_null_median": float(observed_r2 - np.median(null)),
        "upper_tail_randomization_p": float((1 + np.sum(null >= observed_r2)) / (1 + len(null))),
        "n_design_permutations": int(N_DESIGN_PERMUTATIONS),
        "permutation_scheme": "patient labels independently permuted within visit ordinal; each visit retains its observed patient-label set and sample count",
    }
    return summary, pd.DataFrame({"permutation_index": np.arange(1, N_DESIGN_PERMUTATIONS + 1), "null_raw_pseudo_r2": null})


def gower_center(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    centre = np.eye(n) - np.ones((n, n)) / n
    return centre @ (-0.5 * distance_matrix ** 2) @ centre


def design_matrix(metadata: pd.DataFrame, terms: list[str]) -> np.ndarray:
    columns = [np.ones(len(metadata))]
    if "patient_identity" in terms:
        columns.append(pd.get_dummies(metadata.patient_id.astype(str), drop_first=True, dtype=float).to_numpy(float))
    if "visit_order" in terms:
        value = metadata.visit.astype(float).to_numpy()
        columns.append(((value - value.mean()) / value.std(ddof=0)).reshape(-1, 1))
    if "antibiotic_exposed" in terms:
        columns.append(metadata.antibiotic_exposed.astype(float).to_numpy().reshape(-1, 1))
    if "depth_cm" in terms:
        value = metadata.depth_cm.astype(float).to_numpy()
        columns.append(((value - value.mean()) / value.std(ddof=0)).reshape(-1, 1))
    return np.column_stack(columns)


def fitted_ss(gower: np.ndarray, x: np.ndarray) -> tuple[float, int]:
    projection = x @ np.linalg.pinv(x)
    return max(float(np.trace(projection @ gower)), 0.0), int(np.linalg.matrix_rank(x))


def pseudo_f(ss_term: float, df_term: int, ss_residual: float, df_residual: int) -> float:
    if df_term <= 0 or df_residual <= 0 or ss_residual <= 0:
        return np.nan
    return float((ss_term / df_term) / (ss_residual / df_residual))


def variance_decomposition(distance_matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Distance-based pseudo-R2; covariate P values are restricted-label sensitivities."""
    terms = ["patient_identity", "visit_order", "antibiotic_exposed", "depth_cm"]
    meta = metadata.loc[distance_matrix.index].copy()
    gower = gower_center(distance_matrix.to_numpy(float))
    total_ss = float(np.trace(gower))
    x_full = design_matrix(meta, terms)
    ss_full, rank_full = fitted_ss(gower, x_full)
    residual_ss = max(total_ss - ss_full, 0.0)
    df_residual = len(meta) - rank_full
    rows = []
    preceding: list[str] = []
    for term in terms:
        x_single = design_matrix(meta, [term])
        ss_single, rank_single = fitted_ss(gower, x_single)
        x_before = design_matrix(meta, preceding)
        ss_before, _ = fitted_ss(gower, x_before)
        x_after = design_matrix(meta, preceding + [term])
        ss_after, _ = fitted_ss(gower, x_after)
        reduced_terms = [item for item in terms if item != term]
        x_reduced = design_matrix(meta, reduced_terms)
        ss_reduced, rank_reduced = fitted_ss(gower, x_reduced)
        unique_ss = max(ss_full - ss_reduced, 0.0)
        unique_df = rank_full - rank_reduced
        observed = pseudo_f(unique_ss, unique_df, residual_ss, df_residual)
        row = {
            "term": term, "n_samples": int(len(meta)), "df_term_unique": int(unique_df), "df_residual_full": int(df_residual),
            "r2_standalone": float(ss_single / total_ss),
            "r2_sequential_patient_first": float(max(ss_after - ss_before, 0.0) / total_ss),
            "r2_unique_joint_model": float(unique_ss / total_ss),
            "pseudo_f_unique": observed, "permutation_p": np.nan, "permutation_scheme": "not_run",
            "full_model_r2": float(ss_full / total_ss),
        }
        null_f = []
        if term == "patient_identity":
            observed_for_p = pseudo_f(ss_single, rank_single - 1, max(total_ss - ss_single, 0.0), len(meta) - rank_single)
            for _ in range(N_COVARIATE_PERMUTATIONS):
                permuted = meta.copy()
                permuted["patient_id"] = RNG.permutation(permuted.patient_id.to_numpy())
                ss_perm, rank_perm = fitted_ss(gower, design_matrix(permuted, ["patient_identity"]))
                null_f.append(pseudo_f(ss_perm, rank_perm - 1, max(total_ss - ss_perm, 0.0), len(meta) - rank_perm))
            row["permutation_scheme"] = "unrestricted_patient-label_clustering_sensitivity"
        else:
            observed_for_p = observed
            metadata_column = {"visit_order": "visit", "antibiotic_exposed": "antibiotic_exposed", "depth_cm": "depth_cm"}[term]
            for _ in range(N_COVARIATE_PERMUTATIONS):
                permuted = meta.copy()
                values = []
                for _, sub in permuted.groupby("patient_id", sort=False):
                    values.extend(RNG.permutation(sub[metadata_column].to_numpy()))
                permuted[metadata_column] = values
                ss_perm_full, _ = fitted_ss(gower, design_matrix(permuted, terms))
                ss_perm_term = max(ss_perm_full - ss_reduced, 0.0)
                null_f.append(pseudo_f(ss_perm_term, unique_df, residual_ss, df_residual))
            row["permutation_scheme"] = "within-patient_covariate-label_sensitivity"
        null = np.asarray([value for value in null_f if np.isfinite(value)])
        if len(null) and np.isfinite(observed_for_p):
            row["permutation_p"] = float((1 + np.sum(null >= observed_for_p)) / (1 + len(null)))
        rows.append(row)
        preceding.append(term)
    rows.append({
        "term": "joint_model", "n_samples": int(len(meta)), "df_term_unique": rank_full - 1, "df_residual_full": df_residual,
        "r2_standalone": float(ss_full / total_ss), "r2_sequential_patient_first": float(ss_full / total_ss),
        "r2_unique_joint_model": float(ss_full / total_ss), "pseudo_f_unique": pseudo_f(ss_full, rank_full - 1, residual_ss, df_residual),
        "permutation_p": np.nan, "permutation_scheme": "not_applicable", "full_model_r2": float(ss_full / total_ss),
    })
    return pd.DataFrame(rows)


def visit0_to_1_transition(distance_matrix: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Visit 0/1 is called pre/post debridement only by the supplied author R code."""
    meta = metadata.loc[metadata.visit.isin([0, 1])].copy()
    rows = []
    for patient, sub in meta.groupby("patient_id", sort=True):
        if set(sub.visit) != {0, 1} or sub.visit.nunique() != 2:
            continue
        baseline = sub.index[sub.visit == 0][0]
        post = sub.index[sub.visit == 1][0]
        own = float(distance_matrix.loc[baseline, post])
        other_0 = meta.index[(meta.patient_id != patient) & (meta.visit == 0)]
        other_1 = meta.index[(meta.patient_id != patient) & (meta.visit == 1)]
        cross = np.concatenate([distance_matrix.loc[baseline, other_1].to_numpy(float), distance_matrix.loc[post, other_0].to_numpy(float)])
        if not len(cross):
            continue
        rows.append({
            "patient_id": int(patient), "visit0_to_visit1_within_distance": own,
            "matched_cross_patient_visit0_to_visit1_distance": float(np.median(cross)),
            "between_minus_within_distance": float(np.median(cross) - own), "n_samples_contributing": 2,
        })
    frame = pd.DataFrame(rows)
    return frame, summarize_patient_gaps(frame, "author_code_defined_visit0_to_visit1_pre_post_debridement_interval")


def antibiotic_pair_summary(distance_matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient, sub in metadata.groupby("patient_id", sort=True):
        for first, second in itertools.combinations(sub.index.tolist(), 2):
            first_state = bool(sub.loc[first, "antibiotic_exposed"])
            second_state = bool(sub.loc[second, "antibiotic_exposed"])
            state = "both_exposed" if first_state and second_state else "both_unexposed" if not first_state and not second_state else "discordant"
            rows.append({"patient_id": int(patient), "pair_antibiotic_state": state, "functional_distance": float(distance_matrix.loc[first, second])})
    pairs = pd.DataFrame(rows)
    return pairs.groupby(["patient_id", "pair_antibiotic_state"], as_index=False).agg(
        n_pairs=("functional_distance", "size"), median_within_patient_functional_distance=("functional_distance", "median")
    )


# --- Input construction ---------------------------------------------------------------------
meta = read_csv("metamap.csv")
meta["id"] = meta["id"].astype(str)
meta["patient_id"] = pd.to_numeric(meta["patient_id"], errors="coerce")
meta["visit"] = pd.to_numeric(meta["visit"], errors="coerce")
meta["depth_cm"] = pd.to_numeric(meta["depth"], errors="coerce")
meta = meta.dropna(subset=["id", "patient_id", "visit", "depth_cm"]).copy()
meta["patient_id"] = meta.patient_id.astype(int)
meta["visit"] = meta.visit.astype(int)
meta = meta.drop_duplicates("id", keep="first").set_index("id", drop=False)
meta["antibiotic_exposed"] = boolean(meta["antibiotics"]) | boolean(meta["v.abx"])

function_long = read_csv("SEED.sub1.map.csv", usecols=["id", "Subsystem.Level.1", "Relative.Abundance"])
function_long["id"] = function_long.id.astype(str)
function_long["Subsystem.Level.1"] = function_long["Subsystem.Level.1"].astype(str).str.strip()
function_long["Relative.Abundance"] = pd.to_numeric(function_long["Relative.Abundance"], errors="coerce").fillna(0.0)
function_raw = function_long.pivot_table(index="id", columns="Subsystem.Level.1", values="Relative.Abundance", aggfunc="sum", fill_value=0.0)
common = function_raw.index.intersection(meta.index)
function = normalise(function_raw.loc[common])
metadata = meta.loc[function.index].sort_values(["patient_id", "visit"]).copy()
function = function.loc[metadata.index]
function_bray = pd.DataFrame(cdist(function, function, metric="braycurtis"), index=function.index, columns=function.index)
aitchison_pseudocount = clr_pseudocount(function)
function_clr = clr(function, aitchison_pseudocount)
function_aitchison = pd.DataFrame(cdist(function_clr, function_clr, metric="euclidean"), index=function.index, columns=function.index)

# A deeper SEED asset is present, but it is a nonrepresentative seven-patient subset
# and therefore cannot validate the 195-sample primary cohort at another resolution.
level3_ids = read_csv("SEED.sub3.map.csv", usecols=["id"])["id"].astype(str).drop_duplicates()
level3_metadata = metadata.loc[metadata.index.intersection(level3_ids)]

audit = pd.DataFrame([{
    "matched_samples": int(len(metadata)), "patients": int(metadata.patient_id.nunique()),
    "patients_at_least_2_visits": int((metadata.groupby("patient_id").size() >= 2).sum()),
    "patients_at_least_3_visits": int((metadata.groupby("patient_id").size() >= 3).sum()),
    "functional_domains": int(function.shape[1]), "functional_resolution_primary": "SEED Level 1 (broad domains)",
    "aitchison_zero_handling": "fixed additive pseudocount before CLR transformation",
    "aitchison_pseudocount": float(aitchison_pseudocount),
    "zero_abundance_cells": int((function.to_numpy(float) == 0).sum()),
    "total_functional_abundance_cells": int(function.size),
    "antibiotic_exposed_samples": int(metadata.antibiotic_exposed.sum()),
    "antibiotic_unexposed_samples": int((~metadata.antibiotic_exposed).sum()),
    "patients_with_within_patient_antibiotic_state_variation": int(sum(sub.antibiotic_exposed.nunique() > 1 for _, sub in metadata.groupby("patient_id"))),
    "visit0_visit1_pairs": int(sum(set(sub.visit) >= {0, 1} for _, sub in metadata.groupby("patient_id"))),
    "debridement_variable_available": False,
    "debridement_proxy": "Original author R code labels visit 0/1 as pre/post-debridement; no individual debridement field exists.",
    "seed_level2_asset_available": False,
    "seed_level3_asset_available": True,
    "seed_level3_samples": int(len(level3_metadata)),
    "seed_level3_patients": int(level3_metadata.patient_id.nunique()),
    "seed_level3_usable_for_primary_resolution_validation": False,
    "seed_level3_resolution_limit": "Only 27 profiles from 7 patients; insufficient and nonrepresentative relative to the 195-profile, 46-patient primary cohort.",
}])
audit.to_csv(RESULTS / "14_FORMAL_ANALYSIS_AUDIT.csv", index=False)


# --- Layer 1: strict matched individuality --------------------------------------------------
matched_definitions = [
    ("primary_bray_same_or_adjacent_visit", function_bray, dict(max_visit_delta=1)),
    ("bray_exact_same_visit", function_bray, dict(max_visit_delta=0)),
    ("aitchison_same_or_adjacent_visit", function_aitchison, dict(max_visit_delta=1)),
    ("bray_same_or_adjacent_visit_patients_at_least_3_visits", function_bray, dict(max_visit_delta=1, eligible_mask=metadata.patient_id.isin(metadata.groupby("patient_id").size().loc[lambda values: values >= 3].index))),
    ("bray_same_or_adjacent_visit_same_antibiotic_state", function_bray, dict(max_visit_delta=1, same_antibiotic_state=True)),
    ("bray_same_or_adjacent_visit_depth_within_0.2cm", function_bray, dict(max_visit_delta=1, depth_tolerance_cm=0.2)),
    ("bray_same_or_adjacent_visit_antibiotic_unexposed_only", function_bray, dict(max_visit_delta=1, eligible_mask=~metadata.antibiotic_exposed)),
]
matched_summaries = []
matched_artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
for label, matrix, kwargs in matched_definitions:
    sample_metrics, patient_metrics, summary = matched_individuality(matrix, metadata, label, **kwargs)
    sample_metrics.to_csv(DERIVED / f"15_sample_matched_individuality_{label}.csv", index=False)
    patient_metrics.to_csv(DERIVED / f"15_patient_matched_individuality_{label}.csv", index=False)
    matched_artifacts[label] = (sample_metrics, patient_metrics)
    matched_summaries.append(summary)

primary_sample_metrics, primary_patient_metrics = matched_artifacts["primary_bray_same_or_adjacent_visit"]
primary_cluster_bootstrap, primary_cluster_bootstrap_distribution = primary_full_recomputation_cluster_bootstrap(function_bray, metadata)
three_visit_sample_metrics, three_visit_patient_metrics = matched_artifacts["bray_same_or_adjacent_visit_patients_at_least_3_visits"]
three_visit_ids = metadata.index[metadata.patient_id.isin(
    metadata.groupby("patient_id").size().loc[lambda values: values >= 3].index
)]
three_visit_cluster_bootstrap, three_visit_cluster_bootstrap_distribution = primary_full_recomputation_cluster_bootstrap(
    function_bray.loc[three_visit_ids, three_visit_ids], metadata.loc[three_visit_ids], seed_offset=223,
    analysis_label="bray_same_or_adjacent_visit_patients_at_least_3_visits_full_recomputation_cluster_bootstrap",
    inferential_role="supporting restriction uncertainty interval; it is not a second primary null-hypothesis test",
)
for summary in matched_summaries:
    if summary["analysis"] == "primary_bray_same_or_adjacent_visit":
        summary["ci95_low"] = primary_cluster_bootstrap["ci95_low"]
        summary["ci95_high"] = primary_cluster_bootstrap["ci95_high"]
        summary["ci_method"] = "full_recomputation_cluster_bootstrap"
        summary["n_cluster_bootstrap_replicates"] = primary_cluster_bootstrap["n_cluster_bootstrap_replicates"]
    if summary["analysis"] == "bray_same_or_adjacent_visit_patients_at_least_3_visits":
        summary["ci95_low"] = three_visit_cluster_bootstrap["ci95_low"]
        summary["ci95_high"] = three_visit_cluster_bootstrap["ci95_high"]
        summary["ci_method"] = "full_recomputation_cluster_bootstrap"
        summary["n_cluster_bootstrap_replicates"] = three_visit_cluster_bootstrap["n_cluster_bootstrap_replicates"]

distance_scale, comparator_audit = primary_distance_scale_and_comparator_audit(primary_sample_metrics, primary_patient_metrics)
matched_results = pd.DataFrame(matched_summaries)
matched_results.to_csv(RESULTS / "15_MATCHED_INDIVIDUALITY_RESULTS.csv", index=False)
pd.DataFrame([primary_cluster_bootstrap]).to_csv(RESULTS / "16_PRIMARY_FULL_RECOMPUTATION_CLUSTER_BOOTSTRAP.csv", index=False)
primary_cluster_bootstrap_distribution.to_csv(DERIVED / "16_primary_full_recomputation_cluster_bootstrap_distribution.csv", index=False)
primary_cluster_bootstrap_distribution.loc[:, [
    "bootstrap_replicate",
    "n_unique_source_patients",
    "n_source_patients_selected_more_than_once",
    "n_bootstrap_copies_from_repeated_source_patients",
    "n_profiles_from_repeated_source_patient_copies",
    "n_pre_exclusion_zero_distance_external_candidates",
]].to_csv(RESULTS / "16_CLUSTER_BOOTSTRAP_SAME_ORIGINAL_COPY_AUDIT.csv", index=False)
pd.DataFrame([three_visit_cluster_bootstrap]).to_csv(RESULTS / "16_PATIENTS_AT_LEAST_3_VISITS_FULL_RECOMPUTATION_CLUSTER_BOOTSTRAP.csv", index=False)
three_visit_cluster_bootstrap_distribution.to_csv(DERIVED / "16_patients_at_least_3_visits_full_recomputation_cluster_bootstrap_distribution.csv", index=False)
distance_scale.to_csv(RESULTS / "16_PRIMARY_DISTANCE_SCALE.csv", index=False)
comparator_audit.to_csv(RESULTS / "16_PRIMARY_COMPARATOR_POOL_AUDIT.csv", index=False)
lopo = lopo_matched(function_bray, metadata, "primary_bray_same_or_adjacent_visit", max_visit_delta=1)
lopo.to_csv(RESULTS / "16_MATCHED_INDIVIDUALITY_LOPO.csv", index=False)

# The design-preserving randomization is the primary inferential analysis. It
# recomputes every within-vs-between contrast after visit-stratified relabeling,
# thereby preserving comparator-pool dependence under the null.
primary_permutation, primary_permutation_null = primary_design_preserving_permutation(function_bray, metadata)
pd.DataFrame([primary_permutation]).to_csv(RESULTS / "16_PRIMARY_DESIGN_PRESERVING_PERMUTATION.csv", index=False)
primary_permutation_null.to_csv(DERIVED / "16_primary_design_preserving_permutation_null.csv", index=False)


# --- Layer 2: distance-based variance decomposition -----------------------------------------
variance = variance_decomposition(function_bray, metadata)
patient_r2_adjustment, patient_r2_null = patient_identity_resolution_adjustment(function_bray, metadata)
patient_row = variance.term == "patient_identity"
variance["r2_df_adjusted"] = np.nan
variance["r2_stage_permuted_null_median"] = np.nan
variance["r2_excess_over_stage_permuted_null_median"] = np.nan
variance["stage_permuted_upper_tail_p"] = np.nan
variance.loc[patient_row, "r2_df_adjusted"] = patient_r2_adjustment["observed_df_adjusted_pseudo_r2"]
variance.loc[patient_row, "r2_stage_permuted_null_median"] = patient_r2_adjustment["permuted_null_median_raw_pseudo_r2"]
variance.loc[patient_row, "r2_excess_over_stage_permuted_null_median"] = patient_r2_adjustment["raw_pseudo_r2_excess_over_null_median"]
variance.loc[patient_row, "stage_permuted_upper_tail_p"] = patient_r2_adjustment["upper_tail_randomization_p"]
variance.loc[patient_row, "permutation_p"] = patient_r2_adjustment["upper_tail_randomization_p"]
variance.loc[patient_row, "permutation_scheme"] = patient_r2_adjustment["permutation_scheme"]
variance.to_csv(RESULTS / "17_FUNCTIONAL_VARIANCE_DECOMPOSITION.csv", index=False)
pd.DataFrame([patient_r2_adjustment]).to_csv(RESULTS / "17_PATIENT_IDENTITY_RESOLUTION_ADJUSTMENT.csv", index=False)
patient_r2_null.to_csv(DERIVED / "17_patient_identity_stage_preserving_null.csv", index=False)


# --- Layer 3: antibiotics and author-code-defined visit 0/1 interval ------------------------
antibiotic_pairs = antibiotic_pair_summary(function_bray, metadata)
antibiotic_pairs.to_csv(DERIVED / "19_patient_antibiotic_pair_distance_summary.csv", index=False)
debridement_transition, debridement_summary = visit0_to_1_transition(function_bray, metadata)
debridement_transition.to_csv(DERIVED / "19_visit0_visit1_transition_metrics.csv", index=False)
keep_labels = {
    "primary_bray_same_or_adjacent_visit", "bray_same_or_adjacent_visit_same_antibiotic_state",
    "bray_same_or_adjacent_visit_antibiotic_unexposed_only",
    "bray_same_or_adjacent_visit_patients_at_least_3_visits",
}
perturbation = pd.DataFrame([*[row for row in matched_summaries if row["analysis"] in keep_labels], debridement_summary])
perturbation.to_csv(RESULTS / "19_PERTURBATION_ROBUSTNESS_RESULTS.csv", index=False)


# --- Figure 1: sampling architecture ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
patient_order = (
    metadata.groupby("patient_id", as_index=False)
    .agg(n_profiles=("id", "size"), maximum_visit_ordinal=("visit", "max"))
    .sort_values(["n_profiles", "maximum_visit_ordinal", "patient_id"], kind="stable")
    .reset_index(drop=True)
)
patient_rows = {int(patient): int(row + 1) for row, patient in enumerate(patient_order.patient_id)}
legend_seen = {"exposed": False, "unexposed": False}
for patient, sub in metadata.groupby("patient_id", sort=True):
    color = "#d95f02" if bool(sub.antibiotic_exposed.any()) else "#1b9e77"
    exposure_key = "exposed" if bool(sub.antibiotic_exposed.any()) else "unexposed"
    label = None
    if not legend_seen[exposure_key]:
        label = "Any supplied systemic-antibiotic exposure" if exposure_key == "exposed" else "No supplied systemic-antibiotic exposure"
        legend_seen[exposure_key] = True
    row = patient_rows[int(patient)]
    ax.plot(sub.visit, [row] * len(sub), color="#bdbdbd", linewidth=0.7, zorder=1)
    ax.scatter(sub.visit, [row] * len(sub), c=color, s=22, zorder=2, label=label)
ax.set(title="Longitudinal sampling architecture", xlabel="Visit ordinal", ylabel="Ordered patient row")
ax.set_yticks(np.arange(1, len(patient_order) + 1, 5))
ax.legend(title="Supplied antibiotic exposure", frameon=False, loc="lower right", fontsize=8, title_fontsize=8)
fig.savefig(FIGURES / "Figure_1_sampling_architecture.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# --- Figure 2: primary and exact-visit individuality -----------------------------------------
plot_defs = ["primary_bray_same_or_adjacent_visit", "bray_exact_same_visit", "aitchison_same_or_adjacent_visit"]
plot = matched_results.set_index("analysis").loc[plot_defs].reset_index()
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
for i, (_, row) in enumerate(plot.iloc[:2].iterrows()):
    points = pd.read_csv(DERIVED / f"15_patient_matched_individuality_{row.analysis}.csv")["between_minus_within_distance"].dropna().to_numpy()
    axes[0].scatter(np.full(len(points), i), points, color=["#2a9d8f", "#457b9d"][i], alpha=0.65, s=26, zorder=2)
    axes[0].errorbar(i, row.median_between_minus_within_distance,
                     yerr=[[row.median_between_minus_within_distance - row.ci95_low], [row.ci95_high - row.median_between_minus_within_distance]],
                     fmt="D", color="#111111", capsize=4, zorder=3)
axes[0].axhline(0, color="#555555", linestyle="--", linewidth=1)
axes[0].set_xticks([0, 1], ["Bray\nsame/adjacent visit", "Bray\nsame visit"])
axes[0].set_ylabel("Patient median matched-between minus within distance")
axes[0].set_title("Bray–Curtis primary and exact-visit sensitivity")
row = plot.iloc[2]
points = pd.read_csv(DERIVED / f"15_patient_matched_individuality_{row.analysis}.csv")["between_minus_within_distance"].dropna().to_numpy()
axes[1].scatter(np.zeros(len(points)), points, color="#6a4c93", alpha=0.65, s=30, zorder=2)
axes[1].errorbar(0, row.median_between_minus_within_distance,
                 yerr=[[row.median_between_minus_within_distance - row.ci95_low], [row.ci95_high - row.median_between_minus_within_distance]],
                 fmt="D", color="#111111", capsize=4, zorder=3)
axes[1].axhline(0, color="#555555", linestyle="--", linewidth=1)
axes[1].set_xticks([0], ["Aitchison\nsame/adjacent visit"])
axes[1].set_ylabel("Patient median matched-between minus within distance")
axes[1].set_title("Compositional-distance sensitivity")
fig.suptitle("Patient-specific functional potential after visit-stage matching", fontsize=16)
fig.savefig(FIGURES / "Figure_2_matched_functional_individuality.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# --- Figure 3: variance decomposition --------------------------------------------------------
terms_plot = variance.loc[variance.term.isin(["patient_identity", "visit_order", "antibiotic_exposed", "depth_cm"])].copy()
labels = {"patient_identity": "Patient identity", "visit_order": "Visit order", "antibiotic_exposed": "Antibiotic (patient-fixed)", "depth_cm": "Wound depth"}
terms_plot["label"] = terms_plot.term.map(labels)
fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
positions = np.arange(len(terms_plot))
ax.barh(positions + 0.18, terms_plot.r2_sequential_patient_first * 100, height=0.35, label="Sequential R² (patient first)", color="#2a9d8f")
ax.barh(positions - 0.18, terms_plot.r2_unique_joint_model * 100, height=0.35, label="Unique joint-model R²", color="#8d99ae")
ax.set_yticks(positions, terms_plot.label)
ax.set_xlabel("Explained functional-distance variation (%)")
ax.set_title("Distance-based functional variation decomposition")
ax.legend(frameon=False)
antibiotic_row = terms_plot.loc[terms_plot.term == "antibiotic_exposed"].iloc[0]
if antibiotic_row.df_term_unique == 0:
    ax.text(0.35, positions[terms_plot.index.get_loc(antibiotic_row.name)], "Not uniquely estimable after patient identity", va="center", fontsize=9, color="#555555")
fig.savefig(FIGURES / "Figure_3_functional_variance_decomposition.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# --- Revised Figure 3: patient-identity resolution correction -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
axes[0].hist(patient_r2_null.null_raw_pseudo_r2 * 100, bins=38, color="#b8d8d8", edgecolor="white")
axes[0].axvline(patient_r2_adjustment["observed_raw_pseudo_r2"] * 100, color="#d1495b", linewidth=2.2, label="Observed raw pseudo-R²")
axes[0].axvline(patient_r2_adjustment["permuted_null_median_raw_pseudo_r2"] * 100, color="#555555", linestyle="--", linewidth=1.4, label="Stage-preserving null median")
axes[0].set(title="Patient-label signal beyond visit-stage structure", xlabel="Raw patient-identity pseudo-R² (%)", ylabel="Permutations")
axes[0].legend(frameon=False, fontsize=8)
axes[0].text(0.02, 0.97,
             f"Degree-of-freedom–adjusted pseudo-R² = {patient_r2_adjustment['observed_df_adjusted_pseudo_r2'] * 100:.1f}%\n"
             f"Excess over null median = {patient_r2_adjustment['raw_pseudo_r2_excess_over_null_median'] * 100:.1f}%\n"
             f"Upper-tail P = {patient_r2_adjustment['upper_tail_randomization_p']:.4f}",
             transform=axes[0].transAxes, va="top", fontsize=9)
components = variance.loc[variance.term.isin(["visit_order", "depth_cm"])].copy()
component_labels = {"visit_order": "Visit order", "depth_cm": "Wound depth"}
axes[1].barh([component_labels[item] for item in components.term], components.r2_unique_joint_model * 100, color="#457b9d")
axes[1].set(title="Conditional variation components", xlabel="Unique joint-model pseudo-R² (%)")
fig.savefig(FIGURES / "Figure_3_patient_identity_resolution_correction.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# --- Figure 4: perturbation robustness -------------------------------------------------------
label_map = {
    "primary_bray_same_or_adjacent_visit": "Primary\nstage-matched",
    "bray_same_or_adjacent_visit_patients_at_least_3_visits": "Patients with\n≥3 visits",
    "bray_same_or_adjacent_visit_same_antibiotic_state": "Match antibiotic\nstate",
    "bray_same_or_adjacent_visit_antibiotic_unexposed_only": "Exclude exposed\nsamples",
    "author_code_defined_visit0_to_visit1_pre_post_debridement_interval": "Visit 0→1\nprotocol interval",
}
pert_plot = perturbation.copy()
pert_plot["label"] = pert_plot.analysis.map(label_map)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
for i, (_, row) in enumerate(pert_plot.iterrows()):
    axes[0].errorbar(i, row.median_between_minus_within_distance,
                     yerr=[[row.median_between_minus_within_distance - row.ci95_low], [row.ci95_high - row.median_between_minus_within_distance]],
                     fmt="o", color="#2a9d8f", capsize=4)
axes[0].axhline(0, color="#555555", linestyle="--", linewidth=1)
axes[0].set_xticks(range(len(pert_plot)), pert_plot.label)
axes[0].set_ylabel("Patient median matched-between minus within distance")
axes[0].set_title("Patient-specific structure under named sensitivities")
axes[1].bar(comparator_audit.visit, comparator_audit.external_patients_median, color="#457b9d", width=0.72)
axes[1].set(title="External comparator availability by visit stage", xlabel="Visit ordinal", ylabel="Distinct external patients in matched pool")
axes[1].set_xticks(comparator_audit.visit)
axes[1].text(0.02, 0.96, "Each contributing profile retained 5–45 external patients.\nLate visit stages had smaller comparison pools.", transform=axes[1].transAxes, va="top", fontsize=8.5, color="#555555")
fig.savefig(FIGURES / "Figure_4_sensitivities_and_comparator_availability.png", dpi=300, bbox_inches="tight")
plt.close(fig)


summary = {
    "analysis_name": "patient_specific_longitudinal_structure_of_microbial_functional_potential",
    "primary_matched_individuality": matched_summaries[0],
    "primary_design_preserving_permutation": primary_permutation,
    "primary_full_recomputation_cluster_bootstrap": primary_cluster_bootstrap,
    "patients_at_least_3_visits_full_recomputation_cluster_bootstrap": three_visit_cluster_bootstrap,
    "primary_distance_scale": distance_scale.iloc[0].to_dict(),
    "primary_comparator_pool_audit": comparator_audit.to_dict(orient="records"),
    "patient_identity_resolution_adjustment": patient_r2_adjustment,
    "primary_lopo_direction_positive_all_omissions": bool((lopo.median_between_minus_within_distance > 0).all()),
    "primary_lopo_effect_min": float(lopo.median_between_minus_within_distance.min()),
    "primary_lopo_effect_max": float(lopo.median_between_minus_within_distance.max()),
    "debridement_data_limit": "No individual debridement variable in supplied profiles; visit 0/1 robustness follows author-supplied figure code only.",
    "interpretation_limit": "Shotgun DNA profiles represent functional potential, not expression or activity.",
    "functional_resolution_limit": "SEED Level 3 is available for only 27 samples from 7 patients and cannot validate the 195-sample, 46-patient primary cohort at a finer resolution.",
}
(RESULTS / "20_FORMAL_ANALYSIS_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("Formal functional-stability analysis completed.")
