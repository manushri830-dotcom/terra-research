"""Stage 3: EXPLORATORY comparison of persistent candidates with FIRMS 'type 2' detections.

FIRMS type 2 = "other static land source". This is NOT a verified industrial label:
  * it is assigned by the FIRMS processing chain, so it is not independent ground truth;
  * a static-source mask is plausibly itself derived from recurrent detections, so some
    agreement with a persistence analysis is expected by construction (circularity);
  * 'static land source' is broader than industry and can include other fixed heat sources.
Therefore results here are descriptive overlap statistics, never accuracy/precision/recall of
industrial-source detection.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import fisher_exact

LIMITATIONS = [
    "FIRMS type 2 is a product flag for 'other static land source', not confirmed industrial heat.",
    "The flag is produced by the FIRMS algorithm and is not independent of persistence, "
    "so overlap may be partly circular.",
    "Persistent cells necessarily have many detections, which gives them more chances to carry "
    "a type-2 flag; the exposure-matched comparison reduces but does not remove this.",
    "Results describe one 90-day winter-season window (Jan-Mar 2026) and one satellite (NOAA-20).",
    "No external ground truth (e.g. a verified facility inventory) is used.",
]


def _table(a: pd.Series, b: pd.Series) -> dict:
    """2x2 table of boolean series a (rows) x b (cols) with Fisher exact test."""
    tp = int((a & b).sum()); tn = int((a & ~b).sum())
    fp = int((~a & b).sum()); fn = int((~a & ~b).sum())
    odds, p = fisher_exact([[tp, tn], [fp, fn]]) if (tp + tn) and (fp + fn) else (float("nan"), float("nan"))
    return {
        "group_a_with_type2": tp, "group_a_without_type2": tn,
        "group_b_with_type2": fp, "group_b_without_type2": fn,
        "share_a_with_type2": tp / (tp + tn) if (tp + tn) else None,
        "share_b_with_type2": fp / (fp + fn) if (fp + fn) else None,
        "fisher_odds_ratio": float(odds), "fisher_p_value": float(p),
    }


def type2_overlap(cell_stats: pd.DataFrame, clusters: pd.DataFrame, min_active_days: int) -> dict:
    s = cell_stats
    persistent = s["active_days"] >= min_active_days
    has_t2 = s["has_type2"]

    result: dict = {"min_active_days": int(min_active_days), "limitations": LIMITATIONS}

    # (a) persistent cells vs all other occupied cells
    result["persistent_vs_all_other_cells"] = _table(persistent, has_t2)

    # (b) exposure-matched: other cells that also had >= min_active_days detections
    matched_pool = persistent | (s["observation_count"] >= min_active_days)
    sub = s[matched_pool]
    result["persistent_vs_exposure_matched_cells"] = _table(sub["active_days"] >= min_active_days,
                                                            sub["has_type2"])
    result["persistent_vs_exposure_matched_cells"]["n_matched_non_persistent_cells"] = \
        int((sub["active_days"] < min_active_days).sum())

    # (c) detection-level share of type-2 detections
    t2_in = int(s.loc[persistent, "type2_count"].sum()); n_in = int(s.loc[persistent, "observation_count"].sum())
    t2_out = int(s.loc[~persistent, "type2_count"].sum()); n_out = int(s.loc[~persistent, "observation_count"].sum())
    result["detection_level"] = {
        "type2_share_in_persistent_cells": t2_in / n_in if n_in else None,
        "type2_share_in_other_cells": t2_out / n_out if n_out else None,
        "type2_detections_total": t2_in + t2_out,
        "type2_detections_in_persistent_cells": t2_in,
    }

    # (d) coverage of type-2 cells by persistent cells
    n_t2_cells = int(has_t2.sum())
    result["type2_cell_coverage"] = {
        "cells_with_type2": n_t2_cells,
        "of_which_persistent": int((has_t2 & persistent).sum()),
        "share_persistent": float((has_t2 & persistent).sum() / n_t2_cells) if n_t2_cells else None,
    }

    # (e) cluster-level categories
    if len(clusters):
        frac = clusters["type2_fraction"]
        result["cluster_level"] = {
            "n_clusters": int(len(clusters)),
            "no_type2_detections": int((clusters["type2_count"] == 0).sum()),
            "some_type2_detections": int(((clusters["type2_count"] > 0) & (frac < 0.5)).sum()),
            "majority_type2_detections": int((frac >= 0.5).sum()),
        }
    else:
        result["cluster_level"] = {"n_clusters": 0}
    return result
