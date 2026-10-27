"""Stage 2: grid-based persistence analysis.

A 'persistent thermal candidate' is a location where VIIRS detected a thermal anomaly on at
least `min_active_days` distinct UTC dates. This says the heat is *recurrent*. It does NOT say
what causes it (industry, crop burning, forest fire, flare, ...).

Known limitations (documented, not hidden):
  * A 0.01 deg cell is ~1.1 km tall but narrower in longitude, so cell area varies with latitude.
  * VIIRS pixel geolocation shifts between overpasses, so one physical source can straddle two
    cells. Adjacent qualifying cells are therefore merged into clusters, and cluster active days
    are the UNION of dates (never a sum).
  * A source split across cells where each cell individually has < min_active_days is missed.
  * Seasonal burning (Jan-Mar) can also create recurrent detections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import PersistenceParams

_EPS = 1e-9  # guards against float error, e.g. 0.29/0.01 = 28.999999999999996


def assign_cells(df: pd.DataFrame, grid_deg: float) -> pd.DataFrame:
    out = df.copy()
    out["cell_i"] = np.floor(out["latitude"].to_numpy() / grid_deg + _EPS).astype("int64")
    out["cell_j"] = np.floor(out["longitude"].to_numpy() / grid_deg + _EPS).astype("int64")
    return out


def cell_statistics(df_cells: pd.DataFrame, grid_deg: float) -> pd.DataFrame:
    d = df_cells.assign(
        _night=df_cells["daynight"].eq("N"),
        _high=df_cells["confidence"].eq("h"),
        _low=df_cells["confidence"].eq("l"),
        _t2=df_cells["type"].eq(2),
    )
    stats = (
        d.groupby(["cell_i", "cell_j"], sort=True)
        .agg(
            observation_count=("frp", "size"),
            active_days=("acq_date", "nunique"),
            first_detection=("acq_date", "min"),
            last_detection=("acq_date", "max"),
            mean_frp=("frp", "mean"),
            median_frp=("frp", "median"),
            max_frp=("frp", "max"),
            night_ratio=("_night", "mean"),
            high_conf_ratio=("_high", "mean"),
            low_conf_ratio=("_low", "mean"),
            type2_count=("_t2", "sum"),
            type2_fraction=("_t2", "mean"),
        )
        .reset_index()
    )
    stats["type2_count"] = stats["type2_count"].astype(int)
    stats["has_type2"] = stats["type2_count"] > 0
    stats["lat_center"] = (stats["cell_i"] + 0.5) * grid_deg
    stats["lon_center"] = (stats["cell_j"] + 0.5) * grid_deg
    return stats


def connected_cell_labels(cell_i, cell_j, connectivity: int = 8) -> np.ndarray:
    """Label groups of grid cells that touch (4- or 8-neighbourhood)."""
    n = len(cell_i)
    if n == 0:
        return np.array([], dtype=int)
    index = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(cell_i, cell_j))}
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    rows, cols = [], []
    for (a, b), k in index.items():
        for da, db in offsets:
            k2 = index.get((a + da, b + db))
            if k2 is not None:
                rows.append(k)
                cols.append(k2)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


@dataclass
class PersistenceResult:
    params: PersistenceParams
    cell_stats: pd.DataFrame        # every occupied cell
    persistent_cells: pd.DataFrame  # cells with active_days >= min_active_days (+ cluster_id)
    clusters: pd.DataFrame          # merged candidates, one row per physical-source candidate


def find_persistent(df: pd.DataFrame, params: PersistenceParams,
                    stats: pd.DataFrame | None = None,
                    df_cells: pd.DataFrame | None = None) -> PersistenceResult:
    params.validate()
    if df_cells is None:
        use = df[df["confidence"] != "l"] if params.exclude_low_confidence else df
        df_cells = assign_cells(use, params.grid_deg)
    if stats is None:
        stats = cell_statistics(df_cells, params.grid_deg)

    persistent = stats[stats["active_days"] >= params.min_active_days].copy()
    persistent["cluster_label"] = connected_cell_labels(
        persistent["cell_i"], persistent["cell_j"], params.connectivity)

    if persistent.empty:
        return PersistenceResult(params, stats, persistent.assign(cluster_id=pd.Series(dtype=str)),
                                 pd.DataFrame())

    det = df_cells.merge(persistent[["cell_i", "cell_j", "cluster_label"]],
                         on=["cell_i", "cell_j"], how="inner")
    det = det.assign(_night=det["daynight"].eq("N"), _high=det["confidence"].eq("h"),
                     _t2=det["type"].eq(2))
    clusters = (
        det.groupby("cluster_label")
        .agg(
            observation_count=("frp", "size"),
            active_days=("acq_date", "nunique"),     # union of dates across member cells
            first_detection=("acq_date", "min"),
            last_detection=("acq_date", "max"),
            mean_frp=("frp", "mean"),
            median_frp=("frp", "median"),
            max_frp=("frp", "max"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            night_ratio=("_night", "mean"),
            high_conf_ratio=("_high", "mean"),
            type2_count=("_t2", "sum"),
            type2_fraction=("_t2", "mean"),
        )
        .reset_index()
    )
    clusters["type2_count"] = clusters["type2_count"].astype(int)
    clusters["n_cells"] = clusters["cluster_label"].map(
        persistent.groupby("cluster_label").size()).astype(int)
    clusters = clusters.sort_values(
        ["active_days", "observation_count", "latitude", "longitude"],
        ascending=[False, False, True, True]).reset_index(drop=True)
    clusters.insert(0, "cluster_id", [f"TC-{i:05d}" for i in range(1, len(clusters) + 1)])
    id_map = dict(zip(clusters["cluster_label"], clusters["cluster_id"]))
    persistent["cluster_id"] = persistent["cluster_label"].map(id_map)
    return PersistenceResult(params, stats, persistent, clusters)


def legacy_cell_count(df: pd.DataFrame, grid_deg: float = 0.01, min_active_days: int = 10) -> int:
    """Re-implements the original script's float-based grid to check we reproduce its count."""
    lat = np.floor(df["latitude"] / grid_deg) * grid_deg
    lon = np.floor(df["longitude"] / grid_deg) * grid_deg
    days = df.assign(_la=lat, _lo=lon).groupby(["_la", "_lo"])["acq_date"].nunique()
    return int((days >= min_active_days).sum())


def sensitivity_table(df: pd.DataFrame,
                      grids=(0.005, 0.01, 0.02),
                      min_days=(5, 10, 15, 20),
                      exclude_low=(False, True),
                      connectivity: int = 8) -> pd.DataFrame:
    """How strongly do the candidate counts depend on analyst-chosen parameters?"""
    rows = []
    for ex in exclude_low:
        use = df[df["confidence"] != "l"] if ex else df
        for g in grids:
            df_cells = assign_cells(use, g)
            stats = cell_statistics(df_cells, g)
            for md in min_days:
                p = PersistenceParams(grid_deg=g, min_active_days=md,
                                      exclude_low_confidence=ex, connectivity=connectivity)
                r = find_persistent(use, p, stats=stats, df_cells=df_cells)
                n_cl = len(r.clusters)
                rows.append({
                    "grid_deg": g, "min_active_days": md, "exclude_low_confidence": ex,
                    "detections_used": int(len(use)), "occupied_cells": int(len(stats)),
                    "persistent_cells": int(len(r.persistent_cells)), "clusters": int(n_cl),
                    "clusters_with_any_type2": int((r.clusters["type2_count"] > 0).sum()) if n_cl else 0,
                })
    return pd.DataFrame(rows)
