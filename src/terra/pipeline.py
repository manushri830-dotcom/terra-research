"""Run Stages 1-3 end to end:  python -m terra.pipeline --raw data/raw/<file>.csv

Outputs (all regenerated from the raw file; nothing is hand-edited):
  data/interim/firms_clean.csv.gz
  data/processed/{cell_statistics.csv.gz, persistent_cells.csv, persistent_clusters.csv}
  reports/{cleaning_report.json, sensitivity.csv, type2_overlap.json, run_manifest.json}
  reports/figures/*.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

from . import __version__
from .compare import type2_overlap
from .config import DEFAULT_RAW, PROJECT_ROOT, PersistenceParams
from .ingest import clean_firms, load_firms
from .persistence import find_persistent, legacy_cell_count, sensitivity_table
from .plots import make_figures


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def run(raw: Path, root: Path, params: PersistenceParams,
        sensitivity: bool = True, figures: bool = True) -> dict:
    interim, processed, reports = root / "data/interim", root / "data/processed", root / "reports"
    for d in (interim, processed, reports):
        d.mkdir(parents=True, exist_ok=True)

    df, clean_report = clean_firms(load_firms(raw))
    clean_report["legacy_script_reproduction_cells_ge_min_days"] = legacy_cell_count(
        df, params.grid_deg, params.min_active_days)
    df.to_csv(interim / "firms_clean.csv.gz", index=False)
    _dump(clean_report, reports / "cleaning_report.json")

    res = find_persistent(df, params)
    res.cell_stats.to_csv(processed / "cell_statistics.csv.gz", index=False)
    res.persistent_cells.to_csv(processed / "persistent_cells.csv", index=False)
    res.clusters.to_csv(processed / "persistent_clusters.csv", index=False)

    overlap = type2_overlap(res.cell_stats, res.clusters, params.min_active_days)
    _dump(overlap, reports / "type2_overlap.json")

    if sensitivity:
        sensitivity_table(df, connectivity=params.connectivity).to_csv(
            reports / "sensitivity.csv", index=False)
    if figures:
        make_figures(df, res.clusters, reports / "figures")

    manifest = {
        "terra_version": __version__,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file": str(raw.name), "input_sha256": _sha256(raw),
        "params": asdict(params),
        "counts": {"rows_clean": len(df), "occupied_cells": len(res.cell_stats),
                   "persistent_cells": len(res.persistent_cells), "clusters": len(res.clusters)},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    _dump(manifest, reports / "run_manifest.json")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TERRA stages 1-3")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--grid-deg", type=float, default=0.01)
    ap.add_argument("--min-active-days", type=int, default=10)
    ap.add_argument("--exclude-low-confidence", action="store_true")
    ap.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    params = PersistenceParams(a.grid_deg, a.min_active_days, a.exclude_low_confidence, a.connectivity)
    m = run(a.raw, a.root, params, not a.no_sensitivity, not a.no_figures)
    print(json.dumps(m["counts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
