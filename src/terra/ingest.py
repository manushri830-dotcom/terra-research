"""Stage 1: load and clean a NASA FIRMS VIIRS CSV export.

Nothing is dropped silently: every removal is counted in the returned report.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import REQUIRED_COLUMNS, VALID_CONFIDENCE, VALID_DAYNIGHT, VALID_TYPES

# acq_time must be read as text: "0721" read as an integer becomes 721 and loses the hour.
_STRING_COLS = {c: str for c in
                ("acq_time", "confidence", "daynight", "satellite", "instrument", "version")}


def load_firms(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FIRMS file not found: {path}")
    df = pd.read_csv(path, dtype=_STRING_COLS)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"FIRMS file is missing required columns: {missing}")
    return df


def clean_firms(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and normalise detections. Returns (clean_df, report)."""
    report: dict = {"rows_in": int(len(df)), "removed": {}}
    out = df.copy()

    def keep(mask: pd.Series, step: str) -> None:
        nonlocal out
        report["removed"][step] = int((~mask).sum())
        out = out[mask].copy()

    for col in ("latitude", "longitude", "frp", "brightness", "bright_t31", "scan", "track"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    keep(out[["latitude", "longitude", "frp"]].notna().all(axis=1), "non_numeric_lat_lon_frp")
    keep(out["latitude"].between(-90, 90) & out["longitude"].between(-180, 180),
         "coordinates_out_of_range")
    keep(out["frp"] >= 0, "negative_frp")

    out["acq_date"] = pd.to_datetime(out["acq_date"], format="%Y-%m-%d", errors="coerce")
    keep(out["acq_date"].notna(), "unparseable_acq_date")

    out["acq_time"] = out["acq_time"].astype(str).str.strip().str.zfill(4)
    hh = pd.to_numeric(out["acq_time"].str[:2], errors="coerce")
    mm = pd.to_numeric(out["acq_time"].str[2:], errors="coerce")
    keep(out["acq_time"].str.fullmatch(r"\d{4}") & (hh < 24) & (mm < 60), "invalid_acq_time")

    out["confidence"] = out["confidence"].astype(str).str.strip().str.lower()
    keep(out["confidence"].isin(VALID_CONFIDENCE), "invalid_confidence")
    out["daynight"] = out["daynight"].astype(str).str.strip().str.upper()
    keep(out["daynight"].isin(VALID_DAYNIGHT), "invalid_daynight")

    out["type"] = pd.to_numeric(out["type"], errors="coerce")
    keep(out["type"].isin(VALID_TYPES), "invalid_type")
    out["type"] = out["type"].astype(int)

    dup = out.duplicated(subset=REQUIRED_COLUMNS)
    keep(~dup, "exact_duplicates")

    hh = out["acq_time"].str[:2].astype(int)
    mm = out["acq_time"].str[2:].astype(int)
    out["acq_datetime_utc"] = out["acq_date"] + pd.to_timedelta(hh * 60 + mm, unit="m")
    out = out.reset_index(drop=True)

    report["rows_out"] = int(len(out))
    report["frp_zero_rows_kept"] = int((out["frp"] == 0).sum())
    if len(out):
        report["date_min"] = str(out["acq_date"].min().date())
        report["date_max"] = str(out["acq_date"].max().date())
        report["n_distinct_dates"] = int(out["acq_date"].nunique())
        report["bbox"] = {
            "lat_min": float(out["latitude"].min()), "lat_max": float(out["latitude"].max()),
            "lon_min": float(out["longitude"].min()), "lon_max": float(out["longitude"].max()),
        }
    report["type_counts"] = {int(k): int(v) for k, v in out["type"].value_counts().sort_index().items()}
    report["confidence_counts"] = {str(k): int(v) for k, v in out["confidence"].value_counts().items()}
    report["daynight_counts"] = {str(k): int(v) for k, v in out["daynight"].value_counts().items()}
    return out, report
