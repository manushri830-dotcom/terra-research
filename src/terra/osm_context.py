"""Stage 4 (part 2): OpenStreetMap geographic-context enrichment of persistent clusters.

Reads  data/processed/persistent_clusters.csv   (Stage 3 output, never modified)
Writes data/cache/osm/*.json                    raw cached Overpass responses
       data/interim/osm_progress.jsonl          append-only per-cluster records (resume state)
       data/processed/osm_context.csv           one row per cluster
       reports/osm_failures.csv, reports/osm_enrichment_summary.json

IMPORTANT interpretation rules
  * Counts of OSM tags near a cluster are DESCRIPTIVE CONTEXT. No tag proves that a thermal
    anomaly is industrial. There is deliberately no 'is_industrial' column.
  * status 'ok_no_features' = the query succeeded and returned no matching elements. It does not
    mean nothing exists there: OSM coverage in India is uneven and tags are volunteer-contributed.
  * status 'failed' = the query did NOT succeed. These clusters have UNKNOWN context.
  * OSM is queried at run time (not Jan-Mar 2026), so land use may differ from the detection period.
  * The search is centred on the cluster centroid; large multi-cell clusters can extend beyond it.

Run:  python -m terra.osm_context --contact you@example.com --limit 3     (pilot)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from . import __version__
from .config import PROJECT_ROOT
from .osm_client import (DEFAULT_ENDPOINT, DiskCache, FetchResult, OverpassClient, RateLimiter,
                         RequestBudgetExceeded, query_hash, utc_now_iso)

QUERY_VERSION = "1"

NATURAL_VALUES = ("wood", "scrub", "grassland", "wetland", "water", "bare_rock", "heath")
INDUSTRIAL_MAN_MADE = ("works", "chimney", "storage_tank", "wastewater_plant",
                       "petroleum_well", "kiln", "silo")
POWER_VALUES = ("plant", "generator")
ROAD_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                "residential", "service", "track")

LIMITATIONS = [
    "OSM tags are descriptive context only; no tag confirms the source of a thermal anomaly.",
    "'ok_no_features' means the query succeeded with no matching elements, not that nothing exists; "
    "OSM completeness varies and is often lower in rural India.",
    "'failed' clusters have unknown context and must not be treated as 'no features'.",
    "OSM was queried at run time, not during the Jan-Mar 2026 detection window.",
    "The search is centred on the cluster centroid with a fixed radius; multi-cell clusters can "
    "extend beyond it.",
]

OUTPUT_COLUMNS = [
    "cluster_id", "latitude", "longitude", "n_cells", "radius_m", "status", "error_type",
    "error_detail", "http_attempts", "from_cache", "queried_utc", "query_version", "query_hash",
    "n_elements", "n_elements_without_location",
    "n_landuse_features", "landuse_industrial", "landuse_farmland", "landuse_forest",
    "landuse_residential", "landuse_counts_json",
    "n_natural_features", "natural_counts_json",
    "n_industrial_tagged_features", "industrial_tag_counts_json",
    "nearest_industrial_tagged_m", "nearest_industrial_tagged_tag",
    "n_road_features", "road_class_counts_json", "nearest_road_m", "nearest_road_class",
]
_SUMMARY_KEYS = OUTPUT_COLUMNS[OUTPUT_COLUMNS.index("n_elements"):]


# ----------------------------------------------------------------------------- query
def build_query(lat: float, lon: float, radius_m: int, timeout_s: int = 60,
                maxsize_bytes: int = 67_108_864) -> str:
    around = f"(around:{int(radius_m)},{lat:.6f},{lon:.6f})"
    nat = "|".join(NATURAL_VALUES)
    mm = "|".join(INDUSTRIAL_MAN_MADE)
    pw = "|".join(POWER_VALUES)
    rd = "|".join(ROAD_CLASSES)
    parts = [
        f'nwr["landuse"]{around};',
        f'nwr["natural"~"^({nat})$"]{around};',
        f'nwr["industrial"]{around};',
        f'nwr["man_made"~"^({mm})$"]{around};',
        f'nwr["power"~"^({pw})$"]{around};',
        f'way["highway"~"^({rd})(_link)?$"]{around};',
    ]
    body = "\n  ".join(parts)
    return (f"[out:json][timeout:{int(timeout_s)}][maxsize:{int(maxsize_bytes)}];\n"
            f"(\n  {body}\n);\nout tags center qt;")


# ------------------------------------------------------------------------- summarising
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def element_location(el: dict) -> Optional[tuple[float, float]]:
    if "lat" in el and "lon" in el:
        return float(el["lat"]), float(el["lon"])
    c = el.get("center")
    if isinstance(c, dict) and "lat" in c and "lon" in c:
        return float(c["lat"]), float(c["lon"])
    return None


def _json(counter: Counter) -> str:
    return json.dumps(dict(sorted(counter.items())), separators=(",", ":"))


def empty_summary() -> dict:
    return {k: None for k in _SUMMARY_KEYS}


def summarise_elements(elements: list, lat: float, lon: float) -> dict:
    landuse, natural, industrial, roads = Counter(), Counter(), Counter(), Counter()
    n_without_loc = 0
    near_ind: Optional[tuple[float, str]] = None
    near_road: Optional[tuple[float, str]] = None
    n_ind_features = 0
    road_set = set(ROAD_CLASSES)

    for el in elements:
        tags = el.get("tags") or {}
        loc = element_location(el)
        if loc is None:
            n_without_loc += 1
        dist = haversine_m(lat, lon, *loc) if loc else None

        if "landuse" in tags:
            landuse[tags["landuse"]] += 1
        if tags.get("natural") in NATURAL_VALUES:
            natural[tags["natural"]] += 1

        labels = []
        if "industrial" in tags:
            labels.append(f"industrial={tags['industrial']}")
        if tags.get("landuse") == "industrial":
            labels.append("landuse=industrial")
        if tags.get("man_made") in INDUSTRIAL_MAN_MADE:
            labels.append(f"man_made={tags['man_made']}")
        if tags.get("power") in POWER_VALUES:
            labels.append(f"power={tags['power']}")
        if labels:
            n_ind_features += 1
            industrial.update(labels)
            if dist is not None and (near_ind is None or dist < near_ind[0]):
                near_ind = (dist, sorted(labels)[0])

        hw = tags.get("highway")
        if hw is not None and hw.replace("_link", "") in road_set:
            roads[hw] += 1
            if dist is not None and (near_road is None or dist < near_road[0]):
                near_road = (dist, hw)

    return {
        "n_elements": len(elements),
        "n_elements_without_location": n_without_loc,
        "n_landuse_features": sum(landuse.values()),
        "landuse_industrial": landuse.get("industrial", 0),
        "landuse_farmland": landuse.get("farmland", 0),
        "landuse_forest": landuse.get("forest", 0),
        "landuse_residential": landuse.get("residential", 0),
        "landuse_counts_json": _json(landuse),
        "n_natural_features": sum(natural.values()),
        "natural_counts_json": _json(natural),
        "n_industrial_tagged_features": n_ind_features,
        "industrial_tag_counts_json": _json(industrial),
        "nearest_industrial_tagged_m": round(near_ind[0], 1) if near_ind else None,
        "nearest_industrial_tagged_tag": near_ind[1] if near_ind else None,
        "n_road_features": sum(roads.values()),
        "road_class_counts_json": _json(roads),
        "nearest_road_m": round(near_road[0], 1) if near_road else None,
        "nearest_road_class": near_road[1] if near_road else None,
    }


# ----------------------------------------------------------------------- input / records
def load_clusters(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stage 3 cluster file not found: {path} (run terra.pipeline first)")
    df = pd.read_csv(path)
    missing = [c for c in ("cluster_id", "latitude", "longitude") if c not in df.columns]
    if missing:
        raise ValueError(f"cluster file is missing columns: {missing}")
    if df["cluster_id"].isna().any() or not df["cluster_id"].is_unique:
        raise ValueError("cluster_id must be non-null and unique")
    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["latitude", "longitude"]].isna().any().any():
        raise ValueError("latitude/longitude contain missing or non-numeric values")
    if not df["latitude"].between(-90, 90).all() or not df["longitude"].between(-180, 180).all():
        raise ValueError("latitude/longitude out of range")
    return df.reset_index(drop=True)


def make_record(row, radius_m: int, qhash: str, fr: FetchResult) -> dict:
    n_cells = getattr(row, "n_cells", None)
    base = {
        "cluster_id": str(row.cluster_id), "latitude": float(row.latitude),
        "longitude": float(row.longitude),
        "n_cells": int(n_cells) if n_cells is not None and not pd.isna(n_cells) else None,
        "radius_m": int(radius_m), "query_version": QUERY_VERSION, "query_hash": qhash,
        "http_attempts": fr.attempts, "from_cache": fr.from_cache,
        "queried_utc": fr.fetched_utc or utc_now_iso(),
    }
    if fr.ok:
        summary = summarise_elements(fr.response.get("elements", []), base["latitude"], base["longitude"])
        status = "ok_features" if summary["n_elements"] > 0 else "ok_no_features"
        return {**base, "status": status, "error_type": None, "error_detail": None, **summary}
    return {**base, "status": "failed", "error_type": fr.error_type,
            "error_detail": fr.error_detail, **empty_summary()}


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_progress(path: Path) -> dict:
    """Latest record per cluster_id. A truncated final line (crash mid-write) is ignored."""
    latest: dict = {}
    if not Path(path).exists():
        return latest
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and "cluster_id" in rec:
                latest[rec["cluster_id"]] = rec
    return latest


def select_work(clusters: pd.DataFrame, progress: dict, radius_m: int,
                limit: Optional[int] = None, retry_failed: bool = True):
    """Decide which clusters still need a query. Returns (todo, n_done, n_failed_skipped)."""
    rows = clusters.head(limit) if limit is not None else clusters
    todo, n_done, n_failed_skipped = [], 0, 0
    for row in rows.itertuples(index=False):
        q = build_query(float(row.latitude), float(row.longitude), radius_m)
        qh = query_hash(q)
        rec = progress.get(str(row.cluster_id))
        if rec and str(rec.get("status", "")).startswith("ok") \
                and rec.get("radius_m") == radius_m and rec.get("query_hash") == qh:
            n_done += 1
        elif rec and rec.get("status") == "failed" and not retry_failed:
            n_failed_skipped += 1
        else:
            todo.append((row, q, qh))
    return todo, n_done, n_failed_skipped


# --------------------------------------------------------------------------- orchestrator
def run_enrichment(clusters: pd.DataFrame, client, progress_path: Path, radius_m: int,
                   limit: Optional[int] = None, retry_failed: bool = True,
                   max_consecutive_failures: int = 5, log=print) -> dict:
    progress = load_progress(progress_path)
    todo, n_done, n_failed_skipped = select_work(clusters, progress, radius_m, limit, retry_failed)
    stats = {"to_query": len(todo), "already_done_skipped": n_done,
             "failed_skipped": n_failed_skipped, "processed": 0, "succeeded": 0, "failed": 0,
             "aborted": None}
    consecutive = 0
    try:
        for i, (row, q, qh) in enumerate(todo, 1):
            try:
                fr = client.fetch(q)
            except RequestBudgetExceeded:
                stats["aborted"] = "request_budget_exhausted"
                break
            rec = make_record(row, radius_m, qh, fr)
            append_jsonl(progress_path, rec)
            stats["processed"] += 1
            src = "cache" if fr.from_cache else f"http x{fr.attempts}"
            if fr.ok:
                stats["succeeded"] += 1
                consecutive = 0
                log(f"[{i}/{len(todo)}] {rec['cluster_id']} {rec['status']} "
                    f"elements={rec['n_elements']} ({src})")
            else:
                stats["failed"] += 1
                consecutive += 1
                log(f"[{i}/{len(todo)}] {rec['cluster_id']} FAILED {rec['error_type']} ({src})")
                if consecutive >= max_consecutive_failures:
                    stats["aborted"] = "too_many_consecutive_failures"
                    break
    except KeyboardInterrupt:
        stats["aborted"] = "interrupted"
    return stats


# ---------------------------------------------------------------------------- outputs
def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    try:
        os.replace(tmp, path)
    except PermissionError as e:
        raise RuntimeError(f"Cannot write {path}. If it is open in Excel, close it and re-run.") from e


def build_outputs(clusters: pd.DataFrame, progress: dict, out_csv: Path, failures_csv: Path,
                  summary_json: Path, params: dict, run_stats: dict) -> dict:
    rows = []
    for row in clusters.itertuples(index=False):
        rec = progress.get(str(row.cluster_id))
        if rec is None:
            n_cells = getattr(row, "n_cells", None)
            rec = {"cluster_id": str(row.cluster_id), "latitude": float(row.latitude),
                   "longitude": float(row.longitude),
                   "n_cells": int(n_cells) if n_cells is not None and not pd.isna(n_cells) else None,
                   "status": "not_attempted"}
        rows.append({c: rec.get(c) for c in OUTPUT_COLUMNS})
    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    _atomic_write(out_csv, lambda p: table.to_csv(p, index=False, encoding="utf-8"))

    failed = table[table["status"] == "failed"][
        ["cluster_id", "latitude", "longitude", "error_type", "error_detail",
         "http_attempts", "queried_utc"]]
    _atomic_write(failures_csv, lambda p: failed.to_csv(p, index=False, encoding="utf-8"))

    counts = {k: int(v) for k, v in table["status"].value_counts().items()}
    summary = {
        "generated_utc": utc_now_iso(), "terra_version": __version__,
        "query_version": QUERY_VERSION, "params": params, "n_clusters": int(len(table)),
        "status_counts": counts, "last_run": run_stats, "limitations": LIMITATIONS,
    }
    _atomic_write(summary_json, lambda p: Path(p).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"))
    return summary


# -------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TERRA stage 4: OSM context enrichment")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--clusters", type=Path, default=None,
                    help="default: <root>/data/processed/persistent_clusters.csv")
    ap.add_argument("--radius-m", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="only consider the first N clusters")
    ap.add_argument("--contact", default=None, help="email or URL put in the User-Agent (required for live runs)")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--min-interval-s", type=float, default=5.0)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--backoff-base-s", type=float, default=10.0)
    ap.add_argument("--max-requests", type=int, default=400, help="hard cap on HTTP requests this run")
    ap.add_argument("--max-consecutive-failures", type=int, default=5)
    ap.add_argument("--no-retry-failed", action="store_true",
                    help="do not re-attempt clusters that failed in an earlier run")
    ap.add_argument("--dry-run", action="store_true", help="plan only; makes no HTTP requests")
    a = ap.parse_args(argv)

    root = a.root
    clusters_path = a.clusters or (root / "data/processed/persistent_clusters.csv")
    cache = DiskCache(root / "data/cache/osm")
    progress_path = root / "data/interim/osm_progress.jsonl"
    clusters = load_clusters(clusters_path)
    progress = load_progress(progress_path)
    todo, n_done, n_failed_skipped = select_work(clusters, progress, a.radius_m, a.limit,
                                                 not a.no_retry_failed)
    n_cached = sum(1 for _, _, qh in todo if cache.get(qh) is not None)
    n_http = len(todo) - n_cached
    print(f"clusters in file: {len(clusters)} | to query: {len(todo)} "
          f"(cached: {n_cached}, need HTTP: {n_http}) | already done: {n_done} "
          f"| failed-skipped: {n_failed_skipped}")
    print(f"minimum time for HTTP requests at {a.min_interval_s:g}s spacing: "
          f"~{n_http * a.min_interval_s / 60:.1f} min (plus query time)")
    if a.dry_run:
        print("dry run: no requests made, no files written.")
        return 0
    if not a.contact:
        print("ERROR: --contact is required for live runs (identifies you to the Overpass operators), "
              "e.g. --contact your.name@example.com", file=sys.stderr)
        return 2

    ua = f"TERRA-research-prototype/{__version__} (academic research; contact: {a.contact})"
    client = OverpassClient(cache, ua, RateLimiter(a.min_interval_s), endpoint=a.endpoint,
                            max_attempts=a.max_attempts, backoff_base_s=a.backoff_base_s,
                            max_requests=a.max_requests)
    stats = run_enrichment(clusters, client, progress_path, a.radius_m, a.limit,
                           not a.no_retry_failed, a.max_consecutive_failures)
    stats.update({"http_requests_made": client.requests_made, "cache_hits": client.cache_hits})
    params = {"radius_m": a.radius_m, "endpoint": a.endpoint, "min_interval_s": a.min_interval_s,
              "max_attempts": a.max_attempts, "max_requests": a.max_requests, "limit": a.limit}
    summary = build_outputs(clusters, load_progress(progress_path),
                            root / "data/processed/osm_context.csv",
                            root / "reports/osm_failures.csv",
                            root / "reports/osm_enrichment_summary.json", params, stats)
    print(json.dumps({"status_counts": summary["status_counts"], "last_run": stats}, indent=2))
    if stats["aborted"]:
        print(f"NOTE: run stopped early ({stats['aborted']}). Re-run the same command to resume.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
