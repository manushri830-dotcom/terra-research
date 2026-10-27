"""Read-only verification of Stage 4 outputs against the Stage 3 cluster file.

    python -m terra.osm_verify            exit 0 = valid and complete
                                          exit 1 = structural problem (do not trust the outputs)
                                          exit 3 = valid but incomplete (failed / not_attempted rows)
Nothing is written or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .osm_context import OUTPUT_COLUMNS

ALLOWED = {"ok_features", "ok_no_features", "failed", "not_attempted"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> dict:
    root = Path(root)
    p = {
        "clusters": root / "data/processed/persistent_clusters.csv",
        "table": root / "data/processed/osm_context.csv",
        "failures": root / "reports/osm_failures.csv",
        "summary": root / "reports/osm_enrichment_summary.json",
        "progress": root / "data/interim/osm_progress.jsonl",
    }
    problems: list[str] = []
    warnings: list[str] = []
    out: dict = {"problems": problems, "warnings": warnings}

    missing = [k for k, v in p.items() if not v.exists()]
    if missing:
        problems.append("missing files: " + ", ".join(str(p[k]) for k in missing))
        out.update(valid=False, complete=False)
        return out

    clusters = pd.read_csv(p["clusters"])
    table = pd.read_csv(p["table"])
    out["n_clusters"] = int(len(clusters))
    out["stage3_clusters_sha256"] = _sha256(p["clusters"])

    if list(table.columns) != OUTPUT_COLUMNS:
        problems.append("osm_context.csv columns differ from the expected schema")
    if len(table) != len(clusters):
        problems.append(f"row count {len(table)} != cluster count {len(clusters)}")
    elif list(table["cluster_id"]) != list(clusters["cluster_id"]):
        problems.append("cluster_id values/order differ from persistent_clusters.csv")
    else:
        for col in ("latitude", "longitude"):
            if not np.array_equal(table[col].to_numpy(dtype=float), clusters[col].to_numpy(dtype=float)):
                problems.append(f"{col} differs from persistent_clusters.csv")

    bad_status = sorted(set(table["status"].dropna()) - ALLOWED) if "status" in table else []
    if bad_status or ("status" in table and table["status"].isna().any()):
        problems.append(f"invalid or missing status values: {bad_status}")
    counts = {str(k): int(v) for k, v in table["status"].value_counts().items()} if "status" in table else {}
    out["status_counts"] = counts

    if not problems:
        ne, et = table["n_elements"], table["error_type"]
        s = table["status"]
        rules = {
            "ok_features rows must have n_elements > 0": ((s == "ok_features") & ~(ne > 0)),
            "ok_no_features rows must have n_elements == 0": ((s == "ok_no_features") & ~(ne == 0)),
            "ok rows must not have error_type": (s.str.startswith("ok") & et.notna()),
            "failed rows must have error_type and unknown (empty) n_elements": ((s == "failed") & (et.isna() | ne.notna())),
            "not_attempted rows must have empty n_elements": ((s == "not_attempted") & ne.notna()),
        }
        for msg, bad in rules.items():
            if bad.any():
                problems.append(f"{msg}: {list(table.loc[bad, 'cluster_id'])[:5]}")

        failures = pd.read_csv(p["failures"])
        if set(failures["cluster_id"]) != set(table.loc[s == "failed", "cluster_id"]):
            problems.append("osm_failures.csv does not match the failed rows")

        summary = json.loads(p["summary"].read_text(encoding="utf-8"))
        if summary.get("status_counts") != counts or summary.get("n_clusters") != len(table):
            problems.append("osm_enrichment_summary.json is out of date with osm_context.csv")

        latest, lines, bad_lines = {}, {}, 0
        for line in p["progress"].read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad_lines += 1
                continue
            latest[rec["cluster_id"]] = rec
            lines[rec["cluster_id"]] = lines.get(rec["cluster_id"], 0) + 1
        out["progress_lines"] = int(sum(lines.values()))
        out["clusters_with_multiple_progress_lines"] = sorted(k for k, v in lines.items() if v > 1)
        if bad_lines:
            warnings.append(f"{bad_lines} unparseable line(s) in progress file (e.g. a crash mid-write)")
        unknown = set(latest) - set(clusters["cluster_id"])
        if unknown:
            problems.append(f"progress file has ids not in the cluster file: {sorted(unknown)[:5]}")
        mism = [r.cluster_id for r in table.itertuples()
                if (latest.get(r.cluster_id, {}).get("status", "not_attempted") != r.status)]
        if mism:
            problems.append("osm_context.csv is out of date with the progress file "
                            f"(re-run the enrichment command to rebuild it): {mism[:5]}")

    n_failed, n_na = counts.get("failed", 0), counts.get("not_attempted", 0)
    if n_failed:
        warnings.append(f"{n_failed} cluster(s) FAILED: their context is UNKNOWN, not 'no features'")
    if n_na:
        warnings.append(f"{n_na} cluster(s) not yet attempted")
    out["valid"] = not problems
    out["complete"] = out["valid"] and n_failed == 0 and n_na == 0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify TERRA Stage 4 outputs (read-only)")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    r = verify(ap.parse_args(argv).root)
    print(f"clusters in Stage 3 file : {r.get('n_clusters', '?')}")
    print(f"status counts            : {r.get('status_counts', {})}")
    if "progress_lines" in r:
        print(f"progress-file lines      : {r['progress_lines']} "
              f"(clusters with >1 line, i.e. retried: {len(r['clusters_with_multiple_progress_lines'])})")
    if "stage3_clusters_sha256" in r:
        print(f"persistent_clusters.csv  : sha256 {r['stage3_clusters_sha256']}")
    for w in r["warnings"]:
        print("WARNING:", w)
    for pr in r["problems"]:
        print("PROBLEM:", pr)
    if not r["valid"]:
        print("RESULT: INVALID - do not use these outputs until the problems are resolved.")
        return 1
    if not r["complete"]:
        print("RESULT: VALID BUT INCOMPLETE - re-run the enrichment command to resume/retry.")
        return 3
    print(f"RESULT: VALID AND COMPLETE - all {r['n_clusters']} clusters have a definitive OSM query result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
