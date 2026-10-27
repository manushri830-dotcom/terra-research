# TERRA: persistent thermal-anomaly analysis (NASA FIRMS VIIRS, India)

Exploratory research prototype. It identifies locations with *recurrent* satellite thermal detections
and adds descriptive geographic context. It does **not** identify or confirm what causes any anomaly.

## Status
| Stage | Component | State |
|---|---|---|
| 1 | Ingest + cleaning (`ingest.py`) | verified on real data (260,303 rows) |
| 2 | Persistence + adjacent-cell clustering (`persistence.py`) | verified: 716 persistent cells -> 314 clusters |
| 3 | Exploratory FIRMS type-2 overlap (`compare.py`) | verified; descriptive only |
| 4 | OSM context enrichment (`osm_client.py`, `osm_context.py`, `osm_verify.py`) | completed and verified: all 314 clusters have definitive OSM query results; 306 original successes and 8 recovered after HTTP 502 failures. Geographic context is descriptive, not source confirmation. |
| 5-7 | API, dashboard, ML | not built |

Stage 4 completed on 25 October 2026. The verifier reported VALID AND COMPLETE for all 314 clusters. OSM results reflect the time of querying and may be incomplete.

## Scientific scope and limitations
* Persistence = detections on >= 10 distinct UTC dates in a 0.01 deg cell; adjacent cells are merged
  (cluster active days are a union of dates). Window: Jan-Mar 2026, NOAA-20 VIIRS only.
* FIRMS `type 2` ("other static land source") is **not** confirmed industrial heat and may be partly
  circular with persistence. The overlap in `reports/type2_overlap.json` is descriptive only.
* OSM tags are volunteer-contributed, incomplete (especially in rural India), and were queried at run
  time, not in Jan-Mar 2026. They describe surroundings; they never confirm a heat source.
* The OSM search is centred on the cluster centroid with a fixed radius (default 1000 m); large
  multi-cell clusters can extend beyond it.
* `ok_no_features` = query succeeded, nothing matched. `failed` = query did not succeed, context
  unknown. These must never be merged.
* No external ground truth (e.g. a verified facility inventory) is used anywhere.

## Run (PowerShell, from the project folder)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# place the FIRMS CSV at data\raw\fire_archive_J1V-C2_810603.csv
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m terra.pipeline                       # Stages 1-3
.\.venv\Scripts\python.exe -m unittest discover -s tests           # expect: Ran 61 tests ... OK

# Stage 4 (public Overpass API; be polite)
.\.venv\Scripts\python.exe -m terra.osm_context --dry-run
.\.venv\Scripts\python.exe -m terra.osm_context --contact "you@example.com"   # resumable
.\.venv\Scripts\python.exe -m terra.osm_verify                     # exit 0 = valid and complete
```
Re-running `terra.osm_context` resumes: finished clusters are skipped, failed ones are retried.
Do not change `--radius-m` mid-way: it changes every query and invalidates earlier results.

## Outputs
Stages 1-3: `data/processed/persistent_clusters.csv`, `reports/type2_overlap.json`, `reports/run_manifest.json`, ...
Stage 4 (separate files; Stage 1-3 outputs are never modified): `data/cache/osm/`, `data/interim/osm_progress.jsonl`,
`data/processed/osm_context.csv`, `reports/osm_failures.csv`, `reports/osm_enrichment_summary.json`.
