"""Tests for OSM enrichment logic. A stub client replaces HTTP entirely; no internet is used."""
import _path  # noqa: F401
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from terra import osm_context as oc
from terra.osm_client import FetchResult, RequestBudgetExceeded, query_hash

CLUSTERS = pd.DataFrame({
    "cluster_id": ["TC-00001", "TC-00002", "TC-00003", "TC-00004"],
    "latitude": [21.105379123456, 20.966655, 18.6868, 23.7530001],
    "longitude": [72.641102987654, 86.00944, 73.03633, 86.4070002],
    "n_cells": [5, 6, 6, 29],
    "active_days": [90, 89, 89, 88],
})

ELEMENTS = [
    {"type": "node", "lat": 21.001, "lon": 72.0, "tags": {"man_made": "works"}},
    {"type": "way", "center": {"lat": 21.003, "lon": 72.0}, "tags": {"landuse": "industrial"}},
    {"type": "way", "center": {"lat": 21.0, "lon": 72.002}, "tags": {"landuse": "farmland"}},
    {"type": "way", "center": {"lat": 21.0, "lon": 72.0005}, "tags": {"highway": "primary"}},
    {"type": "way", "center": {"lat": 21.0, "lon": 72.0001}, "tags": {"highway": "footway"}},
    {"type": "way", "tags": {"landuse": "residential"}},                      # no location
    {"type": "way", "center": {"lat": 21.01, "lon": 72.0}, "tags": {"natural": "wood"}},
    {"type": "node", "lat": 21.0, "lon": 72.0, "tags": {"natural": "cliff"}},  # not in NATURAL_VALUES
]


def ok(elements):
    return FetchResult(True, {"elements": elements}, attempts=1, fetched_utc="2026-10-24T00:00:00+00:00")


def fail(kind="http_504"):
    return FetchResult(False, attempts=3, error_type=kind, error_detail=kind)


class StubClient:
    def __init__(self, results):
        self.results = list(results)
        self.queries = []

    def fetch(self, query):
        self.queries.append(query)
        item = self.results.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class QueryAndSummaryTests(unittest.TestCase):
    def test_build_query(self):
        q = oc.build_query(21.105379, 72.641102, 1000)
        self.assertIn("(around:1000,21.105379,72.641102)", q)
        self.assertIn("[out:json][timeout:60]", q)
        self.assertTrue(q.rstrip().endswith("out tags center qt;"))
        self.assertEqual(q, oc.build_query(21.105379, 72.641102, 1000))
        self.assertNotEqual(query_hash(q), query_hash(oc.build_query(21.105379, 72.641102, 500)))

    def test_haversine(self):
        self.assertAlmostEqual(oc.haversine_m(0, 0, 1, 0), 111_195, delta=300)
        self.assertEqual(oc.haversine_m(10, 10, 10, 10), 0.0)

    def test_summarise_counts_and_distances(self):
        s = oc.summarise_elements(ELEMENTS, 21.0, 72.0)
        self.assertEqual(s["n_elements"], 8)
        self.assertEqual(s["n_elements_without_location"], 1)
        self.assertEqual((s["n_landuse_features"], s["landuse_industrial"], s["landuse_farmland"],
                          s["landuse_forest"], s["landuse_residential"]), (3, 1, 1, 0, 1))
        self.assertEqual(s["n_natural_features"], 1)                       # cliff ignored
        self.assertEqual(json.loads(s["natural_counts_json"]), {"wood": 1})
        self.assertEqual(s["n_industrial_tagged_features"], 2)
        self.assertEqual(json.loads(s["industrial_tag_counts_json"]),
                         {"landuse=industrial": 1, "man_made=works": 1})
        self.assertAlmostEqual(s["nearest_industrial_tagged_m"], 111.2, delta=1.0)
        self.assertEqual(s["nearest_industrial_tagged_tag"], "man_made=works")
        self.assertEqual(s["n_road_features"], 1)                          # footway ignored
        self.assertEqual(s["nearest_road_class"], "primary")
        self.assertTrue(50 < s["nearest_road_m"] < 54)

    def test_summarise_empty(self):
        s = oc.summarise_elements([], 21.0, 72.0)
        self.assertEqual(s["n_elements"], 0)
        self.assertEqual(s["n_landuse_features"], 0)
        self.assertIsNone(s["nearest_industrial_tagged_m"])
        self.assertIsNone(s["nearest_road_m"])
        self.assertNotIn("is_industrial", oc.OUTPUT_COLUMNS)


class RecordTests(unittest.TestCase):
    def row(self):
        return next(CLUSTERS.itertuples(index=False))

    def test_three_distinct_states(self):
        r_feat = oc.make_record(self.row(), 1000, "h", ok(ELEMENTS))
        r_none = oc.make_record(self.row(), 1000, "h", ok([]))
        r_fail = oc.make_record(self.row(), 1000, "h", fail())
        self.assertEqual(r_feat["status"], "ok_features")
        self.assertEqual(r_none["status"], "ok_no_features")
        self.assertEqual(r_none["n_elements"], 0)            # a real zero, not missing
        self.assertEqual(r_fail["status"], "failed")
        self.assertIsNone(r_fail["n_elements"])              # unknown, not zero
        self.assertEqual(r_fail["error_type"], "http_504")

    def test_identity_preserved_exactly(self):
        r = oc.make_record(self.row(), 1000, "h", ok([]))
        self.assertEqual((r["cluster_id"], r["latitude"], r["longitude"], r["n_cells"]),
                         ("TC-00001", 21.105379123456, 72.641102987654, 5))


class ProgressTests(unittest.TestCase):
    def test_latest_wins_and_truncated_line_ignored(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "p.jsonl"
            oc.append_jsonl(p, {"cluster_id": "A", "status": "failed"})
            oc.append_jsonl(p, {"cluster_id": "A", "status": "ok_features"})
            oc.append_jsonl(p, {"cluster_id": "B", "status": "ok_no_features"})
            with open(p, "a", encoding="utf-8") as f:
                f.write('{"cluster_id": "C", "stat')           # crash mid-write
            got = oc.load_progress(p)
            self.assertEqual(got["A"]["status"], "ok_features")
            self.assertEqual(set(got), {"A", "B"})

    def test_missing_progress_file(self):
        self.assertEqual(oc.load_progress(Path("nope.jsonl")), {})


class LoadClustersTests(unittest.TestCase):
    def write(self, df):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        p = Path(d.name) / "c.csv"; df.to_csv(p, index=False)
        return p

    def test_valid(self):
        self.assertEqual(len(oc.load_clusters(self.write(CLUSTERS))), 4)

    def test_invalid_inputs(self):
        with self.assertRaises(FileNotFoundError):
            oc.load_clusters("missing.csv")
        with self.assertRaises(ValueError):
            oc.load_clusters(self.write(CLUSTERS.drop(columns=["latitude"])))
        with self.assertRaises(ValueError):
            oc.load_clusters(self.write(pd.concat([CLUSTERS, CLUSTERS.head(1)])))   # duplicate id
        bad = CLUSTERS.copy(); bad.loc[0, "latitude"] = 123.0
        with self.assertRaises(ValueError):
            oc.load_clusters(self.write(bad))


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory(); self.addCleanup(self._d.cleanup)
        self.root = Path(self._d.name)
        self.progress = self.root / "data/interim/osm_progress.jsonl"
        self.quiet = lambda *_: None

    def run_(self, results, **kw):
        client = StubClient(results)
        kw.setdefault("radius_m", 1000)
        stats = oc.run_enrichment(CLUSTERS, client, self.progress, log=self.quiet, **kw)
        return client, stats

    def test_first_run_then_resume_retries_only_failed(self):
        client, st = self.run_([ok(ELEMENTS), ok([]), fail(), ok(ELEMENTS)])
        self.assertEqual((st["succeeded"], st["failed"], len(client.queries)), (3, 1, 4))
        client2, st2 = self.run_([ok(ELEMENTS)])             # only TC-00003 should be re-queried
        self.assertEqual(len(client2.queries), 1)
        self.assertEqual(st2["already_done_skipped"], 3)
        prog = oc.load_progress(self.progress)
        self.assertTrue(all(r["status"].startswith("ok") for r in prog.values()))

    def test_no_retry_failed_flag(self):
        self.run_([fail(), fail(), fail(), fail()], max_consecutive_failures=99)
        client, st = self.run_([], retry_failed=False)
        self.assertEqual(len(client.queries), 0)
        self.assertEqual(st["failed_skipped"], 4)

    def test_radius_change_invalidates_previous_results(self):
        self.run_([ok([])] * 4)
        client, _ = self.run_([ok([])] * 4, radius_m=500)
        self.assertEqual(len(client.queries), 4)

    def test_limit_considers_only_first_n_clusters(self):
        client, st = self.run_([ok([]), ok([])], limit=2)
        self.assertEqual(len(client.queries), 2)
        self.assertEqual(set(oc.load_progress(self.progress)), {"TC-00001", "TC-00002"})

    def test_aborts_after_consecutive_failures(self):
        client, st = self.run_([fail()] * 4, max_consecutive_failures=2)
        self.assertEqual(st["aborted"], "too_many_consecutive_failures")
        self.assertEqual(len(client.queries), 2)

    def test_success_resets_consecutive_failure_counter(self):
        _, st = self.run_([fail(), ok([]), fail(), ok([])], max_consecutive_failures=2)
        self.assertIsNone(st["aborted"])

    def test_budget_exhaustion_stops_cleanly(self):
        _, st = self.run_([ok([]), RequestBudgetExceeded("cap")])
        self.assertEqual(st["aborted"], "request_budget_exhausted")
        self.assertEqual(st["processed"], 1)

    def test_keyboard_interrupt_keeps_completed_work(self):
        _, st = self.run_([ok([]), KeyboardInterrupt()])
        self.assertEqual(st["aborted"], "interrupted")
        self.assertEqual(len(oc.load_progress(self.progress)), 1)

    def test_outputs_distinguish_states_and_preserve_identity(self):
        self.run_([ok(ELEMENTS), ok([]), fail("http_504"), RequestBudgetExceeded("cap")])
        out, fails, summ = (self.root / "o.csv", self.root / "f.csv", self.root / "s.json")
        summary = oc.build_outputs(CLUSTERS, oc.load_progress(self.progress), out, fails, summ,
                                   {"radius_m": 1000}, {"aborted": "request_budget_exhausted"})
        table = pd.read_csv(out)
        self.assertEqual(list(table["cluster_id"]), list(CLUSTERS["cluster_id"]))
        self.assertEqual(list(table["status"]),
                         ["ok_features", "ok_no_features", "failed", "not_attempted"])
        self.assertEqual(list(table["latitude"]), list(CLUSTERS["latitude"]))
        self.assertEqual(list(table["longitude"]), list(CLUSTERS["longitude"]))
        self.assertEqual(table.loc[1, "n_elements"], 0)                    # ok_no_features = real 0
        self.assertTrue(pd.isna(table.loc[2, "n_elements"]))               # failed = unknown
        self.assertEqual(list(pd.read_csv(fails)["cluster_id"]), ["TC-00003"])
        self.assertEqual(sum(summary["status_counts"].values()), len(CLUSTERS))
        self.assertGreaterEqual(len(summary["limitations"]), 4)
        self.assertEqual(list(table.columns), oc.OUTPUT_COLUMNS)

    def test_stage_1_to_3_inputs_are_not_modified(self):
        proc = self.root / "data/processed"; proc.mkdir(parents=True)
        src = proc / "persistent_clusters.csv"; CLUSTERS.to_csv(src, index=False)
        before = sha(src)
        clusters = oc.load_clusters(src)
        oc.run_enrichment(clusters, StubClient([ok([])] * 4), self.progress, 1000, log=self.quiet)
        oc.build_outputs(clusters, oc.load_progress(self.progress), proc / "osm_context.csv",
                         self.root / "reports/osm_failures.csv",
                         self.root / "reports/osm_enrichment_summary.json", {}, {})
        self.assertEqual(sha(src), before)
        self.assertTrue((proc / "osm_context.csv").exists())


class CliTests(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory(); self.addCleanup(self._d.cleanup)
        self.root = Path(self._d.name)
        (self.root / "data/processed").mkdir(parents=True)
        CLUSTERS.to_csv(self.root / "data/processed/persistent_clusters.csv", index=False)

    def listing(self):
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))

    def test_dry_run_writes_nothing_and_needs_no_contact(self):
        before = self.listing()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = oc.main(["--root", str(self.root), "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.listing(), before)
        self.assertIn("to query: 4", buf.getvalue())

    def test_live_run_without_contact_is_refused_before_any_request(self):
        before = self.listing()
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = oc.main(["--root", str(self.root)])
        self.assertEqual(rc, 2)
        self.assertIn("--contact", err.getvalue())
        self.assertEqual(self.listing(), before)


if __name__ == "__main__":
    unittest.main()
