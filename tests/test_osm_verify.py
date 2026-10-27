import _path  # noqa: F401
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from terra import osm_context as oc
from terra import osm_verify as ov
from terra.osm_client import FetchResult, RequestBudgetExceeded

CLUSTERS = pd.DataFrame({
    "cluster_id": ["TC-00001", "TC-00002", "TC-00003"],
    "latitude": [21.105379123456, 20.966655, 18.6868],
    "longitude": [72.641102987654, 86.00944, 73.03633],
    "n_cells": [5, 6, 6],
})
EL = [{"type": "way", "center": {"lat": 21.1, "lon": 72.6}, "tags": {"landuse": "farmland"}}]


def ok(els):
    return FetchResult(True, {"elements": els}, attempts=1, fetched_utc="2026-10-24T00:00:00+00:00")


def bad():
    return FetchResult(False, attempts=3, error_type="http_504", error_detail="HTTP 504")


class Stub:
    def __init__(self, results): self.results = list(results)
    def fetch(self, q):
        r = self.results.pop(0)
        if isinstance(r, BaseException): raise r
        return r


class VerifyTests(unittest.TestCase):
    def build(self, results):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        self.root = Path(d.name)
        (self.root / "data/processed").mkdir(parents=True)
        CLUSTERS.to_csv(self.root / "data/processed/persistent_clusters.csv", index=False)
        self.progress = self.root / "data/interim/osm_progress.jsonl"
        oc.run_enrichment(CLUSTERS, Stub(results), self.progress, 1000, log=lambda *_: None)
        self.rebuild()

    def rebuild(self):
        oc.build_outputs(CLUSTERS, oc.load_progress(self.progress),
                         self.root / "data/processed/osm_context.csv",
                         self.root / "reports/osm_failures.csv",
                         self.root / "reports/osm_enrichment_summary.json", {}, {})

    def cli(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return ov.main(["--root", str(self.root)])

    def edit_table(self, fn):
        p = self.root / "data/processed/osm_context.csv"
        t = pd.read_csv(p); fn(t); t.to_csv(p, index=False)

    def test_complete(self):
        self.build([ok(EL), ok([]), ok(EL)])
        r = ov.verify(self.root)
        self.assertTrue(r["valid"] and r["complete"])
        self.assertEqual(self.cli(), 0)

    def test_failed_and_not_attempted_are_valid_but_incomplete(self):
        self.build([ok(EL), bad(), RequestBudgetExceeded("cap")])
        r = ov.verify(self.root)
        self.assertTrue(r["valid"])
        self.assertFalse(r["complete"])
        self.assertEqual(r["status_counts"], {"ok_features": 1, "failed": 1, "not_attempted": 1})
        self.assertEqual(self.cli(), 3)

    def test_detects_tampering(self):
        self.build([ok(EL), ok([]), ok(EL)])
        cases = {
            "latitude changed": lambda t: t.__setitem__("latitude", t["latitude"] + 0.001),
            "row removed": lambda t: t.drop(index=2, inplace=True),
            "ids reordered": lambda t: t.__setitem__("cluster_id", t["cluster_id"].iloc[::-1].values),
            "bad status": lambda t: t.__setitem__("status", ["ok_features", "weird", "ok_features"]),
            "ok_no_features with elements": lambda t: t.__setitem__("n_elements", [1, 5, 1]),
            "failed shown as zero": lambda t: (t.__setitem__("status", ["ok_features", "failed", "ok_features"])),
        }
        for name, fn in cases.items():
            with self.subTest(name):
                self.build([ok(EL), ok([]), ok(EL)])
                self.edit_table(fn)
                self.assertFalse(ov.verify(self.root)["valid"], name)
                self.assertEqual(self.cli(), 1)

    def test_detects_csv_stale_relative_to_progress(self):
        self.build([ok(EL), bad(), ok(EL)])
        oc.append_jsonl(self.progress, {**oc.load_progress(self.progress)["TC-00002"], "status": "ok_no_features"})
        r = ov.verify(self.root)
        self.assertFalse(r["valid"])
        self.assertTrue(any("out of date" in p for p in r["problems"]))

    def test_missing_files_reported_not_raised(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        r = ov.verify(Path(d.name))
        self.assertFalse(r["valid"])
        self.assertTrue(r["problems"])

    def test_retried_clusters_are_reported(self):
        self.build([bad(), ok([]), ok(EL)])
        oc.run_enrichment(CLUSTERS, Stub([ok(EL)]), self.progress, 1000, log=lambda *_: None)
        self.rebuild()
        r = ov.verify(self.root)
        self.assertTrue(r["complete"])
        self.assertEqual(r["clusters_with_multiple_progress_lines"], ["TC-00001"])


if __name__ == "__main__":
    unittest.main()
