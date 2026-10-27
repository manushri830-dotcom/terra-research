import _path  # noqa: F401
import unittest

from helpers import days, make_detections
from terra.compare import type2_overlap
from terra.config import PersistenceParams
from terra.ingest import clean_firms
from terra.persistence import find_persistent


class CompareTests(unittest.TestCase):
    def setUp(self):
        rows = (
            days(20.005, 80.005, 10, type=2)            # persistent, type 2
            + days(22.005, 82.005, 10, type=0)          # persistent, no type 2
            + [{"lat": 24.005, "lon": 84.005, "date": "2026-01-01", "frp": 1.0 + i, "type": 0}
               for i in range(12)]                        # matched exposure (12 obs, 1 day), no type 2
            + days(26.005, 86.005, 3, type=2)           # sparse cell with type 2
        )
        df, _ = clean_firms(make_detections(rows))
        self.res = find_persistent(df, PersistenceParams(min_active_days=10))
        self.out = type2_overlap(self.res.cell_stats, self.res.clusters, 10)

    def test_counts(self):
        t = self.out["persistent_vs_all_other_cells"]
        self.assertEqual((t["group_a_with_type2"], t["group_a_without_type2"]), (1, 1))
        self.assertEqual((t["group_b_with_type2"], t["group_b_without_type2"]), (1, 1))
        m = self.out["persistent_vs_exposure_matched_cells"]
        self.assertEqual(m["n_matched_non_persistent_cells"], 1)
        self.assertEqual(self.out["cluster_level"]["majority_type2_detections"], 1)
        self.assertEqual(self.out["cluster_level"]["no_type2_detections"], 1)
        self.assertEqual(self.out["type2_cell_coverage"]["cells_with_type2"], 2)
        self.assertEqual(self.out["type2_cell_coverage"]["of_which_persistent"], 1)

    def test_limitations_always_reported(self):
        self.assertGreaterEqual(len(self.out["limitations"]), 4)
        joined = " ".join(self.out["limitations"]).lower()
        self.assertIn("not confirmed industrial", joined)
        self.assertIn("circular", joined)

    def test_no_accuracy_style_keys(self):
        for bad in ("precision", "recall", "accuracy", "f1"):
            self.assertNotIn(bad, str(self.out.keys()).lower())


if __name__ == "__main__":
    unittest.main()
