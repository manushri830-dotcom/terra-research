import _path  # noqa: F401
import unittest

from helpers import days, make_detections
from terra.config import PersistenceParams
from terra.ingest import clean_firms
from terra.persistence import (assign_cells, connected_cell_labels, find_persistent,
                               legacy_cell_count)


def prep(rows):
    clean, _ = clean_firms(make_detections(rows))
    return clean


class PersistenceTests(unittest.TestCase):
    def test_threshold_is_inclusive_and_counts_distinct_days(self):
        df = prep(days(20.005, 80.005, 10) + days(25.005, 85.005, 9))
        r = find_persistent(df, PersistenceParams(min_active_days=10))
        self.assertEqual(len(r.clusters), 1)
        self.assertEqual(int(r.clusters.loc[0, "active_days"]), 10)

    def test_many_detections_same_day_is_one_active_day(self):
        rows = [{"lat": 20.005, "lon": 80.005, "date": "2026-01-01"} for _ in range(5)]
        rows = [dict(r, frp=1.0 + i) for i, r in enumerate(rows)]  # distinct rows, same day
        r = find_persistent(prep(rows), PersistenceParams(min_active_days=2))
        self.assertEqual(len(r.clusters), 0)
        self.assertEqual(int(r.cell_stats.loc[0, "active_days"]), 1)
        self.assertEqual(int(r.cell_stats.loc[0, "observation_count"]), 5)

    def test_float_boundary_cells(self):
        df = prep([{"lat": 0.29, "lon": 0.07, "date": "2026-01-01"}])
        cells = assign_cells(df, 0.01)
        self.assertEqual(int(cells.loc[0, "cell_i"]), 29)  # naive floor(0.29/0.01) gives 28
        self.assertEqual(int(cells.loc[0, "cell_j"]), 7)

    def test_adjacent_cells_merge_and_active_days_are_union(self):
        # source jitters between two touching cells; days overlap on 5..10
        rows = days(20.005, 80.005, 10, start=1) + days(20.015, 80.005, 10, start=5)
        r = find_persistent(prep(rows), PersistenceParams(min_active_days=10))
        self.assertEqual(len(r.clusters), 1)
        c = r.clusters.iloc[0]
        self.assertEqual(int(c["n_cells"]), 2)
        self.assertEqual(int(c["active_days"]), 14)          # union (1..14), not 20
        self.assertEqual(int(c["observation_count"]), 20)

    def test_separated_cells_stay_separate_and_connectivity_modes(self):
        gap = days(20.005, 80.005, 10) + days(20.035, 80.005, 10)   # 3 cells apart
        self.assertEqual(len(find_persistent(prep(gap), PersistenceParams()).clusters), 2)
        diag = days(20.005, 80.005, 10) + days(20.015, 80.015, 10)  # touch at a corner
        self.assertEqual(len(find_persistent(prep(diag), PersistenceParams(connectivity=8)).clusters), 1)
        self.assertEqual(len(find_persistent(prep(diag), PersistenceParams(connectivity=4)).clusters), 2)

    def test_connected_labels_empty(self):
        self.assertEqual(len(connected_cell_labels([], [], 8)), 0)

    def test_exclude_low_confidence(self):
        rows = days(20.005, 80.005, 10, conf="l")
        self.assertEqual(len(find_persistent(prep(rows), PersistenceParams()).clusters), 1)
        self.assertEqual(len(find_persistent(prep(rows),
                         PersistenceParams(exclude_low_confidence=True)).clusters), 0)

    def test_ids_deterministic_and_ranked(self):
        rows = days(20.005, 80.005, 12) + days(25.005, 85.005, 10)
        a = find_persistent(prep(rows), PersistenceParams()).clusters
        b = find_persistent(prep(rows), PersistenceParams()).clusters
        self.assertTrue(a.equals(b))
        self.assertEqual(a.loc[0, "cluster_id"], "TC-00001")
        self.assertEqual(int(a.loc[0, "active_days"]), 12)

    def test_legacy_count_matches_on_simple_data(self):
        df = prep(days(20.005, 80.005, 10) + days(25.005, 85.005, 4))
        self.assertEqual(legacy_cell_count(df), 1)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            PersistenceParams(grid_deg=0).validate()
        with self.assertRaises(ValueError):
            PersistenceParams(connectivity=6).validate()


if __name__ == "__main__":
    unittest.main()
