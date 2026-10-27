import _path  # noqa: F401
import tempfile
import unittest
from pathlib import Path

from helpers import days, make_detections
from terra.ingest import clean_firms, load_firms


class IngestTests(unittest.TestCase):
    def test_missing_column_raises(self):
        df = make_detections(days(20.0, 80.0, 2)).drop(columns=["frp"])
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.csv"; df.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                load_firms(p)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_firms("does_not_exist.csv")

    def test_acq_time_leading_zero_preserved(self):
        df = make_detections([{"lat": 20.0, "lon": 80.0, "date": "2026-01-05", "time": "0721"}])
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.csv"; df.to_csv(p, index=False)
            clean, _ = clean_firms(load_firms(p))
        self.assertEqual(clean["acq_datetime_utc"].iloc[0].hour, 7)
        self.assertEqual(clean["acq_datetime_utc"].iloc[0].minute, 21)

    def test_invalid_rows_dropped_and_counted(self):
        rows = days(20.0, 80.0, 2) + [
            {"lat": 95.0, "lon": 80.0, "date": "2026-01-03"},            # bad latitude
            {"lat": 20.0, "lon": 80.0, "date": "2026-01-04", "frp": -1}, # negative FRP
            {"lat": 20.0, "lon": 80.0, "date": "2026-01-05", "conf": "x"},
            {"lat": 20.0, "lon": 80.0, "date": "2026-01-06", "type": 9},
            {"lat": 20.0, "lon": 80.0, "date": "not-a-date"},
            {"lat": 20.0, "lon": 80.0, "date": "2026-01-07", "time": "2575"},
        ]
        clean, rep = clean_firms(make_detections(rows))
        self.assertEqual(len(clean), 2)
        self.assertEqual(rep["rows_in"], 8)
        self.assertEqual(rep["rows_in"], rep["rows_out"] + sum(rep["removed"].values()))
        self.assertEqual(rep["removed"]["coordinates_out_of_range"], 1)
        self.assertEqual(rep["removed"]["negative_frp"], 1)

    def test_exact_duplicates_removed_zero_frp_kept(self):
        rows = days(20.0, 80.0, 1) * 2 + [{"lat": 21.0, "lon": 81.0, "date": "2026-01-02", "frp": 0.0}]
        clean, rep = clean_firms(make_detections(rows))
        self.assertEqual(rep["removed"]["exact_duplicates"], 1)
        self.assertEqual(rep["frp_zero_rows_kept"], 1)
        self.assertEqual(len(clean), 2)


if __name__ == "__main__":
    unittest.main()
