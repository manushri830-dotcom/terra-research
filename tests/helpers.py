import _path  # noqa: F401
import pandas as pd
from terra.config import REQUIRED_COLUMNS


def make_detections(rows):
    """rows: dicts with at least lat, lon, date. Optional: conf, dn, type, frp, time."""
    recs = []
    for r in rows:
        recs.append({
            "latitude": r["lat"], "longitude": r["lon"], "brightness": 330.0, "scan": 0.5,
            "track": 0.5, "acq_date": r["date"], "acq_time": r.get("time", "0730"),
            "satellite": "N20", "instrument": "VIIRS", "confidence": r.get("conf", "n"),
            "version": "2", "bright_t31": 295.0, "frp": r.get("frp", 5.0),
            "daynight": r.get("dn", "D"), "type": r.get("type", 0),
        })
    return pd.DataFrame(recs, columns=REQUIRED_COLUMNS)


def days(lat, lon, n, start=1, **kw):
    """n detections at one location on n consecutive distinct dates (Jan 2026)."""
    return [{"lat": lat, "lon": lon, "date": f"2026-01-{start + k:02d}", **kw} for k in range(n)]
