"""Central configuration. Every analysis parameter lives here so runs are reproducible."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "fire_archive_J1V-C2_810603.csv"

# Columns that must exist in a FIRMS VIIRS 375 m CSV export.
REQUIRED_COLUMNS = [
    "latitude", "longitude", "brightness", "scan", "track", "acq_date", "acq_time",
    "satellite", "instrument", "confidence", "version", "bright_t31", "frp",
    "daynight", "type",
]

VALID_CONFIDENCE = {"l", "n", "h"}      # VIIRS: low / nominal / high
VALID_DAYNIGHT = {"D", "N"}
# FIRMS 'type': 0 presumed vegetation fire, 1 active volcano,
# 2 other static land source, 3 offshore detection.
VALID_TYPES = {0, 1, 2, 3}


@dataclass(frozen=True)
class PersistenceParams:
    grid_deg: float = 0.01              # ~1.1 km in latitude; narrower in longitude
    min_active_days: int = 10           # distinct UTC dates with >=1 detection
    exclude_low_confidence: bool = False
    connectivity: int = 8               # 4 or 8-neighbour merging of adjacent cells

    def validate(self) -> None:
        if self.grid_deg <= 0:
            raise ValueError("grid_deg must be positive")
        if self.min_active_days < 1:
            raise ValueError("min_active_days must be >= 1")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
