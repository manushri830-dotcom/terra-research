"""Static figures for the research summary (no interactivity)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_figures(df: pd.DataFrame, clusters: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hexbin(df["longitude"], df["latitude"], gridsize=180, bins="log", cmap="Greys", mincnt=1)
    if len(clusters):
        t2 = clusters["type2_fraction"] >= 0.5
        ax.scatter(clusters.loc[~t2, "longitude"], clusters.loc[~t2, "latitude"], s=10,
                   c="tab:orange", label="persistent, <50% type-2 detections")
        ax.scatter(clusters.loc[t2, "longitude"], clusters.loc[t2, "latitude"], s=10,
                   c="tab:blue", label="persistent, majority type-2 detections")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Persistent thermal candidates vs. all VIIRS detections (Jan-Mar 2026)")
    ax.legend(loc="lower left", fontsize=8)
    p = out_dir / "persistent_candidates_map.png"; fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig); paths.append(p)

    if len(clusters):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(clusters["active_days"], bins=30)
        ax.set_xlabel("Active days in cluster (of 90)"); ax.set_ylabel("Clusters")
        ax.set_title("Persistence of candidate clusters")
        p = out_dir / "cluster_active_days.png"; fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig); paths.append(p)
    return paths
