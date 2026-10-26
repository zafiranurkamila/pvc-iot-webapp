"""Copy the demo recordings from the research repository (data/processed) into samples/.

The 12 recordings are held-out test patients of the paper (6 RVOT, 6 LVOT), drawn at random (seed 42) from the
66 test patients, so the functional test uses patients that were never used for training or model selection.

Usage:  python fetch_samples.py
"""
import shutil
from pathlib import Path

from pipeline import RESEARCH_REPO

SAMPLE_IDS = {  # hospital_id: label in Diagnosis.xlsx
    1009731: "RVOT", 981890: "RVOT", 1020065: "RVOT", 827868: "RVOT", 869487: "RVOT", 911001: "RVOT",
    929662: "LVOT", 960365: "LVOT", 763863: "LVOT", 1022324: "LVOT", 851385: "LVOT", 928501: "LVOT",
}

out = Path(__file__).resolve().parent / "samples"
out.mkdir(exist_ok=True)
for pid in SAMPLE_IDS:
    src = RESEARCH_REPO / "data" / "processed" / f"{pid}.csv"
    if not src.exists():
        print("missing", src)
        continue
    shutil.copy(src, out / src.name)
print(f"{len(list(out.glob('*.csv')))} recordings in {out}")
