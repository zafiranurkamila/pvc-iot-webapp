"""Functional test of the deployed pipeline on the sample recordings (held-out test patients of the paper).

For each recording in samples/ the pipeline is run as the server runs it; the prediction is compared with the
label of the database (Diagnosis.xlsx, LeftRight). Writes results/functional_test.csv and prints a summary.

Usage:  python functional_test.py
"""
import time
from pathlib import Path

import pandas as pd

from pipeline import RESEARCH_REPO, PVCLocalizer

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "results"


def main():
    labels = pd.read_excel(RESEARCH_REPO / "data" / "external" / "Diagnosis.xlsx").set_index("HospitalID")
    label = labels.LeftRight.map({"Right": "RVOT", "Left": "LVOT"})
    loc = PVCLocalizer()
    rows = []
    for path in sorted(SAMPLES.glob("*.csv")):
        t0 = time.perf_counter()
        r = loc.predict(loc.read_csv(path.read_bytes()))
        wall = (time.perf_counter() - t0) * 1000
        true = label.get(int(path.stem), "?")
        rows.append({"record": path.stem, "true": true, "predicted": r.get("prediction", "none"),
                     "correct": r.get("prediction") == true, "confidence": r.get("confidence"),
                     "n_pvc_beats": r.get("n_pvc_beats", 0), "duration_s": r["duration_s"],
                     **{f"{k}_ms": v for k, v in r["timing_ms"].items()}, "wall_ms": round(wall, 1)})
        print(f"{path.stem:>10}  true {true}  pred {rows[-1]['predicted']:>5}  conf {r.get('confidence')}  "
              f"beats {rows[-1]['n_pvc_beats']}  {wall:.0f} ms")
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "functional_test.csv", index=False)
    ok = df[df.predicted != "none"]
    print(f"\n{len(df)} recordings, {int((df.predicted != 'none').sum())} with PVC beats, "
          f"{int(ok.correct.sum())} correct ({ok.correct.mean():.1%}); "
          f"median processing {df.wall_ms.median():.0f} ms per recording, "
          f"{(df.wall_ms / df.n_pvc_beats.clip(lower=1)).median():.0f} ms per PVC beat")
    print("written", OUT / "functional_test.csv")


if __name__ == "__main__":
    main()
