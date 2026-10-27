"""Processing cost of the PSD-only and the PSD + CWT pipeline on the same recordings (conference reviewer comment 15).

Both models of the paper (fixed setting) run through the deployed pipeline on every recording in samples/; each
recording is processed REPEATS times and the median per stage is kept. Writes results/benchmark_models.csv and
prints the median time per recording and per PVC beat, and the agreement of the predictions.

Usage:  python benchmark_models.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import RESEARCH_REPO, PVCLocalizer

ROOT = Path(__file__).resolve().parent
MODELS = {"PSD only": "psd.pt", "PSD + CWT": "psd_wavelet.pt"}
REPEATS = 3


def main():
    labels = pd.read_excel(RESEARCH_REPO / "data" / "external" / "Diagnosis.xlsx").set_index("HospitalID")
    truth = labels.LeftRight.map({"Right": "RVOT", "Left": "LVOT"})
    rows = []
    for name, file in MODELS.items():
        loc = PVCLocalizer(RESEARCH_REPO / "results" / "models" / file)
        for path in sorted((ROOT / "samples").glob("*.csv")):
            ecg = loc.read_csv(path.read_bytes())
            runs = [loc.predict(ecg) for _ in range(REPEATS)]
            r = runs[-1]
            stages = {k: float(np.median([x["timing_ms"][k] for x in runs])) for k in r["timing_ms"]}
            rows.append({"model": name, "record": path.stem, "true": truth.get(int(path.stem), "?"),
                         "predicted": r.get("prediction"), "n_pvc_beats": r.get("n_pvc_beats", 0),
                         **{f"{k}_ms": v for k, v in stages.items()}})
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "benchmark_models.csv", index=False)
    for name, g in df.groupby("model", sort=False):
        per_beat = (g.total_ms / g.n_pvc_beats).median()
        print(f"{name:10s} median total {g.total_ms.median():7.1f} ms per recording, {per_beat:6.1f} ms per PVC beat, "
              f"CNN {g.cnn_inference_ms.median():5.1f} ms, correct {int((g.predicted == g.true).sum())}/{len(g)}")
    wide = df.pivot(index="record", columns="model", values="predicted")
    print("same prediction in", int((wide["PSD only"] == wide["PSD + CWT"]).sum()), "of", len(wide), "recordings")
    print("written results/benchmark_models.csv")


if __name__ == "__main__":
    main()
