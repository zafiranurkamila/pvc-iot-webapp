"""CWT acceleration in the deployed pipeline (AGERS 2026 camera-ready, reviewer 1).

The deployed PSD + CWT model runs through the server pipeline on the 12 functional-test recordings in samples/ with
three implementations of the same CWT scalogram:
  pywt          per beat with PyWavelets, as in the paper (CPU)
  batched-cpu   the same transform in one batched FFT per recording (pvc_localization.features.wavelet_fast), CPU, float64
  batched-gpu   the same, on the GPU in float32 (only when CUDA is available)
Each recording is processed REPEATS times per mode after a warm-up; the median per stage is kept. The beat-level
LVOT probabilities and the patient decision of each mode are compared with pywt.

Run on the PC that produced the paper's timings. Usage:  python benchmark_cwt.py
Output: results/benchmark_cwt.csv (per recording and mode) and results/benchmark_cwt.json (summary)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline import PVCLocalizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
REPEATS = 5


def main():
    modes = {"pywt": dict(cwt="pywt"), "batched-cpu": dict(cwt="batched", cwt_device="cpu")}
    if torch.cuda.is_available():
        modes["batched-gpu"] = dict(cwt="batched", cwt_device="cuda")
    samples = sorted((ROOT / "samples").glob("*.csv"))
    assert samples, "no recordings in samples/ (run fetch_samples.py)"
    rows, beats = [], {}
    for mode, kw in modes.items():
        loc = PVCLocalizer(**kw)
        ecgs = {p.stem: loc.read_csv(p.read_bytes()) for p in samples}
        for ecg in list(ecgs.values())[:2]:  # warm-up (GPU context, FFT plans, caches)
            loc.predict(ecg)
        for rec, ecg in ecgs.items():
            runs = [loc.predict(ecg) for _ in range(REPEATS)]
            r = runs[-1]
            tm = pd.DataFrame([x["timing_ms"] for x in runs]).median()
            rows.append({"mode": mode, "record": rec, "prediction": r.get("prediction"), "n_pvc_beats": r.get("n_pvc_beats", 0),
                         "width_gap_ms": r.get("width_gap_ms"), "warning": bool(r.get("warning")),
                         **{f"{k}_ms": round(v, 2) for k, v in tm.items()}})
            beats[(mode, rec)] = np.array([b["p_lvot"] for b in r.get("beats", [])])
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "benchmark_cwt.csv", index=False)

    ref = df[df["mode"] == "pywt"].set_index("record")
    summary = {"repeats": REPEATS, "recordings": len(samples), "total_pvc_beats": int(ref.n_pvc_beats.sum()),
               "cpu": torch.get_num_threads(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "modes": {}}
    for mode in modes:
        d = df[df["mode"] == mode].set_index("record").loc[ref.index]
        nb = d.n_pvc_beats.clip(lower=1)
        diffs = [np.abs(beats[(mode, r)] - beats[("pywt", r)]).max() for r in ref.index if len(beats[("pywt", r)])]
        summary["modes"][mode] = {
            "cwt_median_ms_per_recording": float(d.cwt_ms.median()),
            "cwt_sum_ms_12_recordings": float(d.cwt_ms.sum()),
            "cwt_ms_per_beat": float(d.cwt_ms.sum() / d.n_pvc_beats.sum()),
            "total_median_ms_per_recording": float(d.total_ms.median()),
            "total_sum_ms_12_recordings": float(d.total_ms.sum()),
            "cwt_share_of_total_pct": float(100 * d.cwt_ms.sum() / d.total_ms.sum()),
            "total_ms_34_beat_recording": float(d.total_ms[d.n_pvc_beats.idxmax()]),
            "speedup_cwt_vs_pywt": float(ref.cwt_ms.sum() / d.cwt_ms.sum()),
            "speedup_total_vs_pywt": float(ref.total_ms.sum() / d.total_ms.sum()),
            "predictions_equal_to_pywt": int((d.prediction == ref.prediction).sum()),
            "max_abs_beat_prob_diff_vs_pywt": float(max(diffs)),
        }
        s = summary["modes"][mode]
        print(f"{mode:<12} CWT median {s['cwt_median_ms_per_recording']:8.1f} ms/recording, {s['cwt_ms_per_beat']:7.2f} ms/beat "
              f"({s['speedup_cwt_vs_pywt']:6.1f}x); total median {s['total_median_ms_per_recording']:7.1f} ms, "
              f"34-beat recording {s['total_ms_34_beat_recording']:7.1f} ms; CWT share {s['cwt_share_of_total_pct']:.1f}%; "
              f"same decision {s['predictions_equal_to_pywt']}/12; max beat prob diff {s['max_abs_beat_prob_diff_vs_pywt']:.2e}")
    (OUT / "benchmark_cwt.json").write_text(json.dumps(summary, indent=1))
    print("written", OUT / "benchmark_cwt.json")


if __name__ == "__main__":
    main()
