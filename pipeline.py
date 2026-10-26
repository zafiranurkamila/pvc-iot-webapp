"""Inference pipeline of the IoT web application: 12-lead ECG -> PVC beats -> PSD + CWT -> HDR-CNN -> RVOT/LVOT.

All signal processing and the model come from the research package `pvc_localization`
(https://github.com/zafiranurkamila/pvc-localization-cnn), so the web application reproduces the processing of the
paper exactly: band-pass 0.5-40 Hz and 50 Hz notch, R peaks on lead II, PVC beats by the largest gap in QRS width,
800-ms windows z-scored per lead, PSD (Welch, 12 x 32 bins) and CWT scalogram (12 x 64 x 200), and a patient-level
decision from the mean logit margin of the PVC beats.
"""
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

def find_research_repo() -> Path:
    """PVC_REPO if set, otherwise the first folder next to this repository (or one level up) that contains
    src/pvc_localization."""
    if os.environ.get("PVC_REPO"):
        return Path(os.environ["PVC_REPO"])
    here = Path(__file__).resolve().parent
    for base in (here.parent, here.parent.parent):
        for cand in (base / "pvc-localization-cnn", base / "pvc-localization-cnn" / "pvc-localization-cnn"):
            if (cand / "src" / "pvc_localization").is_dir():
                return cand
    raise FileNotFoundError("research repository pvc-localization-cnn not found; set PVC_REPO to its folder")


RESEARCH_REPO = find_research_repo()
sys.path.insert(0, str(RESEARCH_REPO / "src"))

from pvc_localization import config  # noqa: E402
from pvc_localization.features.psd import flatten_psd_features  # noqa: E402
from pvc_localization.features.wavelet import extract_cwt_scalogram  # noqa: E402
from pvc_localization.models.fusion import FusionCNN  # noqa: E402
from pvc_localization.preprocessing.beat_detection import (  # noqa: E402
    detect_r_peaks, flag_wide_beats, qrs_width_ms)
from pvc_localization.preprocessing.filtering import clean_signal  # noqa: E402
from pvc_localization.preprocessing.normalization import zscore_normalize  # noqa: E402

MODEL_PATH = Path(os.environ.get("PVC_MODEL", RESEARCH_REPO / "results" / "models" / "psd_wavelet.pt"))
FS = config.SAMPLING_RATE_HZ
REF = config.LEADS.index(config.RPEAK_DETECTION_LEAD)
LABELS = ("RVOT", "LVOT")
SHOW_LEAD = "V1"  # lead shown in the representation view of the dashboard
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))


class PVCLocalizer:
    """Loads the trained model once and runs the full pipeline on one recording."""

    def __init__(self, model_path: Path = MODEL_PATH):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        hp = ckpt["hyperparameters"]
        self.features = ckpt["features"]
        self.model = FusionCNN(self.features, n_filters=hp["n_filters"], kernel_size=hp["kernel_size"],
                               dropout=hp["dropout"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.model_name = ckpt.get("scenario", model_path.stem)
        self.n_parameters = sum(p.numel() for p in self.model.parameters())

    @staticmethod
    def read_csv(data: bytes) -> np.ndarray:
        """12-lead ECG CSV with one column per lead (names I, II, III, aVR, aVL, aVF, V1-V6), sampled at 2000 Hz."""
        df = pd.read_csv(io.BytesIO(data))
        missing = [lead for lead in config.LEADS if lead not in df.columns]
        if missing:
            raise ValueError(f"missing lead columns: {', '.join(missing)}")
        return df[config.LEADS].to_numpy(dtype=float)

    def predict(self, ecg: np.ndarray) -> dict:
        t = {}
        t0 = time.perf_counter()
        filtered = clean_signal(ecg)
        t["filtering"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        sig = filtered[:, REF]
        peaks = detect_r_peaks(sig)
        widths = np.array([qrs_width_ms(sig, p) for p in peaks]) if len(peaks) else np.array([])
        wide = flag_wide_beats(widths) if len(peaks) >= 2 else np.zeros(len(peaks), dtype=bool)
        pre, post = config.BEAT_PRE_SAMPLES, config.BEAT_POST_SAMPLES
        kept = (peaks - pre >= 0) & (peaks + post <= len(sig))
        pvc_idx = np.where(wide & kept)[0]
        t["beat_extraction"] = time.perf_counter() - t0

        result = {
            "model": self.model_name, "sampling_rate_hz": FS, "duration_s": round(len(ecg) / FS, 2),
            "r_peaks": [{"t": round(float(p) / FS, 4), "qrs_ms": round(float(w), 1), "pvc": bool(wide[i])}
                        for i, (p, w) in enumerate(zip(peaks, widths))],
            "lead_ii": downsample(sig, FS, 250),
        }
        if len(pvc_idx) == 0:
            result.update(status="no_pvc", message="No PVC beat was detected in this recording.",
                          timing_ms={k: round(v * 1000, 1) for k, v in t.items()})
            return result

        t0 = time.perf_counter()
        beats = np.stack([zscore_normalize(filtered[peaks[i] - pre:peaks[i] + post].T) for i in pvc_idx])
        t["normalization"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        psd = np.stack([flatten_psd_features(b) for b in beats])
        t["psd"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cwt = np.stack([extract_cwt_scalogram(b) for b in beats])
        t["cwt"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model({"psd": torch.tensor(psd, dtype=torch.float32),
                                 "wavelet": torch.tensor(cwt, dtype=torch.float32)}).double()
        t["cnn_inference"] = time.perf_counter() - t0
        margin = (logits[:, 1] - logits[:, 0]).numpy()
        prob = torch.softmax(logits, dim=1)[:, 1].numpy()

        patient_margin = float(margin.mean())
        p_lvot = float(1 / (1 + np.exp(-patient_margin)))
        label = LABELS[int(patient_margin > 0)]
        # what the two branches see: PSD and CWT scalogram of lead V1, averaged over the PVC beats
        v1 = config.LEADS.index(SHOW_LEAD)
        n_bins = psd.shape[1] // len(config.LEADS)
        edges = np.linspace(config.FEATURE_FMIN_HZ, config.FEATURE_FMAX_HZ, n_bins + 1)
        result["representations"] = {
            "lead": SHOW_LEAD,
            "psd_hz": np.round((edges[:-1] + edges[1:]) / 2, 2).tolist(),
            "psd_log10": np.round(psd.reshape(len(psd), len(config.LEADS), n_bins)[:, v1].mean(0), 3).tolist(),
            "cwt_hz": np.round(np.geomspace(config.CWT_FMIN_HZ, config.FEATURE_FMAX_HZ, cwt.shape[2]), 2).tolist(),
            "cwt_ms": [-config.BEAT_PRE_MS, config.BEAT_POST_MS],
            "cwt": np.round(cwt[:, v1].mean(0), 3).tolist(),
        }
        result.update(
            status="ok", prediction=label, p_lvot=round(p_lvot, 4),
            confidence=round(p_lvot if label == "LVOT" else 1 - p_lvot, 4),
            n_pvc_beats=int(len(pvc_idx)),
            beats=[{"beat": int(i) + 1, "t": round(float(peaks[i]) / FS, 3), "qrs_ms": round(float(widths[i]), 1),
                    "p_lvot": round(float(pr), 4), "vote": LABELS[int(m > 0)]}
                   for i, pr, m in zip(pvc_idx, prob, margin)],
            timing_ms={k: round(v * 1000, 1) for k, v in t.items()},
        )
        result["timing_ms"]["total"] = round(sum(t.values()) * 1000, 1)
        return result


def downsample(x: np.ndarray, fs: int, target_fs: int) -> dict:
    """Lead II at a lower rate for plotting in the browser."""
    step = max(1, fs // target_fs)
    return {"fs": fs / step, "values": np.round(x[::step], 2).tolist()}
