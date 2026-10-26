# PVC Origin Monitor – IoT web application

Web application for localizing the origin of premature ventricular contractions (RVOT vs LVOT) from the
12-lead ECG. An acquisition node (simulated by `device_simulator.py`) posts a recording to the server; the server
filters the ECG, extracts the PVC beats, computes the PSD and the CWT scalogram of each beat, classifies them with
the trained dual-representation CNN and shows the result on a dashboard.

The signal processing and the model come from the research repository
[pvc-localization-cnn](https://github.com/zafiranurkamila/pvc-localization-cnn); this repository adds only the
server, the dashboard, the device simulator and the functional test.

```
IoT node ──HTTP POST /api/device/upload──▶ Flask server ──▶ pipeline.py ──▶ dashboard (static/)
 (12-lead ECG CSV)                          filtering → PVC beats → PSD + CWT → CNN → RVOT / LVOT
```

## Setup

```bash
git clone https://github.com/zafiranurkamila/pvc-localization-cnn
git clone https://github.com/zafiranurkamila/pvc-iot-webapp
cd pvc-iot-webapp
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

The research repository is expected next to this one (`../pvc-localization-cnn`); otherwise set `PVC_REPO`.
The model weights (`results/models/psd_wavelet.pt` of the research repository, or a path in `PVC_MODEL`) are produced
by its training scripts and are not stored here.

ECG recordings: put 12-lead CSV files (columns `I, II, III, aVR, aVL, aVF, V1–V6`, 2000 Hz) in `samples/`. The
evaluation used recordings of the public database of Zheng et al., *Sci Data* 7, 98 (2020),
doi:10.1038/s41597-020-0440-8; patient recordings are not redistributed in this repository.

## Run

```bash
python fetch_samples.py          # copy the 12 demo recordings from ../pvc-localization-cnn/data/processed
python server.py                 # dashboard at http://localhost:5000
python device_simulator.py       # second terminal: an IoT node streams the samples to the server
python functional_test.py        # prediction vs database label and latency for every sample
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/status` | model name, features, parameter count |
| GET  | `/api/samples` | recordings available in `samples/` |
| POST | `/api/predict` | analyse an uploaded CSV (`file`) or a sample (`?sample=<name>`) |
| POST | `/api/device/upload` | IoT node endpoint: CSV body, headers `X-Device-Id`, `X-Record-Id` |
| GET  | `/api/history` | recent analyses shown in the device-stream table |

Research prototype for decision support; not validated for clinical use.
