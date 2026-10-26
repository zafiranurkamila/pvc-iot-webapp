---
title: PVC Origin Monitor
emoji: 💓
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.9.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: RVOT/LVOT localization of PVCs from the 12-lead ECG
---

Inference API and dashboard of the PVC Origin Monitor (RVOT / LVOT localization of premature ventricular
contractions from the 12-lead ECG). Code: https://github.com/zafiranurkamila/pvc-iot-webapp

The Space runs on the free CPU tier with the Gradio SDK; `app.py` downloads the code from GitHub and serves the
Flask application on port 7860. Files stored in this Space: `app.py`, `requirements.txt`, `README.md`,
`psd_wavelet.pt` (trained model) and `samples/*.csv` (12 held-out recordings).

Sample recordings: Zheng J. et al., "A 12-lead ECG database to identify origins of idiopathic ventricular
arrhythmia containing 334 patients", Scientific Data 7, 98 (2020), doi:10.1038/s41597-020-0440-8 (CC BY 4.0).

Research prototype for decision support; not validated for clinical use.
