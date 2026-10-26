"""System checks of the web application: valid requests, invalid inputs and robustness (no server crash).

Runs the Flask app in-process (test client), so no running server is needed. Each check sends one request and
compares the HTTP status (and, where relevant, the result) with the expected behaviour. Writes
results/system_checks.csv and prints a table.

Usage:  python system_checks.py
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import RESEARCH_REPO  # noqa: F401  (loads the research package first)
from server import SAMPLES, app

ROOT = Path(__file__).resolve().parent
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def main():
    c = app.test_client()
    sample = sorted(SAMPLES.glob("*.csv"))[0]
    ecg = pd.read_csv(sample)[LEADS]
    rows = []

    def check(name, expected, resp, ok):
        body = resp.get_json(silent=True) or {}
        rows.append({"check": name, "expected": expected, "http_status": resp.status_code,
                     "result": body.get("prediction") or body.get("status") or body.get("error", ""),
                     "pass": bool(ok)})

    r = c.post(f"/api/predict?sample={sample.name}")
    check("Valid recording from the device store", "200, origin returned", r,
          r.status_code == 200 and r.get_json().get("status") == "ok")

    r = c.post("/api/device/upload", data=sample.read_bytes(),
               headers={"X-Device-Id": "ecg-node-test", "X-Record-Id": sample.stem, "Content-Type": "text/csv"})
    h = c.get("/api/history").get_json()
    check("Valid upload from an IoT node", "200, logged with device ID", r,
          r.status_code == 200 and h and h[0]["device"] == "ecg-node-test")

    r = c.post("/api/predict", data={"file": (io.BytesIO(csv_bytes(ecg)), "upload.csv")},
               content_type="multipart/form-data")
    check("Valid manual CSV upload", "200, origin returned", r,
          r.status_code == 200 and r.get_json().get("status") == "ok")

    r = c.post("/api/device/upload", data=csv_bytes(ecg.drop(columns=["V6"])))
    check("Recording with a missing lead (V6)", "400, rejected", r, r.status_code == 400)

    r = c.post("/api/device/upload", data=b"")
    check("Empty request body", "400, rejected", r, r.status_code == 400)

    r = c.post("/api/device/upload", data=b"this is not an ECG file")
    check("Non-CSV payload", "400, rejected", r, r.status_code == 400)

    flat = pd.DataFrame(np.zeros((20000, 12)), columns=LEADS)
    r = c.post("/api/device/upload", data=csv_bytes(flat))
    check("Flat-line recording (10 s)", "no crash; no PVC or 422", r,
          r.status_code in (200, 422) and (r.status_code == 422 or r.get_json().get("status") == "no_pvc"))

    r = c.post("/api/device/upload", data=csv_bytes(ecg.iloc[:1000]))
    check("Very short recording (0.5 s)", "no crash; no PVC or 422", r,
          r.status_code in (200, 422) and (r.status_code == 422 or r.get_json().get("status") == "no_pvc"))

    r = c.post("/api/predict?sample=does_not_exist.csv")
    check("Unknown record in the device store", "404", r, r.status_code == 404)

    r = c.post("/api/predict?sample=../server.py")
    check("Path outside the device store", "404, not served", r, r.status_code == 404)

    r = c.post("/api/predict")
    check("Request without a file", "400, rejected", r, r.status_code == 400)

    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "system_checks.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(df.to_string(index=False))
    print(f"\n{int(df['pass'].sum())}/{len(df)} checks passed; written results/system_checks.csv")


if __name__ == "__main__":
    main()
