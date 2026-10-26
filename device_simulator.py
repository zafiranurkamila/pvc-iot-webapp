"""Simulated IoT acquisition node: sends 12-lead ECG recordings to the server over HTTP, one at a time.

In the deployed system an ECG acquisition device would post each recording to /api/device/upload; here the
recordings in samples/ stand in for the device stream.

Usage:
    python device_simulator.py                       # all samples, 3 s apart
    python device_simulator.py --server http://192.168.1.10:5000 --interval 5 --device ecg-node-01
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent / "samples"


def main():
    ap = argparse.ArgumentParser()
    # 127.0.0.1 rather than "localhost": on Windows "localhost" tries IPv6 first and adds ~2 s per request,
    # because the Flask server listens on IPv4 (0.0.0.0) only
    ap.add_argument("--server", default="http://127.0.0.1:5000")
    ap.add_argument("--device", default="ecg-node-01")
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()
    for path in sorted(SAMPLES.glob("*.csv")):
        req = urllib.request.Request(f"{args.server}/api/device/upload", data=path.read_bytes(), method="POST",
                                     headers={"Content-Type": "text/csv", "X-Device-Id": args.device,
                                              "X-Record-Id": path.stem})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.loads(r.read())
        rtt = (time.perf_counter() - t0) * 1000
        print(f"{path.stem:>10}  {res.get('prediction') or res.get('status'):>6}  conf {res.get('confidence')}  "
              f"PVC beats {res.get('n_pvc_beats')}  server {res.get('request_ms')} ms  round trip {rtt:.0f} ms")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
