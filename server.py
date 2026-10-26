"""Flask server of the IoT web application for PVC origin localization.

Endpoints
    GET  /                     dashboard (static/index.html)
    GET  /api/status           model and server information
    GET  /api/samples          sample recordings available on the server (samples/*.csv)
    POST /api/predict          analyse an uploaded 12-lead ECG CSV (form field "file") or a sample (?sample=<name>)
    POST /api/device/upload    endpoint for IoT acquisition nodes: raw CSV body, headers X-Device-Id / X-Record-Id
    GET  /api/history          recent analyses, newest first (dashboard monitoring view)

Run:  python server.py   (http://localhost:5000)
"""
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from pipeline import PVCLocalizer

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
localizer = PVCLocalizer()
history = deque(maxlen=50)


def analyse(data: bytes, source: str, device: str = "web-upload", record: str = "") -> dict:
    t0 = time.perf_counter()
    result = localizer.predict(localizer.read_csv(data))
    result["request_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    entry = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "device": device, "record": record or source,
             "status": result["status"], "prediction": result.get("prediction"),
             "confidence": result.get("confidence"), "n_pvc_beats": result.get("n_pvc_beats", 0),
             "request_ms": result["request_ms"]}
    history.appendleft(entry)
    result["record"] = entry["record"]
    return result


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    return jsonify(model=localizer.model_name, features=localizer.features, parameters=localizer.n_parameters,
                   sampling_rate_hz=2000, leads=12)


@app.get("/api/samples")
def samples():
    return jsonify(sorted(p.name for p in SAMPLES.glob("*.csv")) if SAMPLES.exists() else [])


@app.post("/api/predict")
def predict():
    try:
        if "sample" in request.args:
            path = SAMPLES / Path(request.args["sample"]).name  # no directories from the client
            if not path.is_file():
                return jsonify(error="sample not found"), 404
            return jsonify(analyse(path.read_bytes(), path.name, record=path.stem))
        f = request.files.get("file")
        if f is None:
            return jsonify(error="no file uploaded"), 400
        return jsonify(analyse(f.read(), f.filename, record=Path(f.filename).stem))
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.post("/api/device/upload")
def device_upload():
    try:
        result = analyse(request.get_data(), "device", device=request.headers.get("X-Device-Id", "unknown-device"),
                         record=request.headers.get("X-Record-Id", ""))
        return jsonify({k: result.get(k) for k in ("status", "prediction", "confidence", "p_lvot", "n_pvc_beats",
                                                    "request_ms", "record")})
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.get("/api/history")
def get_history():
    return jsonify(list(history))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
