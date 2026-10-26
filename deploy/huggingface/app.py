"""Entry point of the Hugging Face Space (Gradio SDK, free CPU basic): runs the Flask server of the PVC Origin Monitor.

At start-up the code of the two public GitHub repositories (research package and web application) is downloaded,
the model weights (psd_wavelet.pt) and the sample recordings (samples/*.csv) stored in this Space are linked in,
and the Flask app is served on port 7860, the port Spaces expose.
"""
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = Path("/tmp/pvc-code")
REPOS = {"pvc-localization-cnn": "https://github.com/zafiranurkamila/pvc-localization-cnn/archive/refs/heads/main.zip",
         "pvc-iot-webapp": "https://github.com/zafiranurkamila/pvc-iot-webapp/archive/refs/heads/main.zip"}


def fetch(name: str, url: str) -> Path:
    dest = CODE / name
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(CODE)
        (CODE / f"{name}-main").rename(dest)
    return dest


research = fetch("pvc-localization-cnn", REPOS["pvc-localization-cnn"])
webapp = fetch("pvc-iot-webapp", REPOS["pvc-iot-webapp"])
os.environ["PVC_REPO"] = str(research)
os.environ["PVC_MODEL"] = str(HERE / "psd_wavelet.pt")
if (HERE / "samples").is_dir():
    shutil.copytree(HERE / "samples", webapp / "samples", dirs_exist_ok=True)

sys.path.insert(0, str(webapp))
os.chdir(webapp)
from server import app  # noqa: E402

app.run(host="0.0.0.0", port=7860)
