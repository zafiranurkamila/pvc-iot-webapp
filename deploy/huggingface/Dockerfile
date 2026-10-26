# Inference API of the PVC Origin Monitor on Hugging Face Spaces (Docker SDK, CPU).
# The Space repository contains this Dockerfile, README.md, psd_wavelet.pt and samples/*.csv.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

RUN git clone --depth 1 https://github.com/zafiranurkamila/pvc-localization-cnn.git pvc-localization-cnn \
 && git clone --depth 1 https://github.com/zafiranurkamila/pvc-iot-webapp.git pvc-iot-webapp
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r pvc-iot-webapp/requirements.txt gunicorn

COPY psd_wavelet.pt models/psd_wavelet.pt
COPY samples/ pvc-iot-webapp/samples/
RUN chown -R user:user /home/user/app

ENV PVC_REPO=/home/user/app/pvc-localization-cnn \
    PVC_MODEL=/home/user/app/models/psd_wavelet.pt \
    MPLCONFIGDIR=/tmp
USER user
WORKDIR /home/user/app/pvc-iot-webapp
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "--timeout", "180", "server:app"]
