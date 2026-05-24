# ── BatteryBench Trainer ──────────────────────────────────────
# TensorFlow 2.10 with GPU support
# CUDA 11.2 · cuDNN 8 · Python 3.10

FROM tensorflow/tensorflow:2.10.1-gpu

LABEL maintainer="BatteryBench"
LABEL description="CNN vs LSTM vs Transformer · XJTU Battery SOH"

# ── System dependencies ────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libmysqlclient-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ────────────────────────────────────────────────
COPY src/ src/

# ── Default command ────────────────────────────────────────────
CMD ["python", "src/battery_trainer.py"]
