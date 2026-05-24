# ── BatteryBench Dashboard ────────────────────────────────────
# Flask web dashboard — no GPU needed

FROM python:3.10-slim

LABEL maintainer="BatteryBench"
LABEL description="BatteryBench Flask Dashboard"

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
COPY templates/ templates/
COPY static/ static/

# ── Expose port ────────────────────────────────────────────────
EXPOSE 5000

# ── Default command ────────────────────────────────────────────
CMD ["python", "src/app.py"]
