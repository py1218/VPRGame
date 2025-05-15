############################################
#  Lisa‑Chat Pod — CUDA‑enabled base image #
############################################
FROM --platform=linux/amd64 pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# stdout / stderr unbuffered
ENV PYTHONUNBUFFERED=1

# ─────────────────── system deps ───────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

# ─────────────────── working dir ───────────────────
WORKDIR /workspace

# ─────────────────── Python deps ───────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────── App code & data ───────────────
# • runpod_worker.py      (main serverless handler / FastAPI)
# • start.sh              (launch script)
# • character_db/lisa_db.json  (RAG seed data)
COPY runpod_worker.py ./runpod_worker.py
COPY start.sh        ./start.sh
COPY character_db/   ./character_db/   

# Make sure the launcher is executable
RUN chmod +x /workspace/start.sh

# ─────────────────── Entrypoint ────────────────────
CMD ["bash", "/workspace/start.sh"]