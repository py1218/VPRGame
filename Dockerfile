# Use a CUDA‑enabled PyTorch runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /

# System deps for building & Git (for HF code)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
       git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy & install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py /

# Expose the FastAPI port
EXPOSE 8080

# Entrypoint launches your start.sh (which runs Uvicorn)
CMD ["bash", "start.sh"]