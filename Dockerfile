# Use a CUDA‑enabled PyTorch runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# System deps for building & Git (for HF code)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
       git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy & install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the FastAPI port
EXPOSE 8080

# Entrypoint launches your start.sh (which runs Uvicorn)
CMD ["bash", "start.sh"]