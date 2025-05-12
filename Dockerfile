# Base image with Python 3.10
FROM python:3.10-slim

# System dependencies for building & git (for HF code)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (RunPod default)
EXPOSE 8080

# Entrypoint
CMD ["bash", "start.sh"]