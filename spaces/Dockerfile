FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies + Docker
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    docker.io \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    gradio \
    torch \
    transformers>=4.40.0 \
    datasets>=2.18.0 \
    peft>=0.10.0 \
    bitsandbytes>=0.43.0 \
    accelerate \
    trl>=0.8.0 \
    matplotlib \
    docker \
    fastapi \
    uvicorn \
    pydantic

# Copy application code
COPY . .

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

CMD ["/app/entrypoint.sh"]
