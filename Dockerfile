# Sysadmin Game Environment Server
FROM python:3.11-slim

# Install docker CLI for container management
RUN apt-get update && apt-get install -y \
    docker.io \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy project files
COPY pyproject.toml README.md openenv.yaml ./
COPY sysadmin_env/ sysadmin_env/

# Install dependencies using uv
RUN uv pip install --system -e .

# Expose the Hugging Face Spaces default port
ENV PORT=7860
EXPOSE 7860

# Run the server
CMD ["sh", "-c", "uvicorn sysadmin_env.server.app:app --host 0.0.0.0 --port ${PORT}"]
