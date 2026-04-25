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
COPY pyproject.toml .
COPY sysadmin_env/ sysadmin_env/

# Install dependencies using uv
RUN uv pip install --system -e .

# Expose the server port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "sysadmin_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
