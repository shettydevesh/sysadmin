#!/bin/bash

# Start Docker daemon in background (if available)
if command -v dockerd &> /dev/null; then
    echo "Starting Docker daemon..."
    dockerd &
    sleep 5

    # Build sandbox image
    if docker info &> /dev/null; then
        echo "Building sandbox image..."
        docker build -t sysadmin-sandbox:latest -f docker/sandbox.Dockerfile . || true
    fi
fi

# Start the Gradio app
echo "Starting Gradio app..."
python app.py
