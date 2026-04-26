#!/bin/bash

# Try Docker daemon briefly, don't block if it fails
if command -v dockerd &> /dev/null; then
    echo "Attempting Docker daemon..."
    dockerd &> /dev/null &
    sleep 3

    # Quick check with timeout - don't hang
    if timeout 5 docker info &> /dev/null 2>&1; then
        echo "Docker available, building sandbox..."
        timeout 120 docker build -t sysadmin-sandbox:latest -f docker/sandbox.Dockerfile . || echo "Sandbox build failed (non-critical)"
    else
        echo "Docker not available in this environment (expected on HF Spaces)"
    fi
fi

# Start the Gradio app
echo "Starting Gradio app..."
python app.py
