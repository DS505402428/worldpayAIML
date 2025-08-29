#!/bin/bash

echo "Starting AI Observability Demo..."
echo "=========================================="

# Bring everything up
docker compose up -d

echo "Waiting for Ollama to preload the model..."
until curl -sf http://localhost:11434/api/tags | grep -q "llama3.1:8b"; do
    echo "Waiting for Ollama..."
    sleep 5
done

echo "Ollama is ready with model llama3.1:8b!"
echo "=========================================="
echo "Grafana:    http://localhost:3000 (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo "Ollama:     http://localhost:11434"
echo "AI-Platform:http://localhost:8000"
