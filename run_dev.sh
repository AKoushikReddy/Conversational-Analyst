#!/usr/bin/env bash
set -e
python -m venv .venv && source .venv/bin/activate
pip install -U pip -r requirements.txt
# Health-check Ollama (required per repo note: local + Ollama)
curl -s http://localhost:11434/api/tags >/dev/null || { echo "Start Ollama first: 'ollama serve'"; exit 1; }
# (Optional) ensure a model is present
ollama pull llama3 || true
# Start API
uvicorn server:app --reload --port 8000
