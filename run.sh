#!/usr/bin/env bash
# MINDTRACE Startup Script

set -e

# Load virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Load environment variables if .env exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=========================================================="
echo " Starting MINDTRACE: Multimodal Mental Health Platform"
echo "=========================================================="

exec streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
