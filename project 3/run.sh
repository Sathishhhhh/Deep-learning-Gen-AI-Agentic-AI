#!/bin/bash
# Simple run script - activates venv and launches the app

PARENT_DIR="$(dirname "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")")"

echo "🎯 Starting E-Commerce Chatbot..."
echo ""

# Activate virtual environment
source "$PARENT_DIR/.venv/bin/activate"

# Check if .env is configured
if grep -q "your_openai_api_key_here" .env; then
    echo "❌ ERROR: OPENAI_API_KEY not configured in .env"
    echo "Please edit .env with your actual API key first"
    exit 1
fi

if grep -q "your_password_here" .env; then
    echo "❌ ERROR: MYSQL credentials not configured in .env"
    echo "Please edit .env with your MySQL credentials first"
    exit 1
fi

echo "✅ Configuration verified"
echo "🚀 Launching Streamlit app..."
echo ""

streamlit run app.py
