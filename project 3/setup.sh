#!/bin/bash
# E-Commerce Chatbot Setup Script
# This script helps you configure and initialize the chatbot

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$PROJECT_DIR")"

echo "🚀 E-Commerce Chatbot Setup"
echo "================================"
echo ""

# Check if .env exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "❌ .env file not found!"
    echo "Run: cp .env.template .env"
    exit 1
fi

# Activate venv
echo "📦 Activating Python environment..."
source "$PARENT_DIR/.venv/bin/activate"
echo "✅ Virtual environment activated"
echo ""

# Check if OPENAI_API_KEY is set
OPENAI_KEY=$(grep "OPENAI_API_KEY=" "$PROJECT_DIR/.env" | cut -d'=' -f2)
if [ "$OPENAI_KEY" = "your_openai_api_key_here" ] || [ -z "$OPENAI_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not configured in .env"
    echo "Please edit .env and add your OpenAI API key"
    echo "Get it from: https://platform.openai.com/api-keys"
    echo ""
fi

# Check if MySQL credentials are set
MYSQL_PASS=$(grep "MYSQL_PASSWORD=" "$PROJECT_DIR/.env" | cut -d'=' -f2)
if [ "$MYSQL_PASS" = "your_password_here" ] || [ -z "$MYSQL_PASS" ]; then
    echo "⚠️  MYSQL_PASSWORD not configured in .env"
    echo "Please edit .env and add your MySQL password"
    echo ""
fi

echo "📋 Next Steps:"
echo "1. Edit .env with your credentials:"
echo "   - OPENAI_API_KEY (from platform.openai.com)"
echo "   - MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD"
echo ""
echo "2. Setup MySQL database:"
echo "   mysql -u root -p < init_db.sql"
echo ""
echo "3. Load products:"
echo "   python product_loader.py"
echo ""
echo "4. Run the app:"
echo "   streamlit run app.py"
echo ""

echo "✅ Setup complete! Edit .env and follow the next steps above."
