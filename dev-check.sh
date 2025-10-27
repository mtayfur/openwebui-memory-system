#!/usr/bin/env bash

# Development tools script for openwebui-memory-system

set -e

if [ -f "./.venv/bin/python" ]; then
    PYTHON="./.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo "Python 3 is not installed. Please install Python 3 to proceed."
    exit 1    
fi

echo "🔧 Running development tools..."

echo "🎨 Formatting with Black..."
$PYTHON -m black .
echo "📦 Sorting imports with isort..."
$PYTHON -m isort .

echo "✅ All checks passed!"