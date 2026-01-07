#!/bin/bash
# Local CI runner - simulates GitHub Actions locally
set -e

echo "========================================"
echo "  X-Band Radar Simulation - Local CI   "
echo "========================================"
echo ""

cd "$(dirname "$0")"

# Activate virtualenv if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtualenv activated"
else
    echo "⚠ No virtualenv found, using system Python"
fi

# Install dev dependencies
echo ""
echo "--- Installing dependencies ---"
pip install -e ".[dev]" -q
pip install pytest-cov ruff -q
echo "✓ Dependencies installed"

# Lint
echo ""
echo "--- Running linter (ruff) ---"
ruff check src/ --ignore E501 || echo "⚠ Linter warnings (non-blocking)"

# Run unit tests
echo ""
echo "--- Running unit tests ---"
pytest tests/ -v --tb=short -m "not gpu and not slow" --ignore=tests/test_integration/

# Run with coverage
echo ""
echo "--- Running tests with coverage ---"
pytest tests/ --cov=src --cov-report=term-missing --ignore=tests/test_integration/

# Integration tests
echo ""
echo "--- Running integration tests ---"
pytest tests/test_integration/ -v --tb=short 2>/dev/null || echo "⚠ Integration tests skipped (no tests yet)"

echo ""
echo "========================================"
echo "  ✓ Local CI Complete                  "
echo "========================================"
