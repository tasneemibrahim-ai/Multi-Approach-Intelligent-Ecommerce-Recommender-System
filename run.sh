#!/bin/bash
# ─────────────────────────────────────────────────────────────
# run.sh — Quick launcher for the AIE425 Recommender System
# ─────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  AIE425 Intelligent Recommender System"
echo "  Alamein International University"
echo "============================================"
echo ""

# ── Check Python ─────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.9+."
    exit 1
fi

# ── Install dependencies ──────────────────────────────────────
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt -q

# ── Run CLI demo ──────────────────────────────────────────────
echo ""
echo "[2/3] Running CLI demo (all 9 algorithms, User #0)..."
python3 main.py
echo ""

# ── Launch Streamlit ──────────────────────────────────────────
echo "[3/3] Launching Streamlit dashboard..."
echo "      → Open http://localhost:8501 in your browser"
echo "      → Press Ctrl+C to stop"
echo ""
streamlit run app/streamlit_app.py --server.headless true
