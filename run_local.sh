#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p .cache/pip .cache/matplotlib .cache/pycache .home .tmp

export HOME="$PWD/.home"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
export TMPDIR="$PWD/.tmp"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

if [ ! -x ".venv/bin/streamlit" ]; then
  echo "Missing .venv/bin/streamlit. Install dependencies inside .venv first."
  exit 1
fi

exec .venv/bin/streamlit run src/app.py \
  --server.headless true \
  --server.address 127.0.0.1 \
  --server.port "${PORT:-8501}" \
  --browser.gatherUsageStats false
