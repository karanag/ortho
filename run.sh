#!/bin/bash

# Re-run the GraphCut pipeline exactly as validated in the CLI session.
# 1) Quick non-AUTO blend (fast sanity check, 3000px mosaic)
# 2) AUTO blend with candidate timeouts (rebuilds outputs with --fresh)
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

QUICK_LOG="console_log_quick.txt"
AUTO_LOG="console_log_auto.txt"

# Common configuration shared by both runs.
COMMON_ARGS=(
  --images ./images/2
  --out out
  --exhaustive
  --blend seamhybrid
  --bands 3
  --seam-scale 0.50
  --seam-method graphcut
  --no-flow-refine
  --max_mosaic_px 3000
  --debug-dir out
)

echo ">>> Starting quick GraphCut run (no AUTO)..."
echo ">>> Logging to ${QUICK_LOG}"
python3 -u main.py --fresh "${COMMON_ARGS[@]}" >"${QUICK_LOG}" 2>&1
echo ">>> Quick run finished."

echo
echo ">>> Starting AUTO run (graphcut search with timeouts)..."
echo ">>> Logging to ${AUTO_LOG}"
python3 -u main.py --fresh "${COMMON_ARGS[@]}" --auto >"${AUTO_LOG}" 2>&1
echo ">>> AUTO run finished. Final mosaic at out/orthomosaic_colmap.png"
