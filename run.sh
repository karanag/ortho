#!/bin/bash

# This script executes the COLMAP orthomosaic pipeline and saves all console output.
# The 'set -e' command ensures that the script will exit immediately if any command fails.
set -e

echo ">>> Starting the orthomosaic pipeline..."
echo ">>> All output will be saved to console_log.txt"

# Execute the main python script and redirect both stdout and stderr to the log file.
python3 main.py \
  --images ./images/2 \
  --out outDense \
  --fresh \
  --exhaustive \
  --blend multiband \
  --bands 8 \
  --flow-max-px 3 \
  --stripe-flow-max-px 10 \
  --split-stripes \
  --debug-dir outDense/diagnostics \
  --flow-method farneback_slow \
  --flow-smooth-ksize 21 > console_log.txt 2>&1

echo ">>> Pipeline finished.. Check console_log.txt for the full output."