#!/bin/bash

# This script executes the COLMAP orthomosaic pipeline and saves all console output.
# The 'set -e' command ensures that the script will exit immediately if any command fails.
set -e

echo ">>> Starting the orthomosaic pipeline..."
echo ">>> All output will be saved to console_log.txt"

# Execute the main python script and redirect both stdout and stderr to the log file.
python3 -u main.py \
  --images ./images/3 \
  --out out \
  --fresh \
  --exhaustive \
  --blend seamhybrid \
  --bands 4 \
  --seam-scale 0.30 \
  --seam-method graphcut \
  --flow-max-px 4 \
  --stripe-flow-max-px 12 \
  --split-stripes \
  --debug-dir out \
  --flow-method farneback_slow \
  --flow-downscale 0.5 \
  --flow-smooth-ksize 21 \
  --remove-bg \
  --bg-token-file ./photoroom.txt \
  > console_log.txt 2>&1

echo ">>> Pipeline finished.. Check console_log.txt for the full output."
