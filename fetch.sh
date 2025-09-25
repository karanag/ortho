#!/bin/bash
set -e

# --- Config ---
IP="194.68.245.48"
PORT="22169"
KEY="$HOME/.ssh/id_ed25519"
USER="root"

# --- Files to copy (remote paths) ---
FILES=(
    "/ortho/outDense/diagnostics/diagnostics.json"
    "/ortho/outDense/orthomosaic_colmap.png"
    "/ortho/console_log.txt"
)

# --- Destination ---
DEST="."

# --- Ensure ssh-agent is running ---
if [ -z "$SSH_AUTH_SOCK" ] || ! ssh-add -l &>/dev/null; then
    echo "Starting ssh-agent..."
    eval "$(ssh-agent -s)"
    ssh-add "$KEY"
fi

# --- Build scp command ---
REMOTE_FILES=()
for f in "${FILES[@]}"; do
    REMOTE_FILES+=("$USER@$IP:$f")
done

echo "Copying files: ${FILES[*]}"
scp -P "$PORT" -i "$KEY" "${REMOTE_FILES[@]}" "$DEST"
