#!/usr/bin/env bash
# enroll_and_run.sh — batch-enroll photos then launch face recognition
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PHOTOS_DIR="$SCRIPT_DIR/photos"
TRACKER="$SCRIPT_DIR/hand_tracker.py"

# ─── helpers ──────────────────────────────────────────────────────────────────
ok()   { echo "[OK] $*"; }
info() { echo "[..] $*"; }
err()  { echo "[ERROR] $*" >&2; }

echo "============================================================"
echo " Hand Tracker — Enroll + Run"
echo "============================================================"
echo

# ─── 1. Check venv ────────────────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/Scripts/activate" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    err "Virtual environment not found."
    echo "  Run setup.sh first to install dependencies."
    exit 1
fi

if [ -f "$VENV_DIR/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/Scripts/activate"   # Windows (Git Bash)
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"       # Linux / macOS
fi
ok "Virtual environment active."
echo

# ─── 2. Check photos folder ───────────────────────────────────────────────────
IMAGE_EXTS=("jpg" "jpeg" "png" "bmp" "webp")
photo_count=0

if [ -d "$PHOTOS_DIR" ]; then
    for ext in "${IMAGE_EXTS[@]}"; do
        count=$(find "$PHOTOS_DIR" -maxdepth 1 -iname "*.$ext" 2>/dev/null | wc -l)
        photo_count=$((photo_count + count))
    done
fi

if [ "$photo_count" -eq 0 ]; then
    err "No photos found in: $PHOTOS_DIR"
    echo
    echo "  Drop photos into that folder and name each one after the person:"
    echo "    alice_johnson.jpg  →  'Alice Johnson'"
    echo "    rui.png            →  'Rui'"
    echo "    bob_smith.jpeg     →  'Bob Smith'"
    echo
    echo "  Then run this script again."
    exit 1
fi

ok "Found $photo_count photo(s) in photos/"
echo

# ─── 3. Enroll ────────────────────────────────────────────────────────────────
info "Starting enrollment (auto mode — no prompts)..."
echo

python "$TRACKER" --mode enroll --auto

echo
ok "Enrollment complete."
echo

# ─── 4. Launch face recognition ───────────────────────────────────────────────
echo "------------------------------------------------------------"
echo " Launching face recognition..."
echo " Keys:  Q = quit  |  F = toggle fullscreen"
echo " Window is resizable — drag the edges to any size."
echo "------------------------------------------------------------"
echo

python "$TRACKER" --mode face

echo
ok "Session ended."
