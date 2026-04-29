#!/usr/bin/env bash
# setup.sh — install Python (if needed) + all dependencies
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIRED_MAJOR=3
REQUIRED_MINOR=9

# ─── helpers ──────────────────────────────────────────────────────────────────
ok()   { echo "[OK] $*"; }
info() { echo "[..] $*"; }
warn() { echo "[WARN] $*"; }
err()  { echo "[ERROR] $*" >&2; }

# Returns 0 if the given python command meets the minimum version
python_ok() {
    "$1" -c "import sys; sys.exit(0 if sys.version_info >= ($REQUIRED_MAJOR, $REQUIRED_MINOR) else 1)" 2>/dev/null
}

# Prints the version string of a python command
python_ver() {
    "$1" --version 2>&1 | grep -o '[0-9]*\.[0-9]*\.[0-9]*' | head -1
}

# ─── 1. Locate Python 3.9+ ────────────────────────────────────────────────────
echo "============================================================"
echo " Hand Tracker — Setup"
echo "============================================================"
echo

PYTHON=""
for cmd in py python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$cmd" &>/dev/null 2>&1; then
        if python_ok "$cmd"; then
            PYTHON="$cmd"
            break
        fi
    fi
done

# ─── 2. Auto-install Python if not found ──────────────────────────────────────
if [ -z "$PYTHON" ]; then
    warn "Python $REQUIRED_MAJOR.$REQUIRED_MINOR+ not found. Attempting automatic installation..."
    echo

    INSTALLED=0

    # Windows: winget (built into Windows 11)
    if command -v winget &>/dev/null 2>&1; then
        info "Installing Python 3.11 via winget..."
        winget install --id Python.Python.3.11 --source winget \
            --silent --accept-package-agreements --accept-source-agreements \
            || { warn "winget install returned a non-zero exit code — checking if Python is now available anyway."; }

        # Refresh known install paths
        for base in \
            "$USERPROFILE/AppData/Local/Programs/Python/Python311" \
            "/c/Program Files/Python311" \
            "/c/Users/$USERNAME/AppData/Local/Programs/Python/Python311"
        do
            # Convert Windows path to bash path if needed
            unix_base="$(echo "$base" | sed 's|\\|/|g' | sed 's|^\([A-Za-z]\):|/\L\1|')"
            if [ -f "$unix_base/python.exe" ]; then
                export PATH="$unix_base:$unix_base/Scripts:$PATH"
                break
            fi
        done

        # Re-check after PATH refresh
        for cmd in py python3.11 python3 python; do
            if command -v "$cmd" &>/dev/null 2>&1 && python_ok "$cmd"; then
                PYTHON="$cmd"
                INSTALLED=1
                break
            fi
        done

    # Linux / WSL: apt
    elif command -v apt-get &>/dev/null 2>&1; then
        info "Installing Python 3.11 via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
        PYTHON="python3.11"
        INSTALLED=1

    # macOS: Homebrew
    elif command -v brew &>/dev/null 2>&1; then
        info "Installing Python 3.11 via Homebrew..."
        brew install python@3.11
        PYTHON="$(brew --prefix python@3.11)/bin/python3.11"
        INSTALLED=1
    fi

    if [ $INSTALLED -eq 0 ] || [ -z "$PYTHON" ] || ! python_ok "$PYTHON"; then
        echo
        err "Could not install Python automatically."
        echo "  Please install Python $REQUIRED_MAJOR.$REQUIRED_MINOR+ manually:"
        echo "    Windows: https://www.python.org/downloads/"
        echo "    Tick 'Add Python to PATH' during installation."
        echo "  Then close this terminal and run setup.sh again."
        exit 1
    fi

    ok "Python installed successfully."
fi

PY_VER="$(python_ver "$PYTHON")"
ok "Using Python $PY_VER  ($PYTHON)"
echo

# ─── 3. Create virtual environment ────────────────────────────────────────────
if [ -f "$VENV_DIR/Scripts/activate" ] || [ -f "$VENV_DIR/bin/activate" ]; then
    ok "Virtual environment already exists."
else
    info "Creating virtual environment in .venv ..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created."
fi
echo

# ─── 4. Activate venv ─────────────────────────────────────────────────────────
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/Scripts/activate"   # Windows (Git Bash)
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"       # Linux / macOS
fi

# ─── 5. Upgrade pip ───────────────────────────────────────────────────────────
info "Upgrading pip..."
python -m pip install --upgrade pip --quiet
ok "pip ready."
echo

# ─── 6. Remove conflicting bare opencv-python ─────────────────────────────────
if python -m pip show opencv-python &>/dev/null 2>&1; then
    info "Removing conflicting opencv-python (replaced by opencv-contrib-python)..."
    python -m pip uninstall opencv-python -y --quiet
fi

# ─── 7. Detect GPU for onnxruntime selection ──────────────────────────────────
USE_GPU=0
if command -v nvidia-smi &>/dev/null 2>&1 && nvidia-smi &>/dev/null 2>&1; then
    ok "NVIDIA GPU detected — will install onnxruntime-gpu."
    USE_GPU=1
else
    info "No NVIDIA GPU detected — using CPU onnxruntime."
fi
echo

# ─── 8. Install packages ──────────────────────────────────────────────────────
info "Installing packages from requirements.txt (first run may take several minutes)..."
echo

# Install base requirements (uses onnxruntime CPU by default)
pip install -r "$SCRIPT_DIR/requirements.txt"

# Upgrade to GPU version if applicable
if [ $USE_GPU -eq 1 ]; then
    info "Upgrading to onnxruntime-gpu..."
    pip install --upgrade "onnxruntime-gpu>=1.17.0,<2.0" --quiet
    ok "onnxruntime-gpu installed."
fi

echo
echo "============================================================"
echo " Setup complete!"
echo
echo " Next steps:"
echo "   1. Drop photos into the 'photos/' folder (name each file"
echo "      after the person: alice_johnson.jpg, rui.png, ...)"
echo "   2. Run:  bash enroll_and_run.sh"
echo
echo " Or launch hand-tracking mode directly:"
echo "   bash run.bat  (Windows)  /  source .venv/Scripts/activate && python hand_tracker.py"
echo "============================================================"
echo
