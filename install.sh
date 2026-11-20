#!/usr/bin/env bash
# Install rsi once and use it from any folder (like `claude`).
#
#   ./install.sh                 # venv in <repo>/.venv, editable install, GPU extras, launcher in ~/.local/bin
#   ./install.sh --no-gpu        # core only (no vLLM / math example deps)
#   ./install.sh --venv DIR --bin-dir DIR
#
# Editable install: `git pull` in this repo updates the `rsi` command. Re-run to repair/upgrade.
# Uninstall: rm ~/.local/bin/rsi (and the venv dir if you like).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
BIN_DIR="$HOME/.local/bin"
GPU=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --bin-dir) BIN_DIR="$2"; shift 2 ;;
    --no-gpu) GPU=0; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
say() { printf '\033[1;34m[rsi-install]\033[0m %s\n' "$*"; }

# ---- python >= 3.10
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then PY="$(command -v "$c")"; break; fi
done
[[ -n "$PY" ]] || { echo "need Python >= 3.10"; exit 1; }

# ---- venv + package
if [[ ! -x "$VENV/bin/python" ]]; then
  say "creating venv $VENV ($("$PY" --version))"
  "$PY" -m venv "$VENV"
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install -q --upgrade pip
say "installing rsi (editable) from $REPO"
"$VPY" -m pip install -q -e "$REPO"

# ---- GPU extras (vLLM etc.), matched to the NVIDIA driver's CUDA version
if [[ $GPU == 1 ]]; then
  if "$VPY" -c 'import vllm, torch, sys; sys.exit(not torch.cuda.is_available())' 2>/dev/null; then
    say "vLLM + CUDA torch already working — skipping GPU extras"
  elif command -v nvidia-smi >/dev/null; then
    CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p' | head -1)"
    say "NVIDIA driver supports CUDA $CUDA"
    if [[ "${CUDA:-0}" -ge 13 ]]; then
      "$VPY" -m pip install -q -e "$REPO[math]"
    elif [[ "${CUDA:-0}" -eq 12 ]]; then
      VLLM_VER="0.31.0"
      "$VPY" -m pip install -q "vllm @ https://github.com/vllm-project/vllm/releases/download/v${VLLM_VER}/vllm-${VLLM_VER}%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
        --extra-index-url https://download.pytorch.org/whl/cu129
      "$VPY" -m pip install -q --force-reinstall --no-deps torchcodec torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu129
      "$VPY" -m pip install -q math-verify datasets pandas pyarrow transformers
    else
      say "WARNING: CUDA ${CUDA:-?} driver too old for vLLM; skipping GPU extras"
    fi
  else
    say "no nvidia-smi: skipping GPU extras (core harness + CPU examples only)"
  fi
fi

# ---- launcher on PATH
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/rsi" <<EOF
#!/usr/bin/env bash
# rsi launcher (installed by $REPO/install.sh)
exec "$VENV/bin/rsi" "\$@"
EOF
chmod +x "$BIN_DIR/rsi"
say "launcher: $BIN_DIR/rsi"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) say "add to your shell rc:  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
"$BIN_DIR/rsi" doctor || true
say "done — try:  cd /anywhere && rsi new my-goal   |   rsi run path/to/goal.yaml"
