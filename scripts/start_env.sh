#!/usr/bin/env bash
set -euo pipefail

# start_env.sh - helper to prepare the development environment
# Usage: ./scripts/start_env.sh [--no-docker] [--no-ryu-copy] [--no-venv]

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

NO_DOCKER=0
NO_RYU_COPY=0
NO_VENV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-docker) NO_DOCKER=1; shift ;;
    --no-ryu-copy) NO_RYU_COPY=1; shift ;;
    --no-venv) NO_VENV=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--no-docker] [--no-ryu-copy] [--no-venv]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "[start_env] Project root: $ROOT_DIR"

if [[ $NO_VENV -eq 0 ]]; then
  if [[ ! -d "$ROOT_DIR/venv" ]]; then
    echo "[start_env] Creating virtual environment..."
    python3 -m venv "$ROOT_DIR/venv"
  fi
  echo "[start_env] Activating virtual environment and installing requirements..."
  # shellcheck disable=SC1090
  source "$ROOT_DIR/venv/bin/activate"
  if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
    pip install --upgrade pip
    pip install -r "$ROOT_DIR/requirements.txt"
  else
    echo "[start_env] No requirements.txt found, skipping pip install"
  fi
fi

if [[ $NO_DOCKER -eq 0 ]]; then
  if command -v docker >/dev/null 2>&1 && command -v docker-compose >/dev/null 2>&1; then
    echo "[start_env] Starting docker compose services in background..."
    docker compose up -d
    sleep 2
  else
    echo "[start_env] Docker or Docker Compose not found; skipping docker startup"
  fi

  if [[ $NO_RYU_COPY -eq 0 ]]; then
    if docker ps --format '{{.Names}}' | grep -q '^ryu$'; then
      if [[ -f "$ROOT_DIR/ryu/apps/slice_switch_13.py" ]]; then
        echo "[start_env] Copying ryu/apps/slice_switch_13.py into container 'ryu'..."
        docker cp "$ROOT_DIR/ryu/apps/slice_switch_13.py" ryu:/apps/slice_switch_13.py || true
        echo "[start_env] Restarting 'ryu' container to pick up changes..."
        docker restart ryu || true
      else
        echo "[start_env] ryu/apps/slice_switch_13.py not found locally; skipping copy"
      fi
    else
      echo "[start_env] Container named 'ryu' not running; skipping ryu copy"
    fi
  fi
fi

echo "[start_env] Done. You can now run training or start Mininet as described in README.md"
