#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv3.14/bin/python" ]; then
    PYTHON_BIN=".venv3.14/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  :
else
  echo "Python >= 3.10 is required, got $PYTHON_VERSION from $PYTHON_BIN" >&2
  exit 1
fi

echo "=== Using Python $PYTHON_VERSION: $PYTHON_BIN ==="
echo "=== Installing Python dependencies ==="
PIP_FLAGS=()
if "$PYTHON_BIN" -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages"; then
  PIP_FLAGS+=(--break-system-packages)
fi
"$PYTHON_BIN" -m pip install "${PIP_FLAGS[@]}" -r backend/web/requirements.txt pandas tqdm openpyxl pyyaml

echo "=== Stopping any existing services on :8000 / :8001 ==="
for PORT in 8000 8001; do
  OLD_PID=$(lsof -ti:"$PORT" 2>/dev/null || true)
  if [ -n "$OLD_PID" ]; then
    kill $OLD_PID 2>/dev/null || true
  fi
done
sleep 0.5

FRONTEND_HTTPS_CERT_FILE="${FRONTEND_HTTPS_CERT_FILE:-.certs/frontend-localhost.crt}"
FRONTEND_HTTPS_KEY_FILE="${FRONTEND_HTTPS_KEY_FILE:-.certs/frontend-localhost.key}"
BACKEND_PID=""
FRONTEND_PID=""

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$(pwd)" "$1" ;;
  esac
}

ensure_frontend_https_cert() {
  if [ -f "$FRONTEND_HTTPS_CERT_FILE" ] && [ -f "$FRONTEND_HTTPS_KEY_FILE" ]; then
    return
  fi
  if ! command -v openssl >/dev/null 2>&1; then
    echo "OpenSSL is required to generate a local HTTPS certificate." >&2
    echo "Set FRONTEND_HTTPS_CERT_FILE and FRONTEND_HTTPS_KEY_FILE to existing files, or install openssl." >&2
    exit 1
  fi

  echo "=== Generating local HTTPS certificate for frontend ==="
  mkdir -p "$(dirname "$FRONTEND_HTTPS_CERT_FILE")" "$(dirname "$FRONTEND_HTTPS_KEY_FILE")"
  CERT_CONFIG="$(mktemp)"
  cat >"$CERT_CONFIG" <<'EOF'
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = localhost

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
EOF
  if ! openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 365 \
    -keyout "$FRONTEND_HTTPS_KEY_FILE" \
    -out "$FRONTEND_HTTPS_CERT_FILE" \
    -config "$CERT_CONFIG" >/dev/null 2>&1; then
    rm -f "$CERT_CONFIG"
    echo "Failed to generate frontend HTTPS certificate." >&2
    exit 1
  fi
  rm -f "$CERT_CONFIG"
}

echo "=== Starting backend :8001 ==="
"$PYTHON_BIN" -m uvicorn backend.web.app:app --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

cleanup() {
  if [ -n "${BACKEND_PID:-}" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [ -n "${FRONTEND_PID:-}" ]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}
trap "cleanup; exit" INT TERM EXIT

echo "=== Waiting for backend health ==="
for _ in $(seq 1 60); do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    wait "$BACKEND_PID"
    exit 1
  fi
  if curl -fsS http://127.0.0.1:8001/api/train/health >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -fsS http://127.0.0.1:8001/api/train/health >/dev/null 2>&1; then
  echo "Backend did not become healthy on http://127.0.0.1:8001/api/train/health" >&2
  exit 1
fi

echo "=== Installing frontend dependencies ==="
cd frontend && npm install 2>/dev/null || true

cd ..
ensure_frontend_https_cert
FRONTEND_HTTPS_CERT_FILE="$(abs_path "$FRONTEND_HTTPS_CERT_FILE")"
FRONTEND_HTTPS_KEY_FILE="$(abs_path "$FRONTEND_HTTPS_KEY_FILE")"
export FRONTEND_HTTPS_CERT_FILE FRONTEND_HTTPS_KEY_FILE

echo "=== Starting frontend https://localhost:8000 ==="
cd frontend
npx vite --host 0.0.0.0 &
FRONTEND_PID=$!

echo ""
echo "=== Services started ==="
echo "Frontend: https://localhost:8000"
echo "Backend:  http://127.0.0.1:8001 (internal HTTP upstream)"
echo "Certificate: $FRONTEND_HTTPS_CERT_FILE"
echo ""
echo "Press Ctrl+C to stop"

while true; do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    wait "$BACKEND_PID"
    exit 1
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    wait "$FRONTEND_PID"
    exit 1
  fi
  sleep 1
done
