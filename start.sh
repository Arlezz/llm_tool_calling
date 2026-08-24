#!/usr/bin/env bash
# Starts mlx_lm.server natively (if it isn't already running) and then brings up the
# Docker-compose API on top of it. The model needs Metal, so it can never run inside the
# container - see README.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL="${MLX_MODEL:-mlx-community/Qwen3-14B-4bit}"
PORT="${MLX_SERVER_PORT:-8080}"
LOG_FILE="mlx_server.log"

if curl -s -o /dev/null "http://127.0.0.1:${PORT}/health"; then
    echo "mlx_lm.server ya está corriendo en el puerto ${PORT}."
else
    echo "Levantando mlx_lm.server nativo (modelo: ${MODEL}, puerto: ${PORT})..."
    uv sync --extra server --quiet
    nohup uv run mlx_lm.server --model "${MODEL}" --host 0.0.0.0 --port "${PORT}" > "${LOG_FILE}" 2>&1 &
    disown

    echo -n "Esperando a que cargue el modelo..."
    until curl -s -o /dev/null "http://127.0.0.1:${PORT}/health"; do
        echo -n "."
        sleep 1
    done
    echo " listo."
fi

echo "Levantando el contenedor de la API..."
docker compose up -d --build

echo
echo "El contenedor se detuvo. mlx_lm.server sigue corriendo en segundo plano (puerto ${PORT})."
echo "Para pararlo: pkill -f 'mlx_lm.server.*--port ${PORT}'"
