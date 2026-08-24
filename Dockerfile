FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first (cached layer) - the "server" extra (mlx-lm) is intentionally left
# out: it's Apple Silicon/Metal-only and this image never touches the model directly, it talks to
# a natively-running mlx_lm.server over HTTP (see llm_client.py).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app/ ./app/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
