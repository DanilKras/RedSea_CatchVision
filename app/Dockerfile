FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --no-cache \
    streamlit \
    httpx \
    pandas \
    plotly

WORKDIR /app
COPY app/ app/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD [ \
    "streamlit", "run", "app/dashboard.py", \
    "--server.headless", "true", \
    "--server.address", "0.0.0.0", \
    "--client.toolbarMode", "minimal" \
]