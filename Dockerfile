FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip install uv

COPY pyproject.toml uv.lock ./
COPY mcp_modules/ ./mcp_modules/
COPY server.py ./
COPY README.md ./

RUN uv sync --frozen

RUN mkdir -p /tmp/ast_cache

EXPOSE 8000

ENV PYTHONPATH=/app
ENV MCP_CACHE_DIR=/tmp/ast_cache

CMD ["uv", "run", "server.py"]
