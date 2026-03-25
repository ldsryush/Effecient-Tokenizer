FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app/ ./app/
COPY scripts/ ./scripts/

# Environment defaults (override via docker run -e or docker-compose.yml)
ENV STORE_BACKEND=memory \
    DISPATCH_DRY_RUN=false \
    LLM_MAX_RETRIES=3 \
    LLM_TIMEOUT_S=60 \
    LLM_FALLBACK_MODEL=gpt-4o-mini

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
