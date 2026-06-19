FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer so it caches across code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and pre-trained artifacts
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8080"]
