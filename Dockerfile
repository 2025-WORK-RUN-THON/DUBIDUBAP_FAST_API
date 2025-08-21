FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements-processing.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-processing.txt

# Copy application code
COPY app/ .

# Create necessary directories
RUN mkdir -p data/processed logs

# Copy processed embeddings data
COPY data/processed/processed_embeddings_final.json data/processed/ 2>/dev/null || echo "No embeddings file found"

# Expose port
EXPOSE 8010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8010/api/v1/health || exit 1

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]