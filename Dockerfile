FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create necessary directories
RUN mkdir -p models saved_results

# Expose port
EXPOSE 8050

# Run with gunicorn for production
CMD exec gunicorn app:server \
    --bind 0.0.0.0:${PORT:-8050} \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --log-level info
