# Use Python 3.11 slim image (will automatically get latest 3.11.x)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 80 for Elastic Beanstalk
EXPOSE 80

# Health check for Elastic Beanstalk (use port 80)
HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health

# Run Streamlit app on port 80
CMD ["streamlit", "run", "application.py", "--server.port=80", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]