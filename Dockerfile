# Multi-stage build for optimized image size
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-distutils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Production stage
FROM base as production

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 cache purge

# Copy only necessary application files
COPY app/ ./app/
COPY static/ ./static/
COPY template/ ./template/
COPY models/ ./models/
COPY run.py .

# Create necessary directories
RUN mkdir -p uploads logs

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="" \
    FLASK_ENV=production

# Non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:9001', timeout=5)" || exit 1

# Expose port
EXPOSE 9001

# Run application
CMD ["python3", "run.py", "--device", "cpu"]