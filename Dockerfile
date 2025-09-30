FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY sam_mask_service.py .
COPY tasks.py .
COPY worker.py .

# Create models directory
RUN mkdir -p models

# Download SAM model (optional - comment out if mounting volume)
# RUN wget -O models/sam_vit_b_01ec64.pth \
#     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Expose port
EXPOSE 8739

# Gunicorn for Flask with optimized settings
# IMPORTANT: Only 1 worker to keep single model instance in memory
# Increased threads for better concurrency within single worker
CMD ["gunicorn", "--bind", "0.0.0.0:8739", "--workers", "1", "--threads", "8", "--timeout", "120", "--worker-class", "gthread", "sam_mask_service:app"]