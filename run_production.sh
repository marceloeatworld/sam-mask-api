#!/bin/bash

# Production startup script for SAM service with Uvicorn

echo "🚀 Starting SAM Mask Service with Uvicorn..."

# Optimal Uvicorn configuration for production
uvicorn sam_mask_service:app \
    --host 0.0.0.0 \
    --port 8739 \
    --workers 1 \
    --loop uvloop \
    --access-log \
    --log-level info \
    --timeout-keep-alive 5 \
    --limit-concurrency 100

# Options explained:
# --workers 1: CRITICAL - Only 1 worker to avoid multiple model instances
# --loop uvloop: Fastest event loop (2-4x faster than default)
# --access-log: Log all requests
# --timeout-keep-alive 5: Close idle connections after 5s
# --limit-concurrency 100: Max 100 concurrent connections