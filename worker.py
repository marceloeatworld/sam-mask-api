#!/usr/bin/env python3
"""
RQ Worker for processing SAM mask jobs asynchronously
"""
import os
import logging
import torch
from redis import Redis
from rq import Queue, Connection
from rq.worker import SimpleWorker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure PyTorch BEFORE any other imports
num_threads = int(os.environ.get('TORCH_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '8')))
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(2)
logger.info(f"🔧 PyTorch configured with {num_threads} threads")

# Pre-load SAM model ONCE for the worker process
logger.info("🔧 Pre-loading SAM model for worker...")
import sam_mask_service
# Use the module's singleton instance instead of creating a new one
sam_service_instance = sam_mask_service.sam_service
logger.info("✅ SAM model pre-loaded and ready in worker (using module singleton)")

# Make it available globally for tasks
import tasks
tasks._sam_service = sam_service_instance

# Redis connection
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/1')
redis_conn = Redis.from_url(redis_url)

if __name__ == '__main__':
    try:
        logger.info(f"🔴 Connecting to Redis: {redis_url}")

        # Test Redis connection
        redis_conn.ping()
        logger.info("✅ Redis connection successful")

        # Create queues to listen to
        queues = [Queue('sam-masks', connection=redis_conn)]

        logger.info(f"👷 Starting RQ worker for queue: sam-masks")

        # SimpleWorker doesn't need Connection context manager
        worker = SimpleWorker(queues, connection=redis_conn)
        worker.work()

    except Exception as e:
        logger.error(f"❌ Worker failed to start: {str(e)}")
        raise
