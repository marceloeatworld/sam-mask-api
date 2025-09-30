#!/usr/bin/env python3
"""
RQ Worker for processing SAM mask jobs asynchronously
"""
import os
import logging
import torch
from redis import Redis
from rq import Worker, Queue, Connection

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
from sam_mask_service import SAMMaskService
sam_service_instance = SAMMaskService()
logger.info("✅ SAM model pre-loaded and ready in worker")

# Make it available globally for tasks
import tasks
tasks._sam_service = sam_service_instance

# Redis connection
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/1')
redis_conn = Redis.from_url(redis_url)

if __name__ == '__main__':
    logger.info(f"🔴 Connecting to Redis: {redis_url}")

    # Create queues to listen to
    queues = [Queue('sam-masks', connection=redis_conn)]

    logger.info(f"👷 Starting RQ worker for queue: sam-masks")

    with Connection(redis_conn):
        worker = Worker(queues)
        worker.work()
