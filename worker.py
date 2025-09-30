#!/usr/bin/env python3
"""
RQ Worker for processing SAM mask jobs asynchronously
"""
import os
import logging
from redis import Redis
from rq import Worker, Queue, Connection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
