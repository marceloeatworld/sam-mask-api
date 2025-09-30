#!/usr/bin/env python3
"""
RQ Tasks for SAM mask processing
These functions are executed by the RQ worker
"""
import logging
import base64
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Global variable to hold the SAM service (initialized once per worker)
_sam_service = None

def get_sam_service():
    """Lazy initialization of SAM service (only once per worker process)"""
    global _sam_service
    if _sam_service is None:
        logger.info("🔧 Initializing SAM service in worker process...")
        from sam_mask_service import SAMMaskService
        _sam_service = SAMMaskService()
        logger.info("✅ SAM service initialized and cached in worker")
    return _sam_service

def process_mask_job(image_data, expand_pixels=0, blur_iterations=10, invert=True, mode='first'):
    """
    Process a single mask job (executed by RQ worker)

    Args:
        image_data: dict with 'type' and 'data' keys
            - type: 'url', 'base64', 'path'
            - data: the actual image data
        expand_pixels: pixels to expand mask
        blur_iterations: number of blur iterations
        invert: if True, face=black/bg=white
        mode: 'first', 'combined', or 'crop'

    Returns:
        dict with 'success', 'mask_base64' or 'cropped_faces', 'stats'
    """
    try:
        logger.info(f"🔄 Processing job: mode={mode}, type={image_data['type']}")

        # Get SAM service (initialized once per worker)
        sam_service = get_sam_service()

        # Load image based on type
        image = None

        if image_data['type'] == 'url':
            image = sam_service.download_image_from_url(image_data['data'])
            logger.info(f"✅ Image downloaded from URL")

        elif image_data['type'] == 'base64':
            image_b64 = image_data['data']
            # Remove data URL prefix if present
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]

            image_bytes = base64.b64decode(image_b64)
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            logger.info("✅ Image decoded from base64")

        elif image_data['type'] == 'path':
            import os
            if os.path.exists(image_data['data']):
                image = cv2.imread(image_data['data'])
                logger.info(f"✅ Image loaded from path")
            else:
                raise Exception(f"File not found: {image_data['data']}")

        if image is None:
            raise Exception("Failed to decode image")

        # Process the mask
        result, stats = sam_service.create_mask_from_image(
            image,
            expand_pixels=expand_pixels,
            blur_iterations=blur_iterations,
            invert=invert,
            mode=mode
        )

        if result is None:
            raise Exception(f"Failed to process image: {stats}")

        # Handle different modes
        if mode == 'crop':
            # Crop mode - encode cropped faces
            cropped_faces_b64 = []

            for face_data in result:
                _, buffer = cv2.imencode('.png', face_data['image'])
                face_b64 = base64.b64encode(buffer).decode('utf-8')

                cropped_faces_b64.append({
                    'face_id': face_data['face_id'],
                    'image_base64': face_b64,
                    'width': face_data['image'].shape[1],
                    'height': face_data['image'].shape[0],
                    'original_bbox': face_data['original_bbox'],
                    'crop_bbox': face_data['crop_bbox']
                })

            return {
                'success': True,
                'mode': 'crop',
                'cropped_faces': cropped_faces_b64,
                'stats': stats
            }
        else:
            # Mask modes
            mask = result
            _, buffer = cv2.imencode('.png', mask)
            mask_b64 = base64.b64encode(buffer).decode('utf-8')

            return {
                'success': True,
                'mask_base64': mask_b64,
                'stats': stats,
                'mode': mode
            }

    except Exception as e:
        logger.error(f"❌ Job failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
