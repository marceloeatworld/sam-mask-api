#!/usr/bin/env python3
"""
SAM Mask Service - Based on the working photobooth_pipeline.py code
Creates face masks using SAM model
Accepts image file path or base64 encoded image
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import mediapipe as mp
import os
import time
import logging
from io import BytesIO
import tempfile
import requests
from PIL import Image
from functools import wraps

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key authentication
API_KEY = os.environ.get('SAM_API_KEY', None)

def require_api_key(f):
    """Decorator to require API key for endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip auth if no API key is configured
        if not API_KEY:
            return f(*args, **kwargs)
        
        # Check for API key in header or query param
        provided_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not provided_key:
            return jsonify({'error': 'API key required'}), 401
        
        if provided_key != API_KEY:
            return jsonify({'error': 'Invalid API key'}), 403
        
        return f(*args, **kwargs)
    return decorated_function

class SAMMaskService:
    def __init__(self):
        """Initialize SAM model and MediaPipe once at startup"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_predictor = None
        
        # Initialize MediaPipe for face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for farther faces (up to 5m)
            min_detection_confidence=0.3  # Lower threshold for better multi-face detection
        )
        
        # Initialize SAM model at startup
        self.init_sam('vit_b')
        logger.info("✅ SAM Mask Service ready!")
    
    def init_sam(self, model_type):
        """Initialize SAM model - loads once and stays in memory"""
        # Model stored in Laravel storage/models for security
        # Adjust this path to match your Laravel installation
        checkpoint_path = f"/path/to/your/laravel/storage/models/sam_{model_type}_01ec64.pth"
        
        # For local development (if different):
        # checkpoint_path = f"models/sam_{model_type}_01ec64.pth"
        
        # Download if not exists
        if not os.path.exists(checkpoint_path):
            os.makedirs("models", exist_ok=True)
            import urllib.request
            logger.info(f"📥 Downloading SAM model {model_type}...")
            url = f"https://dl.fbaipublicfiles.com/segment_anything/sam_{model_type}_01ec64.pth"
            urllib.request.urlretrieve(url, checkpoint_path)
            logger.info(f"✅ Model downloaded: {checkpoint_path}")
        else:
            logger.info(f"✅ SAM model already present: {checkpoint_path}")
        
        # Load model
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        logger.info(f"🖥️ Device: {self.device}")
        logger.info(f"✅ SAM model {model_type} initialized")
    
    def get_face_bbox_mediapipe(self, image):
        """Detect face using MediaPipe - same as original code"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        if results.detections:
            h, w = image.shape[:2]
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)
            
            # Slightly expand the box (same as original)
            x = max(0, x - int(box_w * 0.1))
            y = max(0, y - int(box_h * 0.1))
            box_w = min(w - x, int(box_w * 1.2))
            box_h = min(h - y, int(box_h * 1.2))
            
            return x, y, box_w, box_h
        
        return None
    
    def get_all_faces_mediapipe(self, image):
        """Detect ALL faces using MediaPipe - for multi-face mode"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                
                # Slightly expand the box
                x = max(0, x - int(box_w * 0.1))
                y = max(0, y - int(box_h * 0.1))
                box_w = min(w - x, int(box_w * 1.2))
                box_h = min(h - y, int(box_h * 1.2))
                
                faces.append({
                    'x': x,
                    'y': y,
                    'width': box_w,
                    'height': box_h
                })
        
        return faces
    
    def download_image_from_url(self, url):
        """Download image from URL and return as numpy array"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise Exception(f"Failed to download image from URL: {str(e)}")
    
    def create_mask_from_image(self, image, expand_pixels=5, blur_iterations=10, invert=True, mode='first'):
        """
        Create face mask from numpy array image
        Extended with optional multi-face support
        
        Args:
            image: numpy array (BGR)
            expand_pixels: pixels to expand mask (default 5)
            blur_iterations: number of blur iterations (default 10)
            invert: if True, face=black/background=white. if False, face=white/background=black (default True)
            mode: 'first' (single face), 'combined' (all faces in one mask), 'crop' (extract each face as separate image)
        
        Returns:
            For 'first'/'combined': mask array and stats
            For 'crop': list of cropped face images and stats
        """
        h, w = image.shape[:2]
        
        # Configure SAM with the image
        self.sam_predictor.set_image(image)
        
        # Detect faces based on mode
        if mode == 'first':
            # Original behavior - single face
            face_bbox = self.get_face_bbox_mediapipe(image)
            
            if face_bbox is None:
                logger.warning("⚠️ No face detected with MediaPipe")
                return None, {"error": "No face detected"}
            
            x, y, bbox_w, bbox_h = face_bbox
            logger.info(f"✅ Face detected: x={x}, y={y}, w={bbox_w}, h={bbox_h}")
            
            # Points at center and around the face (same as original)
            face_points = np.array([
                [x + bbox_w//2, y + bbox_h//2],      # Center
                [x + bbox_w//2, y + bbox_h//3],      # Top
                [x + bbox_w//3, y + bbox_h//2],      # Left
                [x + 2*bbox_w//3, y + bbox_h//2],    # Right
            ])
            point_labels = np.array([1, 1, 1, 1])  # 1 = foreground
            
            # Bounding box as additional prompt
            input_box = np.array([x, y, x + bbox_w, y + bbox_h])
            
            # Predict the mask
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=face_points,
                point_labels=point_labels,
                box=input_box,
                multimask_output=True
            )
            
            # Select the best mask
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]
            score = scores[best_mask_idx]
            logger.info(f"✅ Mask generated with score: {score:.3f}")
            
            # Convert mask to uint8
            mask = (mask * 255).astype(np.uint8)
            
        elif mode == 'combined':
            # Multi-face mode - create combined mask
            faces = self.get_all_faces_mediapipe(image)
            
            if not faces:
                logger.warning("⚠️ No faces detected with MediaPipe")
                return None, {"error": "No faces detected"}
            
            logger.info(f"✅ Detected {len(faces)} face(s)")
            
            # Create combined mask for all faces
            mask = np.zeros((h, w), dtype=np.uint8)
            scores_list = []
            
            for face in faces:
                x, y, bbox_w, bbox_h = face['x'], face['y'], face['width'], face['height']
                
                # Points for this face
                face_points = np.array([
                    [x + bbox_w//2, y + bbox_h//2],
                    [x + bbox_w//2, y + bbox_h//3],
                    [x + bbox_w//3, y + bbox_h//2],
                    [x + 2*bbox_w//3, y + bbox_h//2],
                ])
                point_labels = np.array([1, 1, 1, 1])
                input_box = np.array([x, y, x + bbox_w, y + bbox_h])
                
                # Predict mask for this face
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=face_points,
                    point_labels=point_labels,
                    box=input_box,
                    multimask_output=True
                )
                
                best_mask_idx = np.argmax(scores)
                face_mask = masks[best_mask_idx]
                face_mask = (face_mask * 255).astype(np.uint8)
                
                # Combine with overall mask
                mask = cv2.bitwise_or(mask, face_mask)
                scores_list.append(scores[best_mask_idx])
            
            score = np.mean(scores_list)
            logger.info(f"✅ Combined mask generated with average score: {score:.3f}")
        
        elif mode == 'crop':
            # Crop mode - extract each face as separate image
            faces = self.get_all_faces_mediapipe(image)
            
            if not faces:
                logger.warning("⚠️ No faces detected with MediaPipe")
                return None, {"error": "No faces detected"}
            
            logger.info(f"✅ Detected {len(faces)} face(s) for cropping")
            
            cropped_faces = []
            
            for i, face in enumerate(faces):
                x, y, bbox_w, bbox_h = face['x'], face['y'], face['width'], face['height']
                
                # Add more padding, especially on top for hair/head
                padding_sides = int(max(bbox_w, bbox_h) * 0.25)  # 25% on sides
                padding_top = int(max(bbox_w, bbox_h) * 0.4)     # 40% on top (for hair)
                padding_bottom = int(max(bbox_w, bbox_h) * 0.25) # 25% on bottom
                
                x_start = max(0, x - padding_sides)
                y_start = max(0, y - padding_top)  # More padding on top
                x_end = min(w, x + bbox_w + padding_sides)
                y_end = min(h, y + bbox_h + padding_bottom)
                
                # Crop the face from original image
                face_crop = image[y_start:y_end, x_start:x_end]
                
                cropped_faces.append({
                    'face_id': i,
                    'image': face_crop,
                    'original_bbox': face,
                    'crop_bbox': {
                        'x': x_start,
                        'y': y_start,
                        'width': x_end - x_start,
                        'height': y_end - y_start
                    }
                })
                
                logger.info(f"   Face {i+1} cropped: {face_crop.shape[1]}x{face_crop.shape[0]} pixels")
            
            # Return cropped faces instead of mask
            stats = {
                "mode": mode,
                "faces_detected": len(faces),
                "faces": faces
            }
            
            return cropped_faces, stats
        
        else:
            return None, {"error": f"Unknown mode: {mode}. Use 'first', 'combined', or 'crop'"}
        
        # Post-process the mask (same as original)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Expand mask by specified pixels
        if expand_pixels > 0:
            expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_pixels, expand_pixels))
            mask = cv2.dilate(mask, expand_kernel, iterations=1)
            logger.info(f"✅ Mask expanded by {expand_pixels} pixels")
        
        # Apply blur specified times
        if blur_iterations > 0:
            for i in range(blur_iterations):
                mask = cv2.GaussianBlur(mask, (21, 21), 0)
            logger.info(f"✅ Blur applied {blur_iterations} times")
        
        # Final thresholding to clean
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Apply inversion based on parameter
        if invert:
            # Invert the mask (face in black, background in white) - Default for Ideogram
            final_mask = cv2.bitwise_not(mask)
            face_value = 0
            bg_value = 255
        else:
            # Normal mask (face in white, background in black)
            final_mask = mask
            face_value = 255
            bg_value = 0
        
        # Calculate statistics
        stats = {
            "face_pixels": int(np.sum(final_mask == face_value)),
            "background_pixels": int(np.sum(final_mask == bg_value)),
            "face_percentage": round(100 * np.sum(final_mask == face_value) / final_mask.size, 2),
            "score": float(score),
            "mode": mode,
            "inverted": invert
        }
        
        # Add face bbox info based on mode
        if mode == 'first':
            stats["face_bbox"] = {
                "x": x,
                "y": y,
                "width": bbox_w,
                "height": bbox_h
            }
        elif mode == 'combined':
            stats["faces_detected"] = len(faces)
            stats["faces"] = faces
        
        return final_mask, stats

# Initialize service
sam_service = SAMMaskService()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'SAM Mask Service (from photobooth_pipeline)',
        'device': sam_service.device,
        'model_loaded': sam_service.sam_predictor is not None
    })

@app.route('/create_mask', methods=['POST'])
@require_api_key
def create_mask():
    """
    Create face mask from image
    
    Accepts:
    - image_url: URL to download image from
    - image_path: path to image file
    - image_base64: base64 encoded image
    - expand_pixels: (optional) default 5
    - blur_iterations: (optional) default 10
    - invert: (optional) default true - if true, face=black/bg=white (Ideogram). if false, face=white/bg=black
    - mode: (optional) default 'first' - 'first' for single face, 'combined' for all faces in one mask
    
    Returns:
    - mask_base64: base64 encoded mask (PNG)
    - stats: mask statistics (includes mode and inverted status)
    """
    try:
        start_time = time.time()
        
        # Get parameters
        if request.is_json:
            data = request.get_json()
        elif request.form:
            data = request.form
        else:
            data = {}
        
        logger.info(f"📝 Received data keys: {list(data.keys())}")
        
        expand_pixels = int(data.get('expand_pixels', 5))
        blur_iterations = int(data.get('blur_iterations', 10))
        mode = data.get('mode', 'first')  # New parameter
        
        # Handle invert parameter (can be string from form or boolean from JSON)
        invert_param = data.get('invert', 'true')
        if isinstance(invert_param, str):
            invert = invert_param.lower() in ['true', '1', 'yes']
        else:
            invert = bool(invert_param)
        
        # Get image
        image = None
        
        # Option 1: Image URL
        if 'image_url' in data:
            image_url = data['image_url']
            logger.info(f"📸 Downloading image from URL: {image_url}")
            try:
                image = sam_service.download_image_from_url(image_url)
                logger.info(f"✅ Image downloaded successfully from URL")
            except Exception as e:
                logger.error(f"❌ Failed to download image: {str(e)}")
                return jsonify({'error': str(e)}), 400
        
        # Option 2: Image path
        elif 'image_path' in data:
            image_path = data['image_path']
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                logger.info(f"📸 Processing image from path: {image_path}")
            else:
                return jsonify({'error': f'File not found: {image_path}'}), 404
        
        # Option 3: Base64 image
        elif 'image_base64' in data:
            image_b64 = data['image_base64']
            # Remove data URL prefix if present
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]
            
            image_data = base64.b64decode(image_b64)
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            logger.info("📸 Processing image from base64")
        
        # Option 4: File upload (multipart)
        elif 'image' in request.files:
            file = request.files['image']
            image_data = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            logger.info(f"📸 Processing uploaded file: {file.filename}")
        
        else:
            return jsonify({'error': 'No image provided. Use image_url, image_path, image_base64, or file upload'}), 400
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Create mask or crop faces
        result, stats = sam_service.create_mask_from_image(
            image, 
            expand_pixels=expand_pixels,
            blur_iterations=blur_iterations,
            invert=invert,
            mode=mode
        )
        
        if result is None:
            return jsonify({'error': 'Failed to process image', 'details': stats}), 500
        
        processing_time = time.time() - start_time
        
        # Handle different modes
        if mode == 'crop':
            # Crop mode - return array of cropped face images
            cropped_faces_b64 = []
            
            for face_data in result:
                # Encode each cropped face as base64 PNG
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
            
            return jsonify({
                'success': True,
                'mode': 'crop',
                'cropped_faces': cropped_faces_b64,
                'stats': stats,
                'processing_time': round(processing_time, 3),
                'parameters': {
                    'mode': mode
                }
            })
        else:
            # Mask modes (first, combined) - return mask
            mask = result
            
            # Encode mask as PNG
            _, buffer = cv2.imencode('.png', mask)
            mask_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'mask_base64': mask_b64,
                'stats': stats,
                'processing_time': round(processing_time, 3),
                'parameters': {
                    'expand_pixels': expand_pixels,
                    'blur_iterations': blur_iterations,
                    'invert': invert,
                    'mode': mode
                }
            })
        
    except Exception as e:
        logger.error(f"Error creating mask: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/create_mask_file', methods=['POST'])
@require_api_key
def create_mask_file():
    """
    Create mask and save to file
    Same as create_mask but saves to disk and returns file path
    """
    try:
        # First create the mask using the main endpoint logic
        response = create_mask()
        
        if response[1] != 200:  # Check status code
            return response
        
        result = response[0].json
        
        # Save mask to file
        mask_b64 = result['mask_base64']
        mask_data = base64.b64decode(mask_b64)
        
        # Create output filename
        output_dir = request.form.get('output_dir', '.')
        output_name = request.form.get('output_name', 'sam_mask.png')
        output_path = os.path.join(output_dir, output_name)
        
        with open(output_path, 'wb') as f:
            f.write(mask_data)
        
        logger.info(f"✅ Mask saved to: {output_path}")
        
        # Add file path to response
        result['mask_file'] = output_path
        del result['mask_base64']  # Remove base64 to save bandwidth
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error saving mask file: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For production use: uvicorn sam_mask_service:app --host 0.0.0.0 --port 8739 --workers 1
    print("🚀 Starting SAM Mask Service on port 8739")
    print("📝 Endpoints:")
    print("   GET  /health           - Check service status")
    print("   POST /create_mask      - Create mask (returns base64)")
    print("   POST /create_mask_file - Create mask (saves to file)")
    app.run(host='0.0.0.0', port=8739, debug=False)