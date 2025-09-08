# SAM Mask API Service

A REST API service that uses Meta's Segment Anything Model (SAM) to automatically generate face masks from images. This service is particularly useful for AI image generation workflows where you need to isolate faces for inpainting or editing.

## Features

- **Automatic Face Detection**: Uses MediaPipe to detect faces in images
- **High-Quality Segmentation**: Leverages SAM (Segment Anything Model) for precise face masking
- **Multiple Input Methods**: Accepts image URLs, base64 encoded images, file paths, or direct file uploads
- **Flexible Output**: Returns masks as base64 PNG or saves to file
- **Multi-Face Support**: Can detect and process multiple faces in a single image
- **Docker Ready**: Containerized for easy deployment
- **API Key Protection**: Optional API key authentication for secure deployments

## Use Cases

- **AI Art Generation**: Create masks for Ideogram, Stable Diffusion, or other AI image generators
- **Photo Editing**: Isolate faces for background removal or replacement
- **Privacy Protection**: Automatically blur or remove faces from images
- **Batch Processing**: Process multiple portraits efficiently via API

## API Endpoints

### Health Check
```bash
GET /health
```
Returns service status and model information.

### Create Mask
```bash
POST /create_mask
```
Generates a face mask and returns it as base64 PNG.

### Create Mask File
```bash
POST /create_mask_file
```
Generates a face mask and saves it to disk.

## Quick Start

### Using Docker Compose

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sam-mask-api.git
cd sam-mask-api
```

2. (Optional) Set API key in `.env`:
```bash
echo "SAM_API_KEY=your-secret-key" > .env
```

3. Start the service:
```bash
docker-compose up -d
```

4. Test the service:
```bash
curl http://localhost:8739/health
```

### Manual Installation

1. Install Python 3.12 and dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
gunicorn --bind 0.0.0.0:8739 --workers 1 --timeout 120 sam_mask_service:app
```

## Usage Examples

### Example 1: Process Image from URL

```bash
curl -X POST http://localhost:8739/create_mask \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/portrait.jpg",
    "expand_pixels": 5,
    "blur_iterations": 10,
    "invert": true
  }'
```

### Example 2: Upload Local Image

```bash
curl -X POST http://localhost:8739/create_mask \
  -F "image=@/path/to/your/photo.jpg" \
  -F "expand_pixels=10" \
  -F "blur_iterations=15"
```

### Example 3: Process Base64 Image (JavaScript)

```javascript
const fs = require('fs');

// Read and encode image
const imageBuffer = fs.readFileSync('photo.jpg');
const base64Image = imageBuffer.toString('base64');

// Send request
const response = await fetch('http://localhost:8739/create_mask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-api-key' // if API key is configured
  },
  body: JSON.stringify({
    image_base64: base64Image,
    expand_pixels: 5,
    blur_iterations: 10,
    invert: true, // true for Ideogram (face=black), false for standard (face=white)
    mode: 'first' // or 'combined' for multiple faces
  })
});

const result = await response.json();
// result.mask_base64 contains the mask as base64 PNG
```

### Example 4: Python Client

```python
import requests
import base64
from PIL import Image
import io

# Using URL
response = requests.post('http://localhost:8739/create_mask', 
    json={
        'image_url': 'https://example.com/portrait.jpg',
        'expand_pixels': 5,
        'blur_iterations': 10,
        'invert': True
    },
    headers={'X-API-Key': 'your-api-key'}  # if configured
)

if response.status_code == 200:
    result = response.json()
    # Decode mask from base64
    mask_data = base64.b64decode(result['mask_base64'])
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_image.save('output_mask.png')
    print(f"Mask saved! Stats: {result['stats']}")
```

### Example 5: Multi-Face Processing

```bash
# Process all faces in one combined mask
curl -X POST http://localhost:8739/create_mask \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/group-photo.jpg",
    "mode": "combined",
    "invert": true
  }'

# Extract each face as separate cropped images
curl -X POST http://localhost:8739/create_mask \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/group-photo.jpg",
    "mode": "crop"
  }'
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_url` | string | - | URL of image to process |
| `image_base64` | string | - | Base64 encoded image |
| `image_path` | string | - | Local file path (server-side) |
| `expand_pixels` | int | 5 | Pixels to expand mask boundary |
| `blur_iterations` | int | 10 | Number of blur passes for smoother edges |
| `invert` | boolean | true | If true: face=black/bg=white (Ideogram style), If false: face=white/bg=black |
| `mode` | string | "first" | Processing mode: "first" (single face), "combined" (all faces), "crop" (extract faces) |

## Response Format

### Standard Mask Response
```json
{
  "success": true,
  "mask_base64": "iVBORw0KGgoAAAANSU...",
  "stats": {
    "face_pixels": 125000,
    "background_pixels": 375000,
    "face_percentage": 25.0,
    "score": 0.995,
    "mode": "first",
    "inverted": true,
    "face_bbox": {
      "x": 100,
      "y": 50,
      "width": 200,
      "height": 250
    }
  },
  "processing_time": 1.234,
  "parameters": {
    "expand_pixels": 5,
    "blur_iterations": 10,
    "invert": true,
    "mode": "first"
  }
}
```

### Crop Mode Response
```json
{
  "success": true,
  "mode": "crop",
  "cropped_faces": [
    {
      "face_id": 0,
      "image_base64": "iVBORw0KGgoAAAANSU...",
      "width": 300,
      "height": 400,
      "original_bbox": {...},
      "crop_bbox": {...}
    }
  ],
  "stats": {
    "mode": "crop",
    "faces_detected": 2,
    "faces": [...]
  }
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SAM_API_KEY` | Optional API key for authentication | None |
| `CUDA_VISIBLE_DEVICES` | GPU device ID (empty for CPU mode) | "" |
| `OMP_NUM_THREADS` | CPU thread count for optimization | 4 |

## System Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: ~3GB for Docker image and SAM model
- **OS**: Linux, macOS, or Windows with Docker

## Model Information

The service uses SAM ViT-B model (375MB) which provides a good balance between performance and accuracy. The model is automatically downloaded on first run and cached in the `models/` directory.

## Performance

- **CPU Mode**: 2-5 seconds per image (depending on CPU)
- **GPU Mode**: 0.5-1 second per image (with CUDA)
- **Memory Usage**: ~2-3GB when model is loaded

## Security

- Optional API key authentication via `X-API-Key` header
- Input validation and error handling
- Runs with limited container privileges
- No data persistence (stateless service)

## Troubleshooting

### Container won't start
Check logs: `docker-compose logs sam-mask-service`

### Out of memory errors
Increase Docker memory allocation or reduce image size before processing

### No face detected
- Ensure image contains clear, visible faces
- Try adjusting MediaPipe confidence: modify `min_detection_confidence` in code
- Check image orientation (should be upright)

### Slow performance
- Use GPU if available (modify docker-compose.yaml for GPU support)
- Reduce image size before processing
- Ensure sufficient CPU/RAM allocated to Docker

## License

This project uses:
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [MediaPipe](https://github.com/google/mediapipe) by Google
- Flask, OpenCV, and other open-source libraries

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Support

For issues or questions, please open an issue on GitHub or contact the maintainers.