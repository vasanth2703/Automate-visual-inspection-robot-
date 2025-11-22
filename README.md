# 🤖 YOLO + PatchCore Industrial Inspection System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Supabase](https://img.shields.io/badge/Supabase-Enabled-brightgreen.svg)](https://supabase.com/)

Complete 2-stage AI-powered defect inspection pipeline with multi-camera scanning simulation for electronic component quality control.

## 🎯 Features

- **YOLOv8 Detection**: 36 electronic component classes
- **PatchCore Anomaly Detection**: Quality analysis with heatmaps
- **3D Simulation**: ThreeJS robot scanning visualization
- **Real-time Dashboard**: React-based monitoring interface
- **Supabase Integration**: Cloud database and storage
- **Multi-camera Fusion**: Combine detections from multiple views
- **REST API**: FastAPI backend with auto-documentation

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/vasanth2703/Automate-visual-inspection-robot-.git
cd Automate-visual-inspection-robot-

# Install dependencies
py -3.12 -m pip install -r requirements.txt

# Setup (see DEPLOYMENT.md for details)
py -3.12 system_control.py --install

# Run system
py -3.12 system_control.py
```

## 📊 Demo

![Dashboard](https://via.placeholder.com/800x400?text=Dashboard+Screenshot)

## 🏗️ Architecture

```
Dataset Images → Camera Simulator → AI Detection → Supabase → Dashboard
                                    ↓
                            YOLO + PatchCore
```

## Features

- **YOLOv8 Multi-Class Detection**: 36 electronic component classes
- **PatchCore Anomaly Detection**: Per-component quality analysis with heatmaps
- **Multi-Camera Fusion**: Combine detections from multiple viewpoints
- **3D Simulation**: ThreeJS-based AMR scanning visualization
- **FastAPI Backend**: REST API for inspection and training
- **Real-time Dashboard**: Interactive web interface

## Quick Start

### 1. Install Dependencies
```bash
py -3.12 system_control.py --install
```

### 2. Check System Status
```bash
py -3.12 system_control.py --status
```

### 3. Train Models (Quick Mode - ~30 minutes)
```bash
py -3.12 system_control.py --train --quick
```

### 4. Test System
```bash
py -3.12 system_control.py --test
```

### 5. Start Backend Server
```bash
py -3.12 system_control.py --server
```
Server runs at: http://localhost:8000

### 6. Open Frontend
Open in browser: `frontend/combined_dashboard.html`

## System Control Commands

```bash
# Interactive menu
py -3.12 system_control.py

# Command line options
py -3.12 system_control.py --install      # Install dependencies
py -3.12 system_control.py --train        # Train models (full)
py -3.12 system_control.py --train --quick  # Train models (quick)
py -3.12 system_control.py --test         # Test system
py -3.12 system_control.py --demo         # Run demo
py -3.12 system_control.py --server       # Start backend
py -3.12 system_control.py --api-test     # Test API
py -3.12 system_control.py --status       # Show status
```

## Dataset Structure

```
images/
├── armature/
│   ├── armature001.jpg
│   ├── armature002.jpg
│   └── ...
├── microchip/
│   ├── microchip001.jpg
│   └── ...
└── [36 component folders]
```

**Requirements:**
- At least 10 images per component
- JPG/JPEG/PNG format
- Images should show components clearly
- For PatchCore: use "normal" (non-defective) images

## Component Classes (36 Total)

armature, attenuator, Bypass-capacitor, cartridge-fuse, clip-lead, electric-relay, Electrolytic-capacitor, filament, heat-sink, induction-coil, Integrated-micro-circuit, jumper-cable, junction-transistor, LED, light-circuit, limiter-clipper, local-oscillator, memory-chip, microchip, microprocessor, multiplexer, omni-directional-antenna, PNP-transistor, potential-divider, potentiometer, pulse-generator, relay, rheostat, semi-conductor, semiconductor-diode, shunt, solenoid, stabilizer, step-down-transformer, step-up-transformer, transistor

## Training Options

### Quick Training (Testing)
```bash
py -3.12 system_control.py --train --quick
```
- YOLO: 30 epochs (~15-30 min on GPU)
- PatchCore: 100 images per component (~10-20 min)
- Total: ~30-50 minutes

### Full Training (Production)
```bash
py -3.12 system_control.py --train
```
- YOLO: 100 epochs (~50-90 min on GPU)
- PatchCore: 200 images per component (~20-40 min)
- Total: ~70-130 minutes

## API Endpoints

### Inspection
- `POST /scan/frame` - Upload image file for inspection
- `POST /scan/frame/base64` - Send base64 encoded image
- `POST /scan/multi-camera` - Multi-camera fusion

### Scanning
- `POST /scan/start` - Start AMR scan
- `GET /scan/status` - Get scan status
- `GET /scan/results` - Get scan results
- `GET /scan/progress` - Get scan progress

### System
- `GET /ai/status` - Check AI system status
- `GET /config` - Get configuration
- `POST /train/yolo` - Trigger YOLO training
- `POST /train/patchcore` - Trigger PatchCore training

### API Documentation
Interactive docs: http://localhost:8000/docs

## Output Format

### Detection Result
```json
{
  "component": "microchip",
  "bbox": [x1, y1, x2, y2],
  "yolo_conf": 0.95,
  "anomaly_score": 0.87,
  "status": "DEFECTIVE",
  "heatmap": "<base64_encoded_heatmap>"
}
```

### Status Values
- `NORMAL` - Component is good
- `DEFECTIVE` - Anomaly detected
- `UNKNOWN` - No PatchCore model available
- `ERROR` - Processing error

## Model Locations

After training, models are saved to:

```
runs/detect/component_detector/weights/
├── best.pt          # Best YOLO model
├── last.pt          # Last checkpoint
└── best.onnx        # ONNX export

models/patchcore/
├── armature.pkl
├── microchip.pkl
└── [36 component models]
```

## Project Structure

```
.
├── system_control.py          # Master control script
├── run_server.py              # Backend launcher
├── requirements.txt           # Dependencies
│
├── ai/                        # AI module
│   ├── detector.py            # YOLO + PatchCore inspector
│   ├── train_yolo.py          # YOLO training
│   ├── train_patchcore.py     # PatchCore training
│   ├── config.py              # Configuration
│   ├── data_utils.py          # Data preprocessing
│   └── evaluate.py            # Evaluation metrics
│
├── backend/                   # FastAPI backend
│   └── main.py                # API endpoints
│
├── simulation/                # AMR simulation
│   ├── simulator.py           # Scanner simulation
│   ├── camera_sim.py          # Camera simulation
│   └── config.py              # Simulation config
│
├── frontend/                  # Web interfaces
│   ├── combined_dashboard.html
│   ├── visualizer_3d.html
│   └── visualizer_4side.html
│
└── images/                    # Dataset
    └── [component folders]
```

## Configuration

Edit `ai/config.py` to customize:

### YOLO Configuration
```python
epochs: int = 100              # Training epochs
batch_size: int = 16           # Batch size
image_size: int = 640          # Input image size
learning_rate: float = 0.01    # Learning rate
```

### PatchCore Configuration
```python
max_images: int = 200          # Max training images
coreset_ratio: float = 0.1     # Coreset sampling ratio
threshold: float = 0.5         # Anomaly threshold
```

### Inspection Configuration
```python
yolo_conf_threshold: float = 0.25    # YOLO confidence
patchcore_threshold: float = 0.5     # Anomaly threshold
fusion_method: str = 'max'           # Multi-camera fusion
```

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size:
```python
# In ai/config.py
YOLOConfig.batch_size = 8
```

### CUDA Not Available
Training will automatically use CPU (slower but works)

### Import Errors
Reinstall dependencies:
```bash
py -3.12 -m pip install -r requirements.txt --upgrade
```

### No Detections
1. Ensure models are trained
2. Check confidence thresholds
3. Verify image quality

### PatchCore Training Fails
- Need at least 10 images per component
- Ensure images are "normal" (non-defective)
- Check available memory

## Performance Tips

### GPU Acceleration
- Install CUDA-enabled PyTorch
- Check: `py -3.12 -c "import torch; print(torch.cuda.is_available())"`

### Faster Training
- Use quick mode: `--train --quick`
- Reduce epochs: Edit `ai/config.py`
- Use smaller YOLO model (already using nano)

### Better Accuracy
- Increase epochs: `--train` (full mode)
- More training images
- Use larger YOLO model (edit `ai/config.py`: yolov8s or yolov8m)

### Memory Optimization
- Reduce batch size
- Reduce image size
- Limit PatchCore images

## Requirements

- Python 3.12
- GPU recommended (CUDA support)
- 8GB+ RAM
- 10GB+ disk space for models

## Dependencies

Core packages:
- ultralytics (YOLOv8)
- torch, torchvision (PyTorch)
- timm (PatchCore backbone)
- opencv-python (Image processing)
- fastapi, uvicorn (Backend)
- albumentations (Augmentation)
- scikit-learn, scipy (ML utilities)

See `requirements.txt` for complete list.

## Development

### Add New Component Class
1. Add images to `images/<component_name>/`
2. Retrain models: `py -3.12 system_control.py --train`

### Customize Augmentation
Edit `ai/data_utils.py` - `DataAugmentor` class

### Add API Endpoint
Edit `backend/main.py` - Add FastAPI route

### Modify Frontend
Edit HTML files in `frontend/`

## Examples

### Single Image Inspection
```python
from ai.detector import IndustrialInspector
import cv2

inspector = IndustrialInspector(
    yolo_path='runs/detect/component_detector/weights/best.pt',
    patchcore_dir='models/patchcore'
)

image = cv2.imread('test.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = inspector.inspect(image_rgb)

for det in results:
    print(f"{det['component']}: {det['status']}")
```

### Multi-Camera Fusion
```python
all_detections = []
for camera_image in camera_images:
    detections = inspector.inspect(camera_image)
    all_detections.append(detections)

fused = inspector.fuse_multi_camera(all_detections)
```

### API Request
```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/scan/frame', files=files)
    
results = response.json()
print(f"Detections: {results['total_components']}")
print(f"Defective: {results['defective_count']}")
```

## 🎨 React Dashboard

Professional industry-level dashboard with real-time Supabase integration.

### Features:
- **Real-time Updates** - Live scan monitoring with Supabase subscriptions
- **Component Statistics** - Track defect rates per component type
- **Detection Details** - View YOLO confidence and anomaly scores
- **Visual Analytics** - Charts and graphs for quality metrics
- **Scan History** - Browse and analyze past inspections
- **Live Feed** - Real-time camera feed display

### Access Dashboard:
```bash
# 1. Start backend with Supabase
py -3.12 run_server.py

# 2. Open dashboard
frontend/react_dashboard.html
```

### Dashboard Sections:
1. **Stats Overview** - Total scans, components, defects
2. **Control Panel** - Start new scans
3. **Scan List** - Recent inspection history
4. **Live Feed** - Real-time camera view
5. **Detection Details** - Component-level results
6. **Component Stats** - Defect rates by type
7. **Charts** - Visual analytics

## License

This project is for industrial inspection and quality control applications.

## Support

For issues or questions:
1. Check system status: `py -3.12 system_control.py --status`
2. Review error messages
3. Verify dataset structure
4. Check model files exist
5. Test Supabase: `py -3.12 test_supabase.py`

## Acknowledgments

- YOLOv8 by Ultralytics
- PatchCore for anomaly detection
- FastAPI for backend framework
- Three.js for 3D visualization
- React for dashboard UI
- Supabase for real-time database
