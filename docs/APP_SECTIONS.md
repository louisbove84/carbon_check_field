## App Sections

### Flutter App (`lib/`)

- `main.dart` - Entry point, Firebase init
- `screens/` - Home, Map, Results, Crop Zones
- `models/` - FieldData, PredictionResult, CropZone
- `services/` - Backend + Firebase integration
- `widgets/` - Reusable UI components

### Backend (`backend/`)

- `app.py` - FastAPI app (Earth Engine + Vertex AI)
- `Dockerfile` - Cloud Run container
- `requirements.txt` - Python dependencies

### ML Pipeline (`ml_pipeline/`)

- `orchestrator/` - Run pipeline steps and deployment logic
- `trainer/` - Vertex training code
- `shared/` - Shared feature extraction
