# CarbonCheck Field 🌾

A Flutter mobile app that helps farmers analyze crop types and estimate carbon credit income using satellite imagery and AI.

## ✨ Key Features

- **Interactive Field Drawing** - Draw field boundaries on satellite maps with real-time acreage calculation
- **Multi-Zone Analysis** - Detects multiple crop zones within large fields (up to 2,000 acres)
- **AI Crop Classification** - Uses Google Vertex AI to predict crop types (Corn, Soybeans, Alfalfa, Winter Wheat)
- **Satellite NDVI Analysis** - Processes Sentinel-2 imagery via Google Earth Engine
- **Carbon Credit Estimates** - Real-world 2025 rates ($10-$25/acre based on crop type)
- **Automated ML Pipeline** - Monthly retraining with fresh satellite data
- **Secure Architecture** - No service account keys in app, all GCP calls through Cloud Run backend

## 🚀 Quick Start

### Prerequisites

- Flutter SDK 3.0+
- Google Cloud project with Earth Engine & Vertex AI enabled
- Firebase project configured
- Google Maps API key

### Setup

1. **Clone and install dependencies**
```bash
git clone <your-repo-url>
cd carbon_check_field
flutter pub get
```

2. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Setup Firebase**
- Download `google-services.json` → `android/app/`
- Download `GoogleService-Info.plist` → `ios/Runner/`
- Enable anonymous authentication in Firebase Console

4. **Deploy backend**
```bash
cd backend
./setup_and_deploy.sh
```

5. **Update backend URL in `.env`**
```
BACKEND_URL=https://your-service-url.run.app
```

6. **Run the app**
```bash
# Web (Chrome)
flutter run -d chrome --web-port=8080

# Android
flutter run -d android

# iOS
flutter run -d ios
```

## 📁 Project Structure

```
lib/                        # Flutter app code
├── main.dart              # Entry point with Firebase init
├── models/                # Data models (FieldData, PredictionResult, CropZone)
├── screens/               # UI screens (Home, Map, Results, CropZones)
├── services/              # Backend & Firebase services
├── utils/                 # Constants and utilities
└── widgets/               # Reusable components

backend/                   # Python FastAPI backend
├── app.py                # Main API with Earth Engine + Vertex AI integration
├── Dockerfile            # Container for Cloud Run
└── requirements.txt      # Python dependencies

ml_pipeline/              # Automated ML training pipeline
├── auto_retrain_model.py # Retrains model monthly
├── monthly_data_collection.py  # Collects training data
└── NDVI_info             # Earth Engine script for data generation
```

## 🤖 Automated ML Pipeline

The project includes a fully automated ML pipeline that:

- ✅ Collects fresh training data from Earth Engine every month
- ✅ Retrains the crop classification model with all historical data
- ✅ Deploys the updated model to Vertex AI automatically

### Deploy the Pipeline

```bash
cd ml_pipeline
./deploy_pipeline.sh
```

The pipeline runs automatically:
- **1st of each month:** Collect 400 new training samples
- **5th of each month:** Retrain model and deploy to production

## 🏗️ Architecture

```
Flutter App (Mobile/Web)
    ↓
Firebase Authentication (anonymous)
    ↓
Cloud Run Backend (FastAPI)
    ↓
├── Google Earth Engine (NDVI features)
├── Vertex AI (crop prediction)
└── USDA CDL (ground truth validation)
```

### Key Technical Features

- **Grid-based classification** - Fields >10 acres split into adaptive grids (max 25 cells)
- **Spatial grouping** - Adjacent cells with same crop merged into zones
- **Polygon validation** - Automatic repair of self-intersecting geometries
- **Optimized cell sizing** - [50, 100, 200, 300, 500] meter grids based on field size

## 🧪 Testing

### Test Endpoint Locally

```bash
cd backend
uvicorn app:app --reload
```

### Run Flutter App Locally

```bash
# Web (easiest for testing)
flutter run -d chrome --web-port=8080

# Android device
flutter run -d <device-id>
```

### Test Deployed Backend

```bash
curl https://your-backend-url.run.app/health
```

## 🚢 Deployment

### Deploy Backend

```bash
cd backend
gcloud run deploy carboncheck-field-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

### Build Android Release

```bash
flutter build apk --release
# Install on device
flutter install -d <device-id>

# Or build app bundle for Google Play
flutter build appbundle --release
```

### Build iOS Release

```bash
flutter build ios --release
# Then open ios/Runner.xcworkspace in Xcode
```

## 🔐 Security

- ✅ **No GCP credentials in app** - All API calls proxied through Cloud Run
- ✅ **Application Default Credentials** - Backend uses Google-managed auth
- ✅ **Firebase token verification** - All requests authenticated
- ✅ **Environment variables** - API keys in `.env` (gitignored)
- ✅ **HTTPS everywhere** - All traffic encrypted

## 💰 Carbon Credit Rates (2025)

| Crop         | $/acre/year |
|--------------|-------------|
| Corn         | $12 - $18   |
| Soybeans     | $15 - $22   |
| Alfalfa      | $18 - $25   |
| Winter Wheat | $10 - $15   |

*Based on Indigo Ag and Truterra markets*

## 🐛 Troubleshooting

### "Backend timeout or 500 error"
- Check backend logs: `gcloud run logs tail carboncheck-field-api --region us-central1`
- Verify Earth Engine authentication is configured
- Check polygon is not self-intersecting (app will auto-fix simple cases)

### "Map is blank"
- Verify Google Maps API key in `.env`
- Enable billing on GCP project
- Enable Maps SDK for Android/iOS

### "Firebase initialization failed"
- Ensure `google-services.json` and `GoogleService-Info.plist` are present
- Enable anonymous auth in Firebase Console

### Android build errors
```bash
cd android
./gradlew clean
flutter clean
flutter pub get
```

## 📊 Monitoring

### Cloud Run Logs
```bash
gcloud run logs tail carboncheck-field-api --region us-central1 --format json
```

### API Usage
- Google Cloud Console → APIs & Services → Dashboard
- Monitor Earth Engine, Vertex AI, and Maps quotas

## 📜 License

MIT License

---

**Built with Flutter, Google Earth Engine, and Vertex AI**

**Default Map Center:** Northeast Wisconsin (44.409438290384166, -88.4304410977501)
