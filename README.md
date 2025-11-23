# CarbonCheck Field 🌾

A secure Flutter mobile app that helps farmers analyze crop types and estimate carbon credit income by drawing field boundaries on satellite maps.

## ✨ Features

- **Interactive Field Drawing** - Draw field boundaries by tapping corners on Google Maps satellite view with real-time acreage calculation
- **Satellite Analysis** - Queries Sentinel-2 imagery and computes 17 NDVI-based features via Earth Engine
- **AI Crop Classification** - Predicts crop type (Corn, Soybeans, Alfalfa, Winter Wheat) using Vertex AI
- **CDL Ground Truth** - Shows USDA Cropland Data Layer results alongside model predictions
- **Address Search** - Quick field location using address lookup
- **Carbon Credit Estimates** - Real-world 2025 rates from Indigo Ag & Truterra ($10-$25/acre)
- **Shareable Results** - Export field analysis for sharing

## 🔒 100% Secure Architecture

**No service account keys in the app!** 

- Firebase Authentication (anonymous login)
- Secure Cloud Run backend with Application Default Credentials
- No GCP credentials stored in mobile app
- HTTPS encryption everywhere

```
Flutter App → Firebase Auth → Cloud Run Backend → Earth Engine + Vertex AI
(no keys!)                   (secure credentials)
```

---

## 🚀 Quick Start

### Prerequisites

- Flutter SDK 3.0+
- Google Cloud project with Earth Engine & Vertex AI enabled
- Firebase project
- Google Maps API key

### 1. Clone & Install

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field
flutter pub get
```

### 2. Setup Firebase

1. Download Firebase config files:
   - `google-services.json` → `android/app/`
   - `GoogleService-Info.plist` → `ios/Runner/`

2. Enable anonymous authentication in Firebase Console

### 3. Add Google Maps API Key

Edit `lib/utils/constants.dart`:
```dart
static const String googleMapsApiKey = 'YOUR_GOOGLE_MAPS_API_KEY';
```

**Important:** The API key is stored locally and .gitignored. See deployment section for production builds.

### 4. Deploy Backend

```bash
cd backend
./setup_and_deploy.sh
```

This deploys the secure Cloud Run backend that handles all GCP API calls.

### 5. Update Backend URL

After deployment, update `lib/utils/constants.dart`:
```dart
static const String backendUrl = 'https://your-service-url.run.app';
```

### 6. Run the App

```bash
# Web (Chrome)
flutter run -d chrome

# iOS
flutter run -d ios

# Android
flutter run -d android
```

---

## 📁 Project Structure

```
lib/
├── main.dart                      # App entry point + Firebase init
├── models/
│   ├── field_data.dart           # Field polygon + geographic data
│   └── prediction_result.dart    # Crop prediction + CO₂ income
├── screens/
│   ├── home_screen.dart          # Landing page
│   ├── map_screen.dart           # Interactive map with drawing
│   └── results_screen.dart       # Analysis results + CDL data
├── services/
│   ├── firebase_service.dart     # Authentication
│   └── backend_service.dart      # Secure API calls to Cloud Run
├── utils/
│   ├── constants.dart            # Configuration (API keys, URLs)
│   └── geo_utils.dart            # Geospatial calculations
└── widgets/                       # Reusable UI components

backend/
├── app.py                         # FastAPI server
├── Dockerfile                     # Container definition
├── requirements.txt               # Python dependencies
└── setup_and_deploy.sh           # Deployment script

ml_pipeline/                        # 🤖 Automated ML Training Pipeline
├── monthly_data_collection.py     # Collects training data from Earth Engine
├── auto_retrain_model.py          # Retrains & deploys model in Vertex AI
├── requirements.txt               # Python dependencies
├── pipeline_config.yaml           # Configuration (crops, schedule, etc.)
├── deploy_pipeline.sh             # One-click deployment
├── README.md                      # Full pipeline documentation
├── QUICK_REFERENCE.md             # Quick commands & URLs
└── local_testing_guide.md         # Local testing instructions
```

---

## 🤖 Automated ML Pipeline

**NEW!** This project includes a fully automated machine learning pipeline that:

- ✅ **Collects** fresh training data from Earth Engine every month
- ✅ **Retrains** the crop classification model with all historical data
- ✅ **Deploys** the updated model to production automatically
- ✅ **Zero manual work** required after setup!

### Quick Setup

```bash
cd ml_pipeline
./deploy_pipeline.sh
```

**That's it!** The pipeline now runs automatically:
- **1st of each month:** Collect 400 new training samples (100 per crop)
- **5th of each month:** Retrain model and deploy to Vertex AI endpoint

### Manual Testing

```bash
# Trigger data collection manually
curl -X POST https://us-central1-ml-pipeline-477612.cloudfunctions.net/monthly-data-collection

# Trigger model retraining manually
curl -X POST https://us-central1-ml-pipeline-477612.cloudfunctions.net/auto-retrain-model
```

### Learn More

- 📖 **Full Documentation:** [`ml_pipeline/README.md`](ml_pipeline/README.md)
- ⚡ **Quick Commands:** [`ml_pipeline/QUICK_REFERENCE.md`](ml_pipeline/QUICK_REFERENCE.md)
- 🧪 **Local Testing:** [`ml_pipeline/local_testing_guide.md`](ml_pipeline/local_testing_guide.md)

---

## 🔧 Architecture

### High-Level Flow

```
1. User draws field polygon on map
2. User taps "Analyze Field"
3. Flutter app:
   - Signs in anonymously with Firebase
   - Gets Firebase ID token
   - Sends polygon + token to Cloud Run backend
4. Cloud Run backend:
   - Verifies Firebase token
   - Calls Earth Engine (17 NDVI features)
   - Calls Vertex AI (crop prediction)
   - Queries USDA CDL (ground truth)
   - Returns JSON with results
5. Flutter displays:
   - Crop type + confidence
   - CDL ground truth + agreement indicator
   - Field area (acres)
   - CO₂ income range
```

### Key Technical Decisions

- **Flutter** - Single codebase for iOS + Android
- **Cloud Run Backend** - Keeps all GCP credentials secure
- **Firebase Auth** - No user registration required (anonymous)
- **Direct REST APIs** - No Earth Engine SDK needed
- **17 NDVI Features** - Temporal + spatial statistics from Sentinel-2

---

## 🧪 Testing

### Test on Chrome (Easiest)

```bash
flutter run -d chrome
```

### Test on iOS Simulator

```bash
open -a Simulator
flutter run -d ios
```

### Test on Android Device

1. Enable USB debugging on your phone
2. Connect via USB
3. Run:
```bash
adb devices
flutter run -d android
```

### Verify Backend

```bash
# Health check
curl https://your-backend-url.run.app/health
```

---

## 🚢 Deployment

### Android APK

```bash
# Debug build
flutter build apk --debug
adb install -r build/app/outputs/flutter-apk/app-debug.apk

# Release build
flutter build appbundle --release
```

Upload `build/app/outputs/bundle/release/app-release.aab` to Google Play Console.

### iOS App Store

```bash
flutter build ios --release
```

Then open `ios/Runner.xcworkspace` in Xcode to archive and upload.

### App Icons

Custom app icon is configured in:
- `android/app/src/main/res/mipmap-*/ic_launcher.png`
- `ios/Runner/Assets.xcassets/AppIcon.appiconset/`

---

## 🔐 Security Details

### What's Secure

✅ **No GCP credentials in app** - All API calls go through Cloud Run backend  
✅ **Application Default Credentials** - Backend uses Google-managed auth  
✅ **Firebase token verification** - Backend validates every request  
✅ **API key restrictions** - Google Maps key restricted by bundle ID  
✅ **HTTPS everywhere** - All network traffic encrypted  

### API Keys Management

**Development:**
- Keys stored locally in `lib/utils/constants.dart`
- File is .gitignored to prevent commits

**Production:**
- For app store builds, keys must be added before compiling
- Consider using Flutter environment variables or Firebase Remote Config
- See `DEPLOYMENT.md` for full strategy

### Firebase Setup

1. Firebase Console → Authentication → Enable "Anonymous"
2. Add iOS app (bundle ID: `com.carboncheck.field`)
3. Add Android app (package name: `com.carboncheck.field`)
4. Download config files and place in respective directories

---

## 💰 Carbon Credit Rates (2025)

| Crop         | Rate ($/acre/year) |
|--------------|--------------------|
| Corn         | $12 - $18          |
| Soybeans     | $15 - $22          |
| Alfalfa      | $18 - $25          |
| Winter Wheat | $10 - $15          |

*Based on Indigo Ag and Truterra carbon credit markets*

---

## 🐛 Troubleshooting

### "Firebase initialization failed"
- Ensure `google-services.json` and `GoogleService-Info.plist` are in correct directories
- Run `flutterfire configure` to regenerate configs

### "Map is blank"
- Check Google Maps API key in `constants.dart`
- Enable billing on your GCP project
- Verify Maps SDK for iOS/Android are enabled

### "Analysis failed"
- Check backend URL in `constants.dart`
- Verify Cloud Run service is deployed
- Check Cloud Run logs: `gcloud run logs read carboncheck-field-api --region us-central1`

### "Device not found" (Android)
- Install adb: `brew install --cask android-platform-tools`
- Reconnect phone and allow USB debugging
- Run: `adb devices` to verify connection

### Gradle errors (Android)
- Run: `cd android && ./gradlew clean`
- Update Gradle wrapper if needed
- Accept licenses: `flutter doctor --android-licenses`

---

## 📊 Monitoring

### Cloud Run Logs
```bash
gcloud run logs tail carboncheck-field-api --region us-central1
```

### Firebase Users
- Firebase Console → Authentication → Users
- Monitor anonymous user count

### API Usage
- Google Cloud Console → APIs & Services → Dashboard
- Monitor Earth Engine and Vertex AI quotas

---

## 💾 Data Flow

### Earth Engine Features (17 values)

1. NDVI statistics: mean, std, min, max
2. Percentiles: p25, p50, p75
3. Temporal: early_season, late_season
4. Location: latitude, longitude, elevation
5. Derived: range, IQR, change, early_ratio, late_ratio

### Backend Response Format

```json
{
  "crop": "Corn",
  "confidence": 0.98,
  "area_acres": 47.2,
  "co2_income_low": 1240,
  "co2_income_high": 1880,
  "cdl_crop": "Corn",
  "cdl_confidence": 0.95,
  "cdl_year": 2024,
  "matches_cdl": true
}
```

---

## 🎯 Future Enhancements

- [ ] Field history storage (SQLite/Firebase)
- [ ] Multi-year trend analysis
- [ ] Offline mode with cached tiles
- [ ] PDF report generation
- [ ] User accounts (email/social login)
- [ ] Team/organization support
- [ ] Carbon marketplace integration

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🙋 Support

For issues or questions:
1. Check troubleshooting section above
2. Review Cloud Run logs for backend errors
3. Check Firebase Auth is properly configured
4. Verify all API keys are correct

---

**Built with ❤️ for farmers using Flutter, Earth Engine, and Vertex AI**

**Default Map Center:** 44.409438290384166, -88.4304410977501 (Northeast Wisconsin)
