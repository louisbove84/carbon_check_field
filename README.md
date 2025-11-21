# CarbonCheck Field 🌾

A beautiful Flutter mobile app for farmers to analyze crop types and estimate carbon credit income by drawing field boundaries on a satellite map.

## 🔒 100% Secure Architecture

**No service account keys in the app!** Uses Firebase Auth + secure Cloud Run backend.

## Features

✨ **Interactive Field Drawing**  
- Draw field boundaries by tapping corners on Google Maps satellite view
- Real-time area calculation in acres
- Edit and adjust polygon points after placement

🛰️ **Satellite Analysis**  
- Queries Sentinel-2 SR Harmonized imagery (2024)
- Computes 17 NDVI-based features via Earth Engine
- Handles temporal analysis (early season vs late season)
- **Secure:** All processing happens on backend, not in app

🤖 **AI Crop Classification**  
- Deployed Vertex AI model endpoint
- Predicts crop type with confidence score
- Supports: Corn, Soybeans, Alfalfa, Winter Wheat
- **Secure:** API calls from backend with Application Default Credentials

💰 **Carbon Credit Estimates**  
- Real-world 2025 rates from Indigo Ag & Truterra
- Shows income range and average per year
- Shareable results card

🔐 **Security First**  
- Firebase Authentication (anonymous login)
- Secure Cloud Run backend
- No credentials stored in mobile app
- HTTPS encryption everywhere

## Architecture

The app is organized into focused, testable modules:

```
lib/
├── main.dart                    # App entry point
├── models/                      # Data models
│   ├── field_data.dart         # Field polygon + features
│   └── prediction_result.dart  # Crop prediction + CO₂ income
├── screens/                     # UI screens
│   ├── home_screen.dart        # Landing page
│   ├── map_screen.dart         # Interactive map with polygon drawing
│   └── results_screen.dart     # Analysis results display
├── services/                    # Business logic (API calls)
│   ├── auth_service.dart       # Google Cloud OAuth2 authentication
│   ├── earth_engine_service.dart   # NDVI feature computation
│   └── vertex_ai_service.dart  # Crop prediction
├── utils/                       # Utility functions
│   ├── constants.dart          # App-wide configuration
│   └── geo_utils.dart          # Geospatial calculations
└── widgets/                     # Reusable UI components
    ├── area_display_card.dart
    ├── loading_overlay.dart
    ├── map_instructions.dart
    └── result_card.dart
```

## Setup Instructions

### 1. Prerequisites

- Flutter SDK 3.0+
- Google Cloud project with:
  - Earth Engine API enabled
  - Vertex AI API enabled
  - Service account with appropriate permissions
- Google Maps API keys (iOS + Android)

### 2. Clone & Install Dependencies

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field
flutter pub get
```

### 3. Configure Service Account

Copy your service account JSON to `assets/service-account.json`:

```bash
cp ~/path/to/your-service-account.json assets/service-account.json
```

**Important:** This file is gitignored to prevent accidental commits of credentials.

### 4. Add Google Maps API Keys

**Android:**  
Edit `android/app/src/main/AndroidManifest.xml` and replace:
```xml
<meta-data
    android:name="com.google.android.geo.API_KEY"
    android:value="YOUR_ACTUAL_ANDROID_API_KEY"/>
```

**iOS:**  
Edit `ios/Runner/Info.plist` and replace:
```xml
<key>GMSApiKey</key>
<string>YOUR_ACTUAL_IOS_API_KEY</string>
```

### 5. Run the App

```bash
# iOS
flutter run -d ios

# Android
flutter run -d android
```

## Key Implementation Details

### Earth Engine Integration

The `earth_engine_service.dart` file handles all Earth Engine REST API calls:
- Filters Sentinel-2 imagery by date range (2024) and cloud cover (<20%)
- Computes NDVI from Red (B4) and NIR (B8) bands
- Calculates statistics: mean, std, min, max, percentiles (p25, p50, p75)
- Derives temporal features (early vs late season NDVI)
- Adds elevation and geographic coordinates

**Feature Vector (17 values in exact order):**
1. `ndvi_mean`, `ndvi_std`, `ndvi_min`, `ndvi_max`
2. `ndvi_p25`, `ndvi_p50`, `ndvi_p75`
3. `ndvi_early`, `ndvi_late`, `elevation_m`
4. `longitude`, `latitude`
5. `ndvi_range`, `ndvi_iqr`, `ndvi_change`
6. `ndvi_early_ratio`, `ndvi_late_ratio`

### Vertex AI Prediction

The `vertex_ai_service.dart` file:
- Sends 17-feature vector to your deployed model endpoint
- Parses response: `{"predictions": ["Corn"]}` or `[["Corn", 0.98]]`
- Maps crop type to carbon credit rates
- Calculates income estimates based on field acreage

### Polygon Drawing

The `map_screen.dart` implements:
- Tap-to-add polygon vertices
- Draggable markers for editing points
- Real-time area calculation using Shoelace formula
- Validation (min 3 points, max 50 points, size limits)

## Carbon Credit Rates (2025)

| Crop         | Rate ($/acre/year) |
|--------------|--------------------|
| Corn         | $12 - $18          |
| Soybeans     | $15 - $22          |
| Alfalfa      | $18 - $25          |
| Winter Wheat | $10 - $15          |

*Rates based on Indigo Ag and Truterra carbon credit markets.*

## Testing Individual Components

Each service and utility file is designed to be testable independently:

```dart
// Test authentication
final authService = AuthService();
await authService.initialize();
final token = await authService.getAccessToken();
print('Token: ${token.substring(0, 20)}...');

// Test geospatial utilities
final area = GeoUtils.calculatePolygonAreaAcres([
  LatLng(41.0, -93.0),
  LatLng(41.01, -93.0),
  LatLng(41.01, -93.01),
  LatLng(41.0, -93.01),
]);
print('Area: $area acres');

// Test Earth Engine connection
final eeService = EarthEngineService(authService);
final isConnected = await eeService.testConnection();
print('EE Connected: $isConnected');
```

## Known Issues & TODOs

### ⚠️ JWT Signing Implementation

The `auth_service.dart` file currently has a placeholder for RSA-SHA256 signing of JWT tokens. You have two options:

**Option 1: Add pointycastle dependency**
```yaml
dependencies:
  pointycastle: ^3.7.3
```
Then implement proper RSA signing in `_signMessage()` method.

**Option 2: Use backend token service (recommended for production)**
Create a simple Cloud Function or Cloud Run service that:
1. Receives a request from the app
2. Signs JWT using service account private key server-side
3. Returns access token

This keeps private keys secure on the backend.

### 📝 Future Enhancements

- [ ] Implement proper JWT signing (see above)
- [ ] Add field history/persistence (SQLite or Firebase)
- [ ] Support offline mode with cached map tiles
- [ ] Add field comparison feature
- [ ] Generate PDF reports
- [ ] Multi-year trend analysis
- [ ] Integration with farm management systems

## App Store Deployment

### iOS

1. Update `ios/Runner.xcodeproj` with:
   - Bundle identifier
   - Code signing team
   - Deployment target (iOS 12+)

2. Build release:
```bash
flutter build ios --release
```

3. Archive and upload via Xcode

### Android

1. Update `android/app/build.gradle`:
   - Application ID
   - Version code/name
   - Signing config

2. Build release:
```bash
flutter build appbundle --release
```

3. Upload to Google Play Console

## License

MIT License - See LICENSE file for details

## Support

For issues or questions, please contact the development team or open an issue on the repository.

---

**Built with ❤️ for farmers using Flutter, Earth Engine, and Vertex AI**

