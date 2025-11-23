# Quick Start Guide 🚀

Get CarbonCheck Field running in 5 minutes!

## Prerequisites

- Flutter SDK installed (`flutter --version`)
- Google Maps API keys (Android + iOS)
- Service account JSON file from Google Cloud

## 1. Install Dependencies

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field
flutter pub get
```

## 2. Add Credentials

```bash
# Copy service account JSON
cp ~/path/to/your-service-account.json assets/service-account.json
```

## 3. Configure API Keys

**Android:** Edit `android/app/src/main/AndroidManifest.xml`
- Replace `YOUR_GOOGLE_MAPS_API_KEY_HERE` with your Android API key

**iOS:** Edit `ios/Runner/Info.plist`
- Replace `YOUR_GOOGLE_MAPS_API_KEY_HERE` with your iOS API key

## 4. Run the App

```bash
# iOS
flutter run -d ios

# Android
flutter run -d android
```

## 5. Test It Out!

1. Tap "Draw Field on Map"
2. Tap 4 corners of a test field
3. Tap "Analyze Field"
4. See crop prediction and CO₂ income estimate!

## Troubleshooting

**App crashes on startup?**
- Check `assets/service-account.json` exists and is valid JSON

**Map is blank?**
- Verify API keys are correct
- Enable billing on your GCP project
- Check Maps SDK is enabled

**Authentication errors?**
- Ensure service account has Earth Engine + Vertex AI permissions
- Register service account with Earth Engine

---

For detailed setup, see `SETUP_GUIDE.md`

