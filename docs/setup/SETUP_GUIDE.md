# CarbonCheck Field - Detailed Setup Guide

This guide walks you through setting up the CarbonCheck Field app from scratch.

## Part 1: Google Cloud Setup

### 1.1 Create/Configure GCP Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Select project `ml-pipeline-477612` (or create a new one)
3. Enable required APIs:
   - Earth Engine API
   - Vertex AI API
   - Cloud Resource Manager API

### 1.2 Create Service Account

```bash
# Create service account
gcloud iam service-accounts create carbon-check-field \
    --display-name="CarbonCheck Field Mobile App" \
    --project=ml-pipeline-477612

# Grant necessary permissions
gcloud projects add-iam-policy-binding ml-pipeline-477612 \
    --member="serviceAccount:carbon-check-field@ml-pipeline-477612.iam.gserviceaccount.com" \
    --role="roles/earthengine.viewer"

gcloud projects add-iam-policy-binding ml-pipeline-477612 \
    --member="serviceAccount:carbon-check-field@ml-pipeline-477612.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Download JSON key
gcloud iam service-accounts keys create service-account.json \
    --iam-account=carbon-check-field@ml-pipeline-477612.iam.gserviceaccount.com
```

### 1.3 Register Service Account with Earth Engine

```bash
# Authenticate with gcloud
gcloud auth login

# Register service account with Earth Engine
earthengine authenticate --service_account=carbon-check-field@ml-pipeline-477612.iam.gserviceaccount.com
```

Alternatively, visit: https://signup.earthengine.google.com/#!/service_accounts

## Part 2: Google Maps API Setup

### 2.1 Enable Maps SDK

In Google Cloud Console:
1. Navigate to "APIs & Services" → "Library"
2. Search and enable:
   - Maps SDK for Android
   - Maps SDK for iOS

### 2.2 Create API Keys

```bash
# Create Android API key
gcloud alpha services api-keys create \
    --display-name="CarbonCheck Android Maps Key" \
    --api-target=service=maps-android-backend.googleapis.com

# Create iOS API key
gcloud alpha services api-keys create \
    --display-name="CarbonCheck iOS Maps Key" \
    --api-target=service=maps-ios-backend.googleapis.com
```

Or create them manually in Console: "APIs & Services" → "Credentials"

### 2.3 Restrict API Keys (Important for Security!)

**Android Key Restrictions:**
- Application restrictions: Android apps
- Add package name: `com.carboncheck.field`
- Add SHA-1 certificate fingerprint (get from Android Studio or keystore)

**iOS Key Restrictions:**
- Application restrictions: iOS apps
- Add bundle identifier: `com.carboncheck.field`

## Part 3: Flutter Project Setup

### 3.1 Install Flutter

If you don't have Flutter installed:

```bash
# macOS
brew install flutter

# Or download from https://flutter.dev/docs/get-started/install
```

Verify installation:
```bash
flutter doctor
```

### 3.2 Clone and Install Dependencies

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field
flutter pub get
```

### 3.3 Add Service Account Credentials

Copy your service account JSON file:

```bash
cp ~/Downloads/service-account.json assets/service-account.json
```

**CRITICAL:** Ensure this file is gitignored (it already is in `.gitignore`)

### 3.4 Configure Google Maps API Keys

**Android:**

Edit `android/app/src/main/AndroidManifest.xml`:

```xml
<meta-data
    android:name="com.google.android.geo.API_KEY"
    android:value="AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"/>
```

**iOS:**

Edit `ios/Runner/Info.plist`:

```xml
<key>GMSApiKey</key>
<string>AIzaSyYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY</string>
```

## Part 4: Running the App

### 4.1 iOS Setup

```bash
cd ios
pod install
cd ..
```

### 4.2 Run on Device/Simulator

```bash
# List available devices
flutter devices

# Run on iOS
flutter run -d iPhone

# Run on Android
flutter run -d android
```

### 4.3 Build Release Version

**iOS:**
```bash
flutter build ios --release
```
Then open `ios/Runner.xcworkspace` in Xcode to archive and distribute.

**Android:**
```bash
flutter build appbundle --release
```
Output will be at: `build/app/outputs/bundle/release/app-release.aab`

## Part 5: Testing the Backend Services

### 5.1 Test Authentication

Create a test file `test/auth_test.dart`:

```dart
import 'package:carbon_check_field/services/auth_service.dart';

void main() async {
  final authService = AuthService();
  await authService.initialize();
  
  final token = await authService.getAccessToken();
  print('✅ Access token obtained: ${token.substring(0, 30)}...');
}
```

Run: `flutter test test/auth_test.dart`

### 5.2 Test Earth Engine Connection

```bash
# Using curl
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  https://earthengine.googleapis.com/v1/projects/ml-pipeline-477612
```

Should return project metadata.

### 5.3 Test Vertex AI Endpoint

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://us-central1-aiplatform.googleapis.com/v1/projects/ml-pipeline-477612/locations/us-central1/endpoints/7591968360607252480:predict \
  -d '{
    "instances": [[0.7, 0.1, 0.5, 0.9, 0.6, 0.7, 0.8, 0.65, 0.75, 300, -93.0, 41.0, 0.4, 0.2, 0.1, 0.93, 1.07]]
  }'
```

Should return a prediction like: `{"predictions": ["Corn"]}`

## Part 6: Troubleshooting

### Issue: "Flutter SDK not found"

```bash
# Set Flutter SDK path
flutter config --android-sdk /path/to/android-sdk
flutter config --android-studio-dir /path/to/android-studio
```

### Issue: "CocoaPods not installed" (iOS)

```bash
sudo gem install cocoapods
pod setup
```

### Issue: "Google Maps not showing"

1. Check API keys are correct
2. Verify billing is enabled on GCP project
3. Check API restrictions allow your package/bundle ID
4. Look at device logs: `flutter logs`

### Issue: "Authentication failed"

1. Verify service account has correct roles
2. Check service-account.json is in `assets/` folder
3. Ensure service account is registered with Earth Engine
4. Try regenerating service account key

### Issue: "Earth Engine computation timeout"

- Reduce date range (try single month instead of full year)
- Reduce region size
- Increase timeout in HTTP client
- Check Earth Engine quotas

## Part 7: Deployment Checklist

### iOS App Store

- [ ] Update bundle identifier in Xcode
- [ ] Configure code signing
- [ ] Add app icons (1024x1024 and various sizes)
- [ ] Create screenshots for App Store Connect
- [ ] Set up privacy policy URL
- [ ] Configure In-App Purchases (if applicable)
- [ ] Submit for review

### Google Play Store

- [ ] Update applicationId in `build.gradle`
- [ ] Create release signing key
- [ ] Configure signing in `build.gradle`
- [ ] Generate app icons and feature graphic
- [ ] Create store listing (descriptions, screenshots)
- [ ] Set up privacy policy URL
- [ ] Create release in Play Console
- [ ] Upload AAB and submit for review

## Part 8: Security Best Practices

1. **Never commit service account JSON to git**
2. **Restrict API keys** to specific package/bundle IDs
3. **Use App Check** for additional API security (optional)
4. **Implement rate limiting** if adding backend services
5. **Consider moving token generation to backend** for production

## Part 9: Performance Optimization

1. **Enable ProGuard/R8** for Android release builds
2. **Use obfuscation** to protect code
3. **Implement caching** for Earth Engine results
4. **Add offline mode** with cached map tiles
5. **Optimize image assets** (use WebP format)
6. **Profile app performance** with Flutter DevTools

## Part 10: Monitoring & Analytics

Consider adding:
- Firebase Crashlytics (crash reporting)
- Firebase Analytics (user behavior)
- Sentry (error tracking)
- Google Analytics for Firebase

---

**Questions?** Open an issue or contact the development team.

