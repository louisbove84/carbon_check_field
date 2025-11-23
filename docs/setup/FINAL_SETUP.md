# Final Setup Steps 🎯

You've completed Firebase setup! Here's what's left:

---

## Step 3: Update Backend URL in Flutter App

### 3.1 Get your Cloud Run URL

If you already deployed, get the URL:

```bash
gcloud run services describe carboncheck-field-api \
  --region us-central1 \
  --format 'value(status.url)'
```

**OR** if you haven't deployed yet:

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/backend
./setup_and_deploy.sh
```

Copy the URL that looks like:
```
https://carboncheck-field-api-abc123xyz-uc.a.run.app
```

### 3.2 Update Flutter app

**File:** `lib/services/backend_service.dart`

**Line 17:** Change from:
```dart
static const String backendUrl = 'https://carboncheck-field-api-XXXXXXXX-uc.a.run.app';
```

**To:**
```dart
static const String backendUrl = 'https://YOUR-ACTUAL-URL.run.app';
```

---

## Step 4: Verify Firebase Configuration Files

Make sure you have these files from Firebase Console:

### iOS:
```bash
ls ios/Runner/GoogleService-Info.plist
```

Should show the file exists (not the .template version)

### Android:
```bash
ls android/app/google-services.json
```

Should show the file exists (not the .template version)

### If files are missing:

1. Go to Firebase Console: https://console.firebase.google.com
2. Select project: `ml-pipeline-477612`
3. iOS app → Download `GoogleService-Info.plist` → Save to `ios/Runner/`
4. Android app → Download `google-services.json` → Save to `android/app/`

---

## Step 5: Update Android build.gradle (Required for Firebase)

Add the Google Services plugin to Android:

**File:** `android/build.gradle`

Add this at the top (after buildscript):
```gradle
buildscript {
    dependencies {
        // Add this line
        classpath 'com.google.gms:google-services:4.4.0'
    }
}
```

**File:** `android/app/build.gradle`

Add this at the very bottom:
```gradle
apply plugin: 'com.google.gms.google-services'
```

---

## Step 6: Install Flutter Dependencies

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field
flutter pub get
```

---

## Step 7: Run the App!

### For iOS:
```bash
cd ios
pod install
cd ..
flutter run -d ios
```

### For Android:
```bash
flutter run -d android
```

### For iOS Simulator (if not on iOS device):
```bash
open -a Simulator  # Opens iOS Simulator
flutter run        # Will auto-detect simulator
```

---

## Step 8: Test the Complete Flow

1. ✅ App opens → Home screen appears
2. ✅ Tap "Draw Field on Map" → Map loads
3. ✅ Tap 4 corners to draw a field → Polygon appears
4. ✅ See area calculated at bottom
5. ✅ Tap "Analyze Field" → Loading screen
6. ✅ Wait 10-30 seconds → Results appear!
7. ✅ See crop type, confidence, and CO₂ income

---

## Troubleshooting

### "Firebase initialization failed"

**Check:**
```bash
# iOS
cat ios/Runner/GoogleService-Info.plist | grep PROJECT_ID

# Android  
cat android/app/google-services.json | grep project_id
```

Should show: `ml-pipeline-477612`

### "Backend unavailable"

**Test backend:**
```bash
curl https://YOUR-BACKEND-URL.run.app/health
```

Should return:
```json
{
  "status": "healthy",
  "earth_engine": "initialized",
  "timestamp": "2024-11-21T..."
}
```

### "Authentication failed"

**Check Firebase anonymous auth is enabled:**
1. Firebase Console → Authentication → Sign-in methods
2. Ensure "Anonymous" is ENABLED

### "Map not loading"

**Check Google Maps API key:**
```bash
# iOS
grep -A 1 "GMSApiKey" ios/Runner/Info.plist

# Android
grep -A 2 "API_KEY" android/app/src/main/AndroidManifest.xml
```

Should show your actual API key (not placeholder).

---

## Quick Command Reference

```bash
# Get backend URL
gcloud run services describe carboncheck-field-api --region us-central1 --format 'value(status.url)'

# View backend logs
gcloud run logs tail carboncheck-field-api --region us-central1

# Test backend
curl https://YOUR-URL.run.app/health

# Run Flutter app
flutter pub get
flutter run

# Check for errors
flutter analyze
```

---

## Success Checklist

Before running app, verify:

- [ ] Backend deployed to Cloud Run
- [ ] Backend URL updated in `backend_service.dart`
- [ ] Firebase iOS app added (GoogleService-Info.plist exists)
- [ ] Firebase Android app added (google-services.json exists)
- [ ] Firebase anonymous auth enabled
- [ ] Google Services plugin added to Android
- [ ] `flutter pub get` completed successfully
- [ ] Google Maps API key still in platform files

---

## Expected First Run

1. App opens (2-3 seconds)
2. Firebase initializes in background
3. Home screen appears immediately
4. Tap "Draw Field on Map"
5. Map loads (may take 5-10 seconds first time)
6. Draw field by tapping corners
7. Tap "Analyze Field"
8. Loading: "Connecting securely..." (2 sec)
9. Loading: "Analyzing satellite imagery..." (10-30 sec)
10. Results screen with crop prediction! 🎉

---

**Ready to run? Follow the steps above!** 🚀

