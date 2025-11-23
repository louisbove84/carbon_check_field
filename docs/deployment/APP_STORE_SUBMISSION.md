# App Store Submission Guide - CarbonCheck Field

Complete guide to submit CarbonCheck Field to Apple App Store and Google Play Store.

---

## 📋 Pre-Submission Checklist

### ✅ Required Before Submission

- [ ] App is fully tested on real iOS device
- [ ] App is fully tested on real Android device
- [ ] All features working (address search, field drawing, analysis, CDL display)
- [ ] Firebase configuration files in place (`google-services.json`, `GoogleService-Info.plist`)
- [ ] Backend API is deployed and stable
- [ ] App icons and screenshots prepared
- [ ] App description and marketing materials ready
- [ ] Privacy policy URL ready
- [ ] Terms of service URL ready

---

## 🍎 iOS App Store Submission

### Step 1: Prepare iOS Build

#### 1.1 Install Xcode (if not already)
```bash
# Check if Xcode is installed
xcode-select -p
```

If not installed:
- Open App Store on Mac
- Search for "Xcode"
- Install (free, but ~12GB)

#### 1.2 Update App Information

Edit `ios/Runner/Info.plist`:
```xml
<key>CFBundleDisplayName</key>
<string>CarbonCheck Field</string>
<key>CFBundleIdentifier</key>
<string>com.carboncheck.field</string>
<key>CFBundleVersion</key>
<string>1</string>
<key>CFBundleShortVersionString</key>
<string>1.0.0</string>
```

#### 1.3 Add App Icons

You need app icons in various sizes:
- 1024x1024 (App Store)
- 180x180 (iPhone)
- 167x167 (iPad Pro)
- 152x152 (iPad)
- 120x120 (iPhone)
- 87x87 (iPhone)
- 80x80 (iPad)
- 76x76 (iPad)
- 60x60 (iPhone)
- 58x58 (iPhone)
- 40x40 (iPhone/iPad)
- 29x29 (iPhone/iPad)
- 20x20 (iPhone/iPad)

**Tool to generate all sizes:** https://appicon.co/

Save icons to: `ios/Runner/Assets.xcassets/AppIcon.appiconset/`

#### 1.4 Build iOS Release

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field

# Clean previous builds
flutter clean

# Get dependencies
flutter pub get

# Build iOS release
flutter build ios --release
```

#### 1.5 Open in Xcode

```bash
open ios/Runner.xcworkspace
```

In Xcode:
1. Select "Runner" in left sidebar
2. Go to "Signing & Capabilities" tab
3. Select your Team (Apple Developer Account)
4. Xcode will auto-generate provisioning profile

#### 1.6 Archive and Upload

In Xcode:
1. Select **Product → Archive**
2. Wait for archive to complete (5-10 minutes)
3. Click **Distribute App**
4. Select **App Store Connect**
5. Click **Upload**
6. Wait for upload (10-20 minutes)

### Step 2: App Store Connect Configuration

#### 2.1 Create App in App Store Connect

1. Go to https://appstoreconnect.apple.com
2. Click **My Apps** → **+** → **New App**
3. Fill in:
   - **Platform:** iOS
   - **Name:** CarbonCheck Field
   - **Primary Language:** English
   - **Bundle ID:** com.carboncheck.field
   - **SKU:** carboncheck-field-001

#### 2.2 App Information

**Category:**
- Primary: Business or Productivity
- Secondary: Agriculture (if available)

**Age Rating:**
- 4+ (No objectionable content)

#### 2.3 App Privacy

**Data Collection:**
- Location: Yes (for map centering)
- User Content: Yes (field boundaries)
- Analytics: No

**Privacy Policy URL:**
- Must provide a valid URL (can host on GitHub Pages)

#### 2.4 Pricing and Availability

- **Price:** Free
- **Availability:** All countries

#### 2.5 Screenshots Required

You need screenshots for:
- **iPhone 6.7" Display** (iPhone 15 Pro Max)
- **iPhone 6.5" Display** (iPhone 14 Pro Max)
- **iPhone 5.5" Display** (iPhone 8 Plus)

Minimum 3 screenshots, maximum 10.

**How to capture:**
1. Run app in iOS Simulator
2. Select device size in Xcode
3. Cmd+S to save screenshot
4. Or use Xcode → **Window → Devices and Simulators → Screenshot**

#### 2.6 App Description

```
Analyze your farm fields and estimate carbon credit income with AI-powered crop classification.

CarbonCheck Field makes it easy for farmers to:
• Draw field boundaries on satellite maps
• Get AI-powered crop type predictions
• See USDA CDL ground truth verification
• Estimate annual carbon credit income
• Search by address to quickly find your fields

How It Works:
1. Search for your farm address or manually navigate
2. Tap to draw your field boundaries on the map
3. Submit for analysis
4. Get instant results with crop type, confidence, and income estimates

Features:
✓ Google Maps satellite imagery
✓ Real-time field area calculation
✓ AI crop classification (Corn, Soybeans, Alfalfa, Winter Wheat)
✓ USDA Cropland Data Layer verification
✓ Carbon credit income estimates (2025 market rates)
✓ Secure Firebase authentication
✓ Share results via text or email

CarbonCheck Field uses advanced machine learning and satellite data to help farmers understand their carbon credit potential. All predictions are verified against USDA CDL ground truth data for maximum accuracy.

Perfect for farmers, agronomists, and agricultural consultants looking to estimate carbon credit income opportunities.
```

**Keywords:**
```
agriculture, farming, carbon credits, crop analysis, satellite imagery, farm management, AI agriculture, carbon farming, sustainable agriculture, precision agriculture
```

#### 2.7 What's New (Version 1.0.0)

```
🌾 Initial Release

• AI-powered crop classification
• USDA CDL ground truth verification
• Address search for easy navigation
• Real-time field area calculation
• Carbon credit income estimates
• Beautiful farmer-friendly interface
```

### Step 3: Submit for Review

1. Select your uploaded build
2. Fill in all required information
3. Click **Save**
4. Click **Submit for Review**

**Review Time:** 24-48 hours typically

### Step 4: After Approval

- App goes live automatically (or on date you choose)
- Users can download from App Store
- Monitor crashes and reviews in App Store Connect

---

## 🤖 Google Play Store Submission

### Step 1: Prepare Android Build

#### 1.1 Create Keystore (for signing)

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/android/app

keytool -genkey -v -keystore carboncheck-release-key.jks \
  -keyalg RSA -keysize 2048 -validity 10000 \
  -alias carboncheck
```

Enter password and details when prompted.

**IMPORTANT:** Save this keystore file and password! You need it for all future updates.

#### 1.2 Configure Signing

Create `android/key.properties`:
```properties
storePassword=YOUR_KEYSTORE_PASSWORD
keyPassword=YOUR_KEY_PASSWORD
keyAlias=carboncheck
storeFile=/Users/beuxb/Desktop/Projects/carbon_check_field/android/app/carboncheck-release-key.jks
```

Add to `.gitignore`:
```
android/key.properties
android/app/*.jks
```

Edit `android/app/build.gradle`:

```gradle
// Add before 'android {' block
def keystoreProperties = new Properties()
def keystorePropertiesFile = rootProject.file('key.properties')
if (keystorePropertiesFile.exists()) {
    keystoreProperties.load(new FileInputStream(keystorePropertiesFile))
}

android {
    // ... existing code ...

    signingConfigs {
        release {
            keyAlias keystoreProperties['keyAlias']
            keyPassword keystoreProperties['keyPassword']
            storeFile keystoreProperties['storeFile'] ? file(keystoreProperties['storeFile']) : null
            storePassword keystoreProperties['storePassword']
        }
    }

    buildTypes {
        release {
            signingConfig signingConfigs.release
            // ... rest of release config
        }
    }
}
```

#### 1.3 Update App Information

Edit `android/app/src/main/AndroidManifest.xml`:
```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.carboncheck.field">
    
    <application
        android:label="CarbonCheck Field"
        android:icon="@mipmap/ic_launcher">
        <!-- ... rest of manifest -->
    </application>
</manifest>
```

#### 1.4 Add App Icons

Generate Android icons using https://appicon.co/

Replace files in:
- `android/app/src/main/res/mipmap-hdpi/`
- `android/app/src/main/res/mipmap-mdpi/`
- `android/app/src/main/res/mipmap-xhdpi/`
- `android/app/src/main/res/mipmap-xxhdpi/`
- `android/app/src/main/res/mipmap-xxxhdpi/`

#### 1.5 Build Android Release

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field

# Clean previous builds
flutter clean

# Get dependencies
flutter pub get

# Build Android app bundle
flutter build appbundle --release
```

Output will be at: `build/app/outputs/bundle/release/app-release.aab`

### Step 2: Google Play Console Setup

#### 2.1 Create Developer Account

1. Go to https://play.google.com/console
2. Pay one-time $25 registration fee
3. Complete account setup

#### 2.2 Create App

1. Click **Create app**
2. Fill in:
   - **App name:** CarbonCheck Field
   - **Default language:** English (United States)
   - **App or game:** App
   - **Free or paid:** Free

#### 2.3 App Content

**Privacy Policy:**
- Must provide URL

**App Access:**
- All features available without restrictions

**Ads:**
- No ads

**Content Rating:**
- Complete questionnaire (select "No" for all inappropriate content)
- Result: Everyone

**Target Audience:**
- Target age: 18+
- Not designed for children

#### 2.4 Store Listing

**App name:** CarbonCheck Field

**Short description (80 chars):**
```
AI-powered crop analysis and carbon credit income estimation for farmers
```

**Full description (4000 chars max):**
```
[Use same description as iOS above]
```

**App icon:** 512x512 PNG (no transparency)

**Feature graphic:** 1024x500 PNG (required for featured placement)

**Screenshots:**
- Minimum 2, maximum 8
- Phone: 16:9 or 9:16 aspect ratio
- Tablet: Optional but recommended

**Video:** (Optional) YouTube link to demo

#### 2.5 Upload App Bundle

1. Go to **Production → Create new release**
2. Upload `app-release.aab`
3. Fill in release notes:

```
Initial Release - Version 1.0.0

✓ AI-powered crop classification
✓ USDA CDL ground truth verification
✓ Address search for farm navigation
✓ Real-time field area calculation
✓ Carbon credit income estimates
✓ Farmer-friendly interface
```

4. Click **Save** → **Review release**
5. Click **Start rollout to Production**

### Step 3: Review Process

- **Review time:** 1-7 days typically
- Monitor status in Play Console
- Respond promptly to any review feedback

---

## 🚀 Post-Launch

### Monitoring

**iOS:**
- App Store Connect → Analytics
- Crashes and feedback

**Android:**
- Play Console → Dashboard
- Crash reports and ANRs

### Updates

When you make changes:

1. Increment version in `pubspec.yaml`:
```yaml
version: 1.0.1+2  # 1.0.1 is version name, 2 is build number
```

2. Rebuild and resubmit to both stores

### Marketing

- Share on social media
- Reach out to farming communities
- Consider agriculture publications
- Attend farming conferences

---

## 📞 Support

### Common Issues

**"Missing compliance" (iOS):**
- Go to App Store Connect
- Select your app version
- Answer export compliance questions
- Usually: No encryption (unless you added custom encryption)

**"App bundle not signed" (Android):**
- Make sure keystore is configured correctly
- Rebuild with release signing

**"Location permission denied":**
- Both stores require clear explanation of location usage
- Update Info.plist and AndroidManifest.xml with usage descriptions

---

## 🎯 Success Metrics to Track

- Downloads
- Daily Active Users (DAU)
- Field analyses completed
- Average field size
- Most common crop types
- Crash rate (keep below 1%)
- User reviews and ratings

---

## 📚 Additional Resources

- [Flutter iOS Deployment](https://docs.flutter.dev/deployment/ios)
- [Flutter Android Deployment](https://docs.flutter.dev/deployment/android)
- [App Store Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)
- [Google Play Policy](https://play.google.com/about/developer-content-policy/)

---

**Good luck with your submission! 🌾📱**

