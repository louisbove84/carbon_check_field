# Deployment Strategy 🚀

## How API Keys Work in Production

Great question! Here's how your app gets deployed to app stores with API keys that aren't in GitHub.

---

## 🏗️ Build Process Overview

```
┌─────────────────────────────────────────────────────┐
│  1. YOUR LOCAL MACHINE (Development)                │
│     ✅ Has real API keys in platform files          │
│     ✅ Can run and test the app                     │
└─────────────────────────────────────────────────────┘
                        │
                        │ flutter build
                        ▼
┌─────────────────────────────────────────────────────┐
│  2. BUILD OUTPUT (App Bundle/IPA)                   │
│     ✅ API keys are COMPILED INTO the app           │
│     ✅ Ready for App Store / Play Store             │
└─────────────────────────────────────────────────────┘
                        │
                        │ upload
                        ▼
┌─────────────────────────────────────────────────────┐
│  3. APP STORES (Production)                         │
│     ✅ Users download app with keys inside          │
│     ✅ App works perfectly for end users            │
└─────────────────────────────────────────────────────┘
```

---

## Option 1: Build Locally (Simplest) ⭐ RECOMMENDED

This is what most small teams/solo developers do:

### Steps:

1. **Your local machine has the API keys** (already done! ✅)
2. **Build the release app locally:**

   ```bash
   # Android
   flutter build appbundle --release
   # Output: build/app/outputs/bundle/release/app-release.aab
   
   # iOS
   flutter build ios --release
   # Then archive in Xcode
   ```

3. **The API keys are compiled into the build** (they become part of the binary)
4. **Upload to app stores:**
   - **Android:** Upload `app-release.aab` to Google Play Console
   - **iOS:** Archive in Xcode and upload to App Store Connect

### ✅ Advantages:
- Simple and straightforward
- No CI/CD setup needed
- Full control over the build
- API keys never touch GitHub

### ⚠️ Considerations:
- Must build on your local machine
- Need both macOS (for iOS) and can use any OS for Android

---

## Option 2: CI/CD with Secrets (Advanced)

For automated deployments using GitHub Actions, Bitrise, Codemagic, etc.

### How It Works:

1. **Store API keys as secrets in your CI/CD platform:**
   - GitHub Actions → Repository Secrets
   - Bitrise → Secret Environment Variables
   - Codemagic → Environment Variables

2. **CI/CD injects keys during build:**
   ```yaml
   # Example: GitHub Actions
   - name: Insert API Keys
     run: |
       sed -i 's/YOUR_GOOGLE_MAPS_API_KEY_HERE/${{ secrets.GOOGLE_MAPS_KEY }}/g' \
         android/app/src/main/AndroidManifest.xml
       
       sed -i 's/YOUR_GOOGLE_MAPS_API_KEY_HERE/${{ secrets.GOOGLE_MAPS_KEY }}/g' \
         ios/Runner/Info.plist
   ```

3. **Build happens in CI/CD pipeline**
4. **Automatically deploy to stores**

### ✅ Advantages:
- Fully automated deployments
- Team members don't need API keys locally
- Consistent builds
- Easy to update keys

### ⚠️ Considerations:
- More complex setup
- Requires CI/CD platform (costs money)
- Need to configure secrets management

---

## Option 3: Build Flavors/Schemes (Production-Grade)

For apps with multiple environments (dev, staging, production).

### How It Works:

Create separate configuration files for each environment:

```
android/
  app/
    src/
      dev/
        AndroidManifest.xml       (dev API keys)
      staging/
        AndroidManifest.xml       (staging API keys)
      production/
        AndroidManifest.xml       (prod API keys)
```

Build specific flavors:
```bash
flutter build appbundle --flavor production --release
```

### ✅ Advantages:
- Separate keys for dev/staging/prod
- Very professional setup
- Easy to switch environments

### ⚠️ Considerations:
- More complex project structure
- Requires understanding build flavors

---

## 🎯 Recommended Workflow for CarbonCheck Field

### For Now (MVP/Initial Launch):

**Use Option 1: Build Locally**

1. Your local files already have the keys ✅
2. Build on your machine:
   ```bash
   # Android
   cd /Users/beuxb/Desktop/Projects/carbon_check_field
   flutter build appbundle --release
   
   # iOS (requires macOS + Xcode)
   flutter build ios --release
   open ios/Runner.xcworkspace
   # Then: Product → Archive in Xcode
   ```

3. Upload to app stores manually
4. Done! 🎉

### Later (When You Have a Team):

**Migrate to Option 2: CI/CD**

Set up GitHub Actions to:
1. Read secrets from GitHub
2. Inject into platform files
3. Build automatically
4. Deploy to stores

I can help you set this up when you're ready!

---

## 📱 What Users Download

When users install your app from app stores:

1. They get a compiled binary (`.aab` for Android, `.ipa` for iOS)
2. The API keys are **embedded in the binary**
3. The app works perfectly - users never see or need the keys
4. Google Maps loads correctly
5. All API calls work

**The keys are NOT in GitHub, but they ARE in the compiled app binary.**

---

## 🔒 Security Notes

### Is This Safe?

**Yes!** Here's why:

1. **API Key Restrictions** protect you:
   - Android keys restricted to your package name (`com.carboncheck.field`)
   - Android keys restricted to your SHA-1 certificate fingerprint
   - iOS keys restricted to your bundle ID (`com.carboncheck.field`)
   - Even if someone extracts the key from your app, they can't use it!

2. **Compiled apps are obfuscated:**
   ```bash
   flutter build appbundle --obfuscate --split-debug-info=./debug-info
   ```
   This makes it much harder to reverse engineer.

3. **Rate limiting** in Google Cloud Console

### Additional Security (Optional):

1. **Use App Check** (Firebase):
   - Verifies requests come from your real app
   - Blocks requests from unauthorized sources

2. **Backend Proxy** (most secure):
   - Don't put API keys in mobile app at all
   - Mobile app calls YOUR backend
   - Your backend calls Google APIs
   - API keys never leave your server

---

## 🧪 Test Your Build Process

Before deploying, test locally:

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field

# Test that keys are present
grep -A 2 "API_KEY" android/app/src/main/AndroidManifest.xml
# Should show: AIzaSyCAf6KxL_60PsKZlVyBLBPIyD6Rs8df0GA

# Build release version (Android)
flutter build appbundle --release

# The resulting file will have keys compiled in
ls -lh build/app/outputs/bundle/release/app-release.aab
```

---

## 📋 Pre-Deployment Checklist

Before uploading to app stores:

### Android:
- [ ] API key added to `AndroidManifest.xml`
- [ ] API key restricted in Google Cloud Console
- [ ] Package name is `com.carboncheck.field`
- [ ] Signing key configured in `build.gradle`
- [ ] Build with: `flutter build appbundle --release`
- [ ] Test the `.aab` file on real device

### iOS:
- [ ] API key added to `Info.plist`
- [ ] API key restricted in Google Cloud Console
- [ ] Bundle ID is `com.carboncheck.field`
- [ ] Code signing configured in Xcode
- [ ] Build with Xcode Archive
- [ ] Test the archive on real device

### Both:
- [ ] Test all features work
- [ ] Google Maps loads correctly
- [ ] Earth Engine calls work
- [ ] Vertex AI predictions work
- [ ] App icons added
- [ ] Splash screens added
- [ ] Privacy policy URL configured

---

## 🆘 Troubleshooting

### "Map not loading in release build"

1. Verify API key is in the file:
   ```bash
   grep "AIzaSyC" android/app/src/main/AndroidManifest.xml
   ```

2. Check API key restrictions in Google Cloud Console

3. For Android: Get release SHA-1 fingerprint:
   ```bash
   keytool -list -v -keystore ~/.android/debug.keystore
   # Add this SHA-1 to your API key restrictions
   ```

### "Authentication failed in production"

1. Check service account JSON is in `assets/service-account.json`
2. Verify it's included in the build:
   ```yaml
   # In pubspec.yaml
   flutter:
     assets:
       - assets/
   ```

3. Test with debug build first, then release

---

## 🎓 Summary

**For CarbonCheck Field:**

1. ✅ Your local files have the real API keys (correct!)
2. ✅ GitHub has template files only (secure!)
3. ✅ When you build locally, keys get compiled into the app
4. ✅ Upload the compiled app to stores
5. ✅ Users download and use the app - it works perfectly!

**The keys never go to GitHub, but they DO go into the compiled app binary.**

---

## 🚀 Ready to Deploy?

When you're ready to build for production:

1. Review the checklist above
2. Build locally with `flutter build appbundle --release`
3. Upload to Google Play Console or App Store Connect
4. Done!

Need help with the actual deployment? Let me know and I can walk you through it! 📱

---

**Next Steps:**
- Read: `SETUP_GUIDE.md` for development setup
- Read: `KEYS_SETUP.md` for API key configuration
- Read: This file for deployment strategy

**Questions?** Feel free to ask!

