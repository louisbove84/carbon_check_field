# Security Audit - API Keys & Secrets

## Summary

**Date:** 2025-11-23  
**Status:** ⚠️ Some API keys found in tracked files

## Files with API Keys

### ✅ Properly Ignored (in .gitignore)
- `ios/Runner/GoogleService-Info.plist` - Firebase config (ignored)
- `ios/Runner/Info.plist` - iOS config (ignored)
- `android/app/google-services.json` - Firebase config (ignored)
- `android/app/src/main/AndroidManifest.xml` - Android config (ignored)

### ✅ Fixed - Moved to Environment Variables

#### 1. `lib/utils/constants.dart`
- **Status:** ✅ **FIXED** - Now loads from `.env` file
- **Implementation:** Uses `flutter_dotenv` to load `GOOGLE_MAPS_API_KEY` from `.env`
- **Security:** API key no longer hardcoded in source

#### 2. `lib/firebase_options.dart`
- **Contains:** Firebase API keys (auto-generated)
- **Keys:** Multiple Firebase API keys for web/android/ios
- **Risk:** Low (Firebase API keys are meant to be public, but should be restricted)
- **Recommendation:**
  - ✅ This file is typically committed (FlutterFire standard)
  - Ensure Firebase API keys have proper restrictions in Firebase Console
  - These keys are restricted by app ID/package name

#### 3. `web/index.html`
- **Status:** ✅ **FIXED** - Now loads from `web/config.js`
- **Implementation:** Dynamically loads API key from `config.js` (generated from `.env`)
- **Security:** API key no longer hardcoded in HTML

## Recommendations

### ✅ Completed Actions

1. **✅ Moved API Keys to Environment Variables:**
   - Created `.env` file (ignored by git)
   - Created `.env.example` template
   - Updated `constants.dart` to load from `.env`
   - Updated `web/index.html` to load from `config.js`
   - Added `web/config.js` to `.gitignore`

2. **✅ Setup Documentation:**
   - Created `ENV_SETUP.md` with setup instructions
   - Created `generate_web_config.sh` helper script

### Remaining Actions

1. **Verify API Key Restrictions:**
   ```bash
   # Check Google Cloud Console → APIs & Services → Credentials
   # Ensure all API keys have:
   # - Application restrictions (Android/iOS package names, HTTP referrers)
   # - API restrictions (only Maps JavaScript API, Geocoding API)
   ```

2. **Install Dependencies:**
   ```bash
   flutter pub get  # Installs flutter_dotenv
   ```

### Long-term Improvements

1. **Move to Build Configuration:**
   - Use `--dart-define` for API keys at build time
   - Example: `flutter build --dart-define=GOOGLE_MAPS_KEY=your_key`

2. **Use Secrets Management:**
   - For CI/CD: Use GitHub Secrets or similar
   - For local: Use `.env` files (already in .gitignore)

3. **API Key Rotation:**
   - Rotate keys if they've been exposed
   - Update restrictions after rotation

## Files Checked

✅ All files in `.gitignore` are properly ignored  
✅ No service account JSON files found in tracked files  
✅ `.env` file is properly ignored  
✅ `web/config.js` is properly ignored  
✅ API keys moved to environment variables  
✅ Only templates (`.env.example`, `config.js.example`) are tracked

## Notes

- **Firebase API keys** in `firebase_options.dart` are **safe to commit** (standard FlutterFire practice)
- **Google Maps API keys** should ideally be in environment variables
- All keys should have proper restrictions in Google Cloud Console

---

**Status:** ✅ **COMPLETED**
- API keys moved to `.env` file
- Web config moved to `config.js` (generated from `.env`)
- All sensitive files properly ignored
- Documentation created (`ENV_SETUP.md`)

**Next Steps:**
1. Run `flutter pub get` to install `flutter_dotenv`
2. Verify API key restrictions in Google Cloud Console
3. Document key rotation process

