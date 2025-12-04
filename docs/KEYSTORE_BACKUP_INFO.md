# Keystore Backup Information

**⚠️ IMPORTANT: Keep this information secure and backed up!**

If you lose your keystore file, you will NOT be able to update your app on Google Play Store. Google Play requires all updates to be signed with the same keystore used for the initial upload.

## Keystore Details

- **Keystore File**: `android/upload-keystore.jks`
- **Keystore Password**: `carboncheck2024`
- **Key Alias**: `upload`
- **Key Password**: `carboncheck2024`
- **Certificate**: CN=CarbonCheck Field, OU=Development, O=CarbonCheck, L=Denver, ST=CO, C=US
- **Valid Until**: April 15, 2053
- **Key Algorithm**: RSA 2048-bit

## Backup Instructions

1. **Copy the keystore file to a secure location:**
   ```bash
   cp android/upload-keystore.jks /path/to/secure/backup/location/
   ```

2. **Store the password securely** (password manager, encrypted file, etc.)

3. **Do NOT commit the keystore to Git** - it's already in `.gitignore`

4. **Recommended backup locations:**
   - Encrypted cloud storage (Google Drive, Dropbox with encryption)
   - External encrypted hard drive
   - Password manager (for the password)
   - Secure physical storage (for the file)

## What to Do If You Lose the Keystore

If you lose your keystore:
- You CANNOT update your existing app on Google Play
- You would need to create a NEW app listing with a different package name
- All existing users would need to uninstall and reinstall the new app
- You would lose all reviews, ratings, and download history

## Current AAB File Location

The release AAB file is located at:
```
build/app/outputs/bundle/release/app-release.aab
```

Full path:
```
/Users/beuxb/Desktop/Projects/carbon_check_field/build/app/outputs/bundle/release/app-release.aab
```

## Rebuilding the AAB

To rebuild the AAB file in the future:
```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field
flutter build appbundle --release
```

The keystore configuration is already set up in `android/app/build.gradle.kts` and `android/key.properties`.

---
**Created**: November 28, 2025
**App**: CarbonCheck Field
**Package**: com.carboncheck.field

