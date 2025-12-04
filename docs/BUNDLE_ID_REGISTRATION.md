# Bundle ID Registration Guide

## Current Bundle ID
`com.carboncheck.carbonCheckField`

## Quick Steps

### Method 1: Register via App Store Connect (Easiest)
1. Go to https://appstoreconnect.apple.com
2. Click **My Apps** → **+** → **New App**
3. When prompted for Bundle ID, click **"Register a new Bundle ID"**
4. Enter: `com.carboncheck.carbonCheckField`
5. Complete the app creation

### Method 2: Register via Developer Portal
1. Go to https://developer.apple.com/account
2. Click **Certificates, Identifiers & Profiles**
3. Click **Identifiers** → **+**
4. Select **App IDs** → **Continue**
5. Select **App** → **Continue**
6. Description: `CarbonCheck Field`
7. Bundle ID: Select **Explicit** → Enter `com.carboncheck.carbonCheckField`
8. Click **Continue** → **Register**

## After Registration

1. Return to Xcode
2. Go to **Signing & Capabilities**
3. Uncheck and re-check **"Automatically manage signing"**
4. Select your **Team**
5. Xcode will automatically create the provisioning profile

## Alternative: Use a Different Bundle ID

If you want to use a simpler bundle ID that Xcode can auto-register:
- `com.yourname.carboncheckfield`
- `com.carboncheck.field`

But you'll need to update it in:
- `ios/Runner.xcodeproj/project.pbxproj`
- App Store Connect (if already created)

