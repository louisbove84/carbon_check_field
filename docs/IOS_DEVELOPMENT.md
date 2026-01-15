# iOS Development Guide 🍎

Complete guide for iOS development, testing, and deployment.

## Table of Contents

1. [Running on iOS Simulator](#running-on-ios-simulator)
2. [Building for App Store](#building-for-app-store)
3. [Troubleshooting](#troubleshooting)

---

## Running on iOS Simulator

Quick guide to run your Flutter app on an iOS Simulator for local testing.

### Prerequisites

1. **Xcode installed** (from Mac App Store)
2. **Xcode Command Line Tools** installed
3. **Flutter SDK** installed and configured
4. **CocoaPods** installed (for iOS dependencies)

### Verify Setup

```bash
# Check Flutter installation
flutter doctor

# Check Xcode installation
xcodebuild -version

# Check CocoaPods
pod --version
```

If `flutter doctor` shows iOS issues, run:
```bash
# Accept Xcode license
sudo xcodebuild -license accept

# Install CocoaPods if missing
sudo gem install cocoapods
```

### Quick Start

#### Option 1: Flutter Auto-Detection (Easiest)

```bash
# Get dependencies
flutter pub get

# Install iOS dependencies
cd ios
pod install
cd ..

# Run on iOS Simulator (Flutter will auto-select or prompt)
flutter run
```

Flutter will:
- Automatically detect available simulators
- Launch a simulator if none are running
- Build and install your app

#### Option 2: Specify Simulator Explicitly

```bash
# List available iOS simulators
flutter devices

# Run on specific simulator
flutter run -d <device-id>

# Example: Run on iPhone 15 Pro
flutter run -d "iPhone 15 Pro"
```

#### Option 3: Open Simulator First, Then Run

```bash
# Open Simulator app manually
open -a Simulator

# Or launch specific device
xcrun simctl boot "iPhone 15 Pro"
open -a Simulator

# Then run Flutter
flutter run
```

### Common Simulator Commands

```bash
# List available simulators
flutter devices
xcrun simctl list devices available

# Boot a specific simulator
xcrun simctl boot "iPhone 15 Pro"

# Shutdown simulator
xcrun simctl shutdown all
xcrun simctl shutdown "iPhone 15 Pro"

# Reset simulator
xcrun simctl erase "iPhone 15 Pro"
```

### Hot Reload & Hot Restart

While the app is running:

- **Hot Reload**: Press `r` in terminal (preserves app state)
- **Hot Restart**: Press `R` in terminal (resets app state)
- **Quit**: Press `q` in terminal

### Debug Mode vs Release Mode

**Debug Mode (Default):**
```bash
flutter run                    # Debug mode
```
- Hot reload enabled
- Slower performance
- Debug symbols included
- Verbose logging

**Release Mode (Production-like):**
```bash
flutter run --release          # Release mode
```
- No hot reload
- Optimized performance
- Smaller app size
- Production-like behavior

---

## Building for App Store

This guide explains how to build and submit the CarbonCheck Field iOS app to the App Store.

### Quick Start

#### Build Script + Xcode GUI (Recommended)

**Step 1: Build Flutter app**
```bash
# Build Flutter app (handles extended attributes cleanup)
./build_ios.sh

# Or with clean
./build_ios.sh --clean
```

The script will:
- Get Flutter dependencies
- Install CocoaPods
- Build Flutter app for iOS release
- Clean extended attributes (fixes code signing issues)
- Open Xcode automatically

**Step 2: Archive in Xcode**

Once Xcode opens:
1. Select **Any iOS Device** or **Generic iOS Device** (top toolbar)
2. **Product** → **Archive**
3. Wait for archive to complete
4. In Organizer, click **Distribute App** → **App Store Connect** → **Upload**

**Why Xcode GUI?**
- CocoaPods targets require automatic signing
- Xcode handles Pod signing automatically
- Easier to manage provisioning profiles
- Organizer simplifies upload process

### Prerequisites

- ✅ Xcode installed (latest version recommended)
- ✅ Flutter SDK installed and configured
- ✅ CocoaPods installed (`sudo gem install cocoapods`)
- ✅ Apple Developer account configured in Xcode
- ✅ Provisioning profile "CarbonCheck Field App Store" installed
- ✅ Distribution certificate installed

### Current App Configuration

- **Version**: 1.0.3
- **Build Number**: 3
- **Bundle ID**: `com.carboncheck.field`
- **Minimum iOS Version**: 14.0
- **App Name**: CarbonCheck Field

### Build Script Details

The `build_ios.sh` script automates the iOS build process:

1. ✅ Checks for Flutter and Xcode
2. ✅ Gets Flutter dependencies (`flutter pub get`)
3. ✅ Installs CocoaPods dependencies (`pod install`)
4. ✅ Builds Flutter app for iOS release (`flutter build ios --release`)
5. ✅ Cleans extended attributes (fixes code signing issues)
6. ✅ Opens Xcode for archiving

### Upload to App Store Connect

After building:

1. **Open Xcode Organizer**:
   - Window → Organizer
   - Or it opens automatically after archive

2. **Select your archive** (should be listed)

3. **Click "Distribute App"**

4. **Choose "App Store Connect"**

5. **Select "Upload"**

6. **Follow the wizard**:
   - Review app information
   - Select distribution options
   - Click Upload

7. **Wait for processing** (10-30 minutes)

8. **Check App Store Connect**:
   - Go to https://appstoreconnect.apple.com
   - Your app → TestFlight tab
   - Build will appear after processing

### Files

- `build_ios.sh` - Build script (prepares Flutter app, opens Xcode)
- `ios/ExportOptions.plist` - Export configuration (for IPA export if needed)
- `build/Runner.xcarchive` - Xcode archive (created in Xcode)

---

## Troubleshooting

### Simulator Issues

#### "No iOS simulators found"
```bash
# Open Xcode and create a simulator
open -a Xcode
# Xcode → Window → Devices and Simulators → + → Add Simulator
```

#### "CocoaPods not installed"
```bash
sudo gem install cocoapods
cd ios
pod install
cd ..
```

#### "Pod install fails"
```bash
cd ios
rm -rf Pods Podfile.lock
pod cache clean --all
pod install --repo-update
cd ..
```

#### "Simulator is slow or unresponsive"
```bash
# Reset simulator
xcrun simctl shutdown all
xcrun simctl erase all

# Or restart specific device
xcrun simctl shutdown "iPhone 15 Pro"
xcrun simctl boot "iPhone 15 Pro"
```

#### "App crashes on launch"
```bash
# View device logs
xcrun simctl spawn booted log stream --level=debug

# Or check Flutter logs
flutter logs
```

### Build Issues

#### "resource fork, Finder information, or similar detritus not allowed"
This is a macOS extended attributes issue. The build script handles this automatically, but if you still encounter it:

**Quick Fix:**
```bash
# Clean extended attributes from Flutter framework
xattr -rc build/ios/Release-iphoneos/Flutter.framework

# Or clean entire build directory
xattr -rc build/ios/Release-iphoneos

# Then rebuild
./build_ios.sh --clean
```

#### "No signing certificate found"
- Open Xcode → Preferences → Accounts
- Add your Apple ID if not present
- Download certificates manually if needed

#### "does not support provisioning profiles"
**This happens when CocoaPods targets get manual provisioning profiles.**

**Solution:** Always use Xcode GUI for archiving (recommended workflow):
1. Build Flutter app: `./build_ios.sh`
2. Archive in Xcode GUI (Xcode handles Pod signing automatically)

#### "Provisioning profile not found"
- Ensure "CarbonCheck Field App Store" profile is installed
- Check Xcode → Preferences → Accounts → Download Manual Profiles
- Use Xcode GUI for archiving (handles profiles automatically)

#### "Build fails with signing errors"
- Ensure you're using the correct provisioning profile
- Check that your Apple Developer account has App Store distribution enabled
- Verify team ID matches: `5ULNK8BCHT`

### App-Specific Issues

#### "Google Maps not showing"
**Verify:**
1. `.env` file exists with `GOOGLE_MAPS_API_KEY`
2. iOS Maps SDK is enabled in Google Cloud Console
3. Bundle ID matches your Google Cloud project

#### "Firebase not initializing"
**Verify:**
1. `ios/Runner/GoogleService-Info.plist` exists
2. Firebase project is configured correctly
3. Anonymous authentication is enabled in Firebase Console

### Build fails with code signing errors
```bash
# Clean build
flutter clean
rm -rf ios/Pods ios/Podfile.lock
cd ios
pod install
cd ..
flutter pub get
flutter run
```

---

## Tips

1. **Keep Simulator Running**: Don't close simulator between runs - Flutter will reuse it
2. **Use Multiple Simulators**: Test on different devices simultaneously
3. **Keyboard Shortcuts**: 
   - `Cmd + Shift + H` - Home button
   - `Cmd + K` - Toggle keyboard
   - `Cmd + Left/Right` - Rotate device
4. **Screenshots**: `xcrun simctl io booted screenshot screenshot.png`
5. **Record Video**: `xcrun simctl io booted recordVideo video.mov`

---

## Quick Reference

### Simulator
```bash
flutter devices              # List devices
flutter run                 # Run on iOS Simulator
flutter run -d <device-id>  # Run on specific device
r                           # Hot reload
R                           # Hot restart
q                           # Quit
```

### Build
```bash
./build_ios.sh              # Build and open Xcode
./build_ios.sh --clean      # Clean build
```

---

## Related Documentation

- [App Store Deployment Guide](./APP_STORE_DEPLOYMENT.md)
- [iOS Upload Guide](./IOS_UPLOAD_GUIDE.md)
- [Provisioning Profile Fix](./FIX_PROVISIONING_PROFILE.md)
