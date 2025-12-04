# iOS Build Upload Guide for App Store Connect

## Step 1: Prepare for Archive in Xcode

1. **Open Xcode** (should already be open with `Runner.xcworkspace`)

2. **Verify Signing Configuration**:
   - Select **Runner** project in left sidebar
   - Select **Runner** target
   - Go to **Signing & Capabilities** tab
   - Ensure:
     - ✅ **Automatically manage signing** is checked
     - **Team** is selected (your Apple Developer account)
     - **Bundle Identifier**: `com.carboncheck.carbonCheckField`
     - **Provisioning Profile** shows "Xcode Managed Profile"

3. **Set Build Configuration to Release**:
   - Look at the **top toolbar** in Xcode (where the Play/Stop buttons are)
   - Next to the device selector, you'll see a dropdown that says **"Runner"** (this is the Scheme selector)
   - Click on **"Runner"** → Select **"Edit Scheme..."** from the dropdown menu
   - OR use menu: Click **"Product"** in the top menu bar → **"Scheme"** → **"Edit Scheme..."**
   - In the window that opens, select **"Archive"** on the left sidebar
   - In the **"Build Configuration"** dropdown, select **"Release"**
   - Click **"Close"**

## Step 2: Create Archive

1. **Select Generic iOS Device**:
   - In the device dropdown (top toolbar), select **Any iOS Device** or **Generic iOS Device**
   - ⚠️ **Important**: Do NOT select a simulator - you must select a device

2. **Create Archive**:
   - **Product** → **Archive**
   - Wait for the build to complete (may take 2-5 minutes)
   - The **Organizer** window will open automatically when done

## Step 3: Upload to App Store Connect

### Option A: Upload via Xcode Organizer (Recommended)

1. In the **Organizer** window, you'll see your archive listed
2. Click **Distribute App**
3. Select **App Store Connect**
4. Click **Next**
5. Choose **Upload**
6. Click **Next**
7. Select your distribution options:
   - ✅ **Include bitcode for iOS content** (if available)
   - ✅ **Upload your app's symbols** (recommended)
8. Click **Next**
9. **Review** the app information
10. Click **Upload**
11. Wait for upload to complete (may take 5-15 minutes)

### Option B: Export IPA and Use Transporter

1. In **Organizer**, select your archive
2. Click **Distribute App**
3. Select **App Store Connect**
4. Click **Next**
5. Choose **Export**
6. Click **Next**
7. Select distribution options
8. Click **Next**
9. Choose export location
10. Click **Export**
11. Download **Transporter** app from Mac App Store
12. Open **Transporter**
13. Drag the exported `.ipa` file into Transporter
14. Click **Deliver**

## Step 4: Verify Upload in App Store Connect

1. Go to [App Store Connect](https://appstoreconnect.apple.com)
2. Select your app (CarbonCheck Field)
3. Go to **TestFlight** tab (builds appear here first)
4. Wait for processing (10-30 minutes)
5. Once processing completes, the build will appear in **App Store** tab
6. You can then select it for your release

## Troubleshooting

### Error: "No signing certificate found"
- **Solution**: Ensure your Apple Developer account is properly configured in Xcode
- Go to **Xcode** → **Preferences** → **Accounts**
- Add your Apple ID if not present
- Download manual profiles if needed

### Error: "Bundle ID not found"
- **Solution**: Create the app in App Store Connect first with bundle ID `com.carboncheck.carbonCheckField`

### Error: "Invalid bundle"
- **Solution**: 
  - Clean build folder: **Product** → **Clean Build Folder** (⇧⌘K)
  - Delete derived data: `rm -rf ~/Library/Developer/Xcode/DerivedData`
  - Rebuild archive

### Build takes too long
- This is normal for release builds
- First build may take 5-10 minutes
- Subsequent builds are faster

## Quick Command Reference

```bash
# Open Xcode workspace
cd /Users/beuxb/Desktop/Projects/carbon_check_field
open ios/Runner.xcworkspace

# Clean build (if needed)
cd ios
xcodebuild clean -workspace Runner.xcworkspace -scheme Runner
```

## Current App Info

- **Version**: 1.0.0
- **Build**: 2
- **Bundle ID**: com.carboncheck.carbonCheckField
- **App Name**: CarbonCheck Field

---

**Next Steps After Upload**:
1. Wait for processing in App Store Connect
2. Complete your app listing (screenshots, description, etc.)
3. Select the build for your release
4. Submit for review

