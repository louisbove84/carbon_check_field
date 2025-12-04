# App Store Deployment Guide for CarbonCheck Field

This guide walks through deploying CarbonCheck Field to the Apple App Store.

## Prerequisites

- ✅ Apple Developer Account (already set up)
- ✅ App icons configured (all sizes present)
- ✅ Bundle ID: `com.carboncheck.carbonCheckField`
- ✅ Version: 1.0.0 (Build 1)

## Step 1: Create App Store Connect Listing

1. Go to [App Store Connect](https://appstoreconnect.apple.com)
2. Click **My Apps** → **+** → **New App**
3. Fill in:
   - **Platform**: iOS
   - **Name**: CarbonCheck Field
   - **Primary Language**: English
   - **Bundle ID**: `com.carboncheck.carbonCheckField` (register if needed)
   - **SKU**: `carboncheck-field-001` (unique identifier)
   - **User Access**: Full Access

## Step 2: Configure App Information

### App Information Tab
- **Category**: 
  - Primary: Business
  - Secondary: Productivity
- **Privacy Policy URL**: (required) - Add your privacy policy URL
- **Subtitle**: "Analyze crops and estimate carbon credits from satellite imagery"

### Pricing and Availability
- Set price (Free or Paid)
- Select countries/regions

## Step 3: Prepare App Store Assets

### Required Assets:
1. **App Icon**: 1024x1024 (already have `Icon-App-1024x1024@1x.png`)
2. **Screenshots** (required for each device size):
   - iPhone 6.7" (iPhone 14 Pro Max, 15 Pro Max): 1290x2796
   - iPhone 6.5" (iPhone 11 Pro Max, XS Max): 1242x2688
   - iPhone 5.5" (iPhone 8 Plus): 1242x2208
   - iPad Pro 12.9": 2048x2732
   - iPad Pro 11": 1668x2388
3. **App Preview Video** (optional but recommended)
4. **Description**: 
   ```
   CarbonCheck Field helps farmers analyze crop types and estimate carbon credit income using satellite imagery and AI.
   
   Features:
   • Draw field boundaries on interactive satellite maps
   • AI-powered crop classification (Corn, Soybeans, Alfalfa, Winter Wheat)
   • Real-time acreage calculation
   • Carbon credit estimates based on 2025 market rates
   • Multi-zone analysis for large fields (up to 2,000 acres)
   • Secure, cloud-based processing
   ```
5. **Keywords**: farming, agriculture, carbon credits, crop analysis, satellite imagery, field mapping
6. **Support URL**: Your support website
7. **Marketing URL** (optional): Your marketing website

## Step 4: Configure Release Build in Xcode

### In Xcode:

1. **Open the project**:
   ```bash
   open ios/Runner.xcworkspace
   ```

2. **Select Runner target** → **Signing & Capabilities**:
   - ✅ Automatically manage signing
   - Select your **Team** (Apple Developer account)
   - Bundle Identifier: `com.carboncheck.carbonCheckField`

3. **Build Settings** → **Release**:
   - Ensure **Code Signing Identity** is set to "Apple Distribution"
   - **Provisioning Profile** should be automatic

4. **Product** → **Scheme** → **Edit Scheme**:
   - Select **Run** → **Build Configuration**: Debug
   - Select **Archive** → **Build Configuration**: Release

## Step 5: Build Archive for Release

### Option A: Using Xcode (Recommended)

1. In Xcode, select **Any iOS Device** or **Generic iOS Device** from device dropdown
2. **Product** → **Archive**
3. Wait for archive to complete
4. **Organizer** window opens automatically
5. Click **Distribute App**
6. Select **App Store Connect**
7. Click **Next** → **Upload**
8. Select your distribution certificate and provisioning profile
9. Click **Upload**

### Option B: Using Command Line

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field

# Build for release
flutter build ios --release

# Archive using xcodebuild
cd ios
xcodebuild -workspace Runner.xcworkspace \
  -scheme Runner \
  -configuration Release \
  -archivePath build/Runner.xcarchive \
  archive

# Export IPA (requires manual steps in Xcode Organizer)
```

## Step 6: Upload to App Store Connect

After archiving in Xcode:
1. **Window** → **Organizer** (or it opens automatically)
2. Select your archive
3. Click **Distribute App**
4. Choose **App Store Connect**
5. Follow the wizard:
   - Distribution method: **Upload**
   - App Thinning: **All compatible device variants**
   - Re-sign if needed
6. Click **Upload**

**Alternative**: Use **Transporter** app (download from Mac App Store)
- Export IPA from Xcode
- Open Transporter
- Drag IPA file
- Click **Deliver**

## Step 7: Complete App Store Connect Listing

1. Go back to App Store Connect
2. Wait for processing (can take 10-30 minutes)
3. Once processing completes:
   - Go to **App Store** tab
   - Fill in all required fields:
     - Screenshots
     - Description
     - Keywords
     - Support URL
     - Privacy Policy URL
   - Upload app preview (optional)
   - Set age rating
   - Add app review information:
     - Demo account (if needed)
     - Notes for reviewer
     - Contact info

## Step 8: Submit for Review

1. In App Store Connect, click **+ Version or Platform**
2. Select the build that finished processing
3. Fill in **What's New in This Version**:
   ```
   Initial release of CarbonCheck Field!
   
   - Draw field boundaries on satellite maps
   - AI-powered crop classification
   - Carbon credit estimates
   - Real-time acreage calculation
   ```
4. Answer **App Review Information**:
   - Demo account credentials (if login required)
   - Notes: "This app uses anonymous Firebase authentication. No login required for testing."
5. Click **Add for Review**
6. Click **Submit for Review**

## Step 9: Monitor Review Status

- Check status in App Store Connect
- Typical review time: 24-48 hours
- You'll receive email notifications for:
  - Submission received
  - In review
  - Approved/Rejected

## Common Issues & Solutions

### Issue: "Missing Compliance"
- **Solution**: Answer export compliance questions in App Store Connect
- Most apps: "No" to encryption questions (unless using custom encryption)

### Issue: "Invalid Bundle"
- **Solution**: Ensure bundle ID matches exactly in Xcode and App Store Connect

### Issue: "Missing Screenshots"
- **Solution**: Take screenshots on actual devices or use simulator
- Required for at least one device family (iPhone or iPad)

### Issue: Code Signing Errors
- **Solution**: 
  - Ensure "Automatically manage signing" is enabled
  - Select correct Team
  - Clean build folder (⇧⌘K) and rebuild

## Version Updates

For future updates:
1. Update version in `pubspec.yaml`: `version: 1.0.1+2`
2. Increment build number each upload
3. Follow same archive/upload process
4. Update "What's New" in App Store Connect

## Resources

- [App Store Connect Help](https://help.apple.com/app-store-connect/)
- [App Store Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)
- [Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)

## Checklist Before Submission

- [ ] App builds and runs successfully in Release mode
- [ ] All app icons are present (1024x1024 required)
- [ ] Screenshots taken for required device sizes
- [ ] App description and metadata complete
- [ ] Privacy policy URL added
- [ ] Support URL added
- [ ] Age rating completed
- [ ] Export compliance answered
- [ ] App review information provided
- [ ] Tested on physical device (recommended)
- [ ] No console errors or crashes
- [ ] All features work as expected

---

**Next Steps**: Start with Step 1 (App Store Connect listing) and work through each step systematically.


