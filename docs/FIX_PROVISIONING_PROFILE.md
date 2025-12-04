# Fix Provisioning Profile Issue

## Problem
Xcode is trying to create a **development** profile, which requires registered devices. For App Store distribution, we need a **distribution** profile (no devices needed).

## Solution Options

### Option 1: Configure for App Store Distribution (Recommended)

1. In Xcode, go to **Signing & Capabilities**
2. Under **Signing**, you should see options for different build configurations
3. Make sure you're looking at the **Release** configuration:
   - Click the dropdown next to "Debug" at the top (if visible)
   - Or go to **Product** → **Scheme** → **Edit Scheme**
   - Select **Archive** → Set to **Release**

4. In **Signing & Capabilities**:
   - Check **"Automatically manage signing"**
   - Select your **Team**
   - Xcode should create a **Distribution** profile automatically

### Option 2: Add a Device (If you have an iPhone/iPad)

1. Connect your iPhone/iPad to your Mac via USB
2. Unlock the device and trust the computer
3. In Xcode, the device should appear in the device dropdown
4. Xcode will automatically register it
5. Then try signing again

### Option 3: Manually Register Device (If you have device but can't connect)

1. Get your device UDID:
   - Connect device to Mac
   - Open **Finder** → Select device → See UDID
   - Or Settings → General → About → Copy Identifier

2. Go to https://developer.apple.com/account
3. **Certificates, Identifiers & Profiles** → **Devices**
4. Click **+** → Enter device name and UDID
5. Register device
6. Go back to Xcode and refresh signing

### Option 4: Create Distribution Profile Manually

1. Go to https://developer.apple.com/account
2. **Certificates, Identifiers & Profiles** → **Profiles**
3. Click **+**
4. Select **App Store** (under Distribution)
5. Select your App ID: `com.carboncheck.carbonCheckField`
6. Select your Distribution Certificate
7. Name it: "CarbonCheck Field App Store"
8. Download the profile
9. In Xcode, uncheck "Automatically manage signing"
10. Select the downloaded profile

## Quick Fix: Try This First

1. In Xcode **Signing & Capabilities**:
   - Uncheck "Automatically manage signing"
   - Check it again
   - Select your Team
   - Make sure you're in **Release** configuration

2. **Product** → **Scheme** → **Edit Scheme**
   - Select **Archive**
   - **Build Configuration**: **Release**
   - Close

3. Try **Product** → **Archive** again

## For App Store: You Don't Need Devices!

Remember: For **App Store distribution**, you don't need physical devices. The issue is Xcode is trying to create a development profile instead of a distribution profile.

The key is making sure you're building in **Release** mode for **Archive**.

