# API Keys Setup Guide 🔑

This guide explains how to securely configure your API keys for CarbonCheck Field.

## ⚠️ Security First!

**NEVER commit these files to git:**
- `.env` (contains your actual keys)
- `assets/service-account.json` (your GCP credentials)

These files are already in `.gitignore` to protect them! ✅

---

## Step 1: Configure Environment Variables

Copy the example file and add your keys:

```bash
cp .env.example .env
```

Then edit `.env` with your actual keys:

```bash
# .env file
GOOGLE_MAPS_ANDROID_KEY=your_actual_android_key_here
GOOGLE_MAPS_IOS_KEY=your_actual_ios_key_here
GCP_PROJECT_ID=ml-pipeline-477612
# ... etc
```

---

## Step 2: Add Google Maps Keys to Platform Files

### Android

Edit `android/app/src/main/AndroidManifest.xml`:

```xml
<!-- Replace YOUR_GOOGLE_MAPS_API_KEY_HERE with your actual key -->
<meta-data
    android:name="com.google.android.geo.API_KEY"
    android:value="YOUR_ACTUAL_API_KEY_HERE"/>
```

### iOS

Edit `ios/Runner/Info.plist`:

```xml
<!-- Replace YOUR_GOOGLE_MAPS_API_KEY_HERE with your actual key -->
<key>GMSApiKey</key>
<string>YOUR_ACTUAL_API_KEY_HERE</string>
```

---

## Step 3: Add Service Account Credentials

Copy your service account JSON file:

```bash
cp ~/path/to/your-service-account.json assets/service-account.json
```

---

## Getting Your API Keys

### Google Maps API Key

1. Go to [Google Cloud Console - Credentials](https://console.cloud.google.com/apis/credentials)
2. Create API Key (or use existing)
3. Click "Edit API key"
4. **Set restrictions:**
   - **Android:** Add package name `com.carboncheck.field` + SHA-1 fingerprint
   - **iOS:** Add bundle ID `com.carboncheck.field`
5. **Restrict API access to:**
   - Maps SDK for Android
   - Maps SDK for iOS

### Service Account Key

1. Go to [Google Cloud Console - Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
2. Select your service account (or create new one)
3. Click "Keys" → "Add Key" → "Create new key"
4. Choose JSON format
5. Download and save as `assets/service-account.json`

---

## Verify Security

Check that sensitive files are ignored:

```bash
# This should show .env in the gitignore list
cat .gitignore | grep -E "(\.env|service-account)"

# This should return empty (files not tracked by git)
git status | grep -E "(\.env|service-account\.json)"
```

---

## For Team Members

When someone clones the repo, they should:

1. Copy `.env.example` to `.env`
2. Add their own API keys to `.env`
3. Get service account JSON and save to `assets/service-account.json`
4. Update `AndroidManifest.xml` and `Info.plist` with their keys

---

## Alternative: Environment-Specific Keys

You can create multiple environment files:

```bash
.env.development    # Development keys
.env.staging        # Staging keys
.env.production     # Production keys
```

Then load the appropriate one based on build configuration.

---

## Security Best Practices

✅ **DO:**
- Keep `.env` and service account JSON in `.gitignore`
- Use API key restrictions (bundle ID, package name)
- Rotate keys periodically
- Use different keys for dev/staging/production
- Monitor API usage in Google Cloud Console

❌ **DON'T:**
- Commit `.env` to git
- Share keys in Slack/email
- Use the same key across multiple apps
- Leave keys unrestricted (anyone can use them)
- Hardcode keys directly in source code

---

## Troubleshooting

**"Map is blank"**
- Verify API key is correct
- Check API is enabled (Maps SDK for Android/iOS)
- Verify restrictions allow your bundle ID/package name
- Check billing is enabled on GCP project

**"Authentication failed"**
- Verify service account JSON is in `assets/` folder
- Check service account has Earth Engine + Vertex AI permissions
- Try regenerating the service account key

---

## Need Help?

- Check Google Cloud Console logs
- Verify keys are active (not deleted/disabled)
- Test with unrestricted key first, then add restrictions
- Contact support if issues persist

---

**🔐 Remember: Your API keys are like passwords - keep them secret!**

