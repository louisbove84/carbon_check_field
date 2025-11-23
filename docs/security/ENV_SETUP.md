# Environment Variables Setup

## Overview

API keys are now stored in `.env` files instead of hardcoded in source files for better security.

## Setup Instructions

### 1. Install Dependencies

```bash
flutter pub get
```

This will install `flutter_dotenv` package.

### 2. Create .env File

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your actual API keys
nano .env  # or use your preferred editor
```

### 3. For Web Builds

The web version needs a `config.js` file. Generate it from `.env`:

```bash
# Option 1: Manual (copy from .env)
# Edit web/config.js and add your key:
# var GOOGLE_MAPS_API_KEY = 'your_key_here';

# Option 2: Automated (create a script)
# Create web/config.js from .env:
grep GOOGLE_MAPS_API_KEY .env | cut -d '=' -f2 | sed "s/^/var GOOGLE_MAPS_API_KEY = '/" | sed "s/$/';/" > web/config.js
```

Or manually copy the key from `.env` to `web/config.js`.

## File Structure

```
carbon_check_field/
├── .env                    # Your actual API keys (NOT in git)
├── .env.example            # Template (in git)
├── web/
│   ├── config.js           # Web API keys (NOT in git)
│   └── config.js.example   # Template (in git)
└── lib/
    └── utils/
        └── constants.dart   # Loads from .env
```

## Environment Variables

### GOOGLE_MAPS_API_KEY

- **Purpose:** Google Maps API key for maps and geocoding
- **Where used:**
  - Mobile apps: Loaded from `.env` via `AppConstants.googleMapsApiKey`
  - Web: Loaded from `web/config.js`
- **Get your key:** https://console.cloud.google.com/apis/credentials
- **Security:** Make sure to restrict by app bundle ID/package name

## Build Process

### Mobile (Android/iOS)

The `.env` file is automatically loaded when the app starts (see `lib/main.dart`).

### Web

1. Generate `web/config.js` from `.env` before building:
   ```bash
   # Quick script to generate config.js
   echo "var GOOGLE_MAPS_API_KEY = '$(grep GOOGLE_MAPS_API_KEY .env | cut -d '=' -f2)';" > web/config.js
   ```

2. Build web app:
   ```bash
   flutter build web
   ```

## CI/CD Integration

For automated builds, set environment variables in your CI/CD platform:

```yaml
# Example: GitHub Actions
env:
  GOOGLE_MAPS_API_KEY: ${{ secrets.GOOGLE_MAPS_API_KEY }}

# Then generate .env and config.js
- name: Create .env
  run: echo "GOOGLE_MAPS_API_KEY=${{ secrets.GOOGLE_MAPS_API_KEY }}" > .env

- name: Create web/config.js
  run: echo "var GOOGLE_MAPS_API_KEY = '${{ secrets.GOOGLE_MAPS_API_KEY }}';" > web/config.js
```

## Troubleshooting

### Error: "GOOGLE_MAPS_API_KEY not found in .env file"

**Solution:** Make sure `.env` file exists in the project root and contains:
```
GOOGLE_MAPS_API_KEY=your_key_here
```

### Web: Maps not loading

**Solution:** 
1. Check `web/config.js` exists and has the key
2. Verify the key is correct
3. Check browser console for errors

### Key not working

**Solution:**
1. Verify key is active in Google Cloud Console
2. Check API restrictions (should allow your app)
3. Verify billing is enabled for the project

## Security Notes

- ✅ `.env` is in `.gitignore` (not committed)
- ✅ `web/config.js` is in `.gitignore` (not committed)
- ✅ `.env.example` and `config.js.example` are safe to commit (templates only)
- ⚠️  Never commit actual API keys to git
- ⚠️  Rotate keys if they've been exposed

