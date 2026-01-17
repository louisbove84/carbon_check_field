## Running Locally

### Prerequisites

- Flutter SDK 3.0+
- Firebase project configured
- Google Maps API key

### App (Web)

```bash
flutter run -d chrome --web-port=8080
```

If you need a local Maps key, create `web/config.local.js`:

```js
var GOOGLE_MAPS_API_KEY = 'REPLACE_WITH_YOUR_KEY';
```

### App (Android / iOS)

```bash
flutter run -d android
flutter run -d ios
```

### Backend (FastAPI)

```bash
cd backend
uvicorn app:app --reload
```

### Test Deployed Backend

```bash
curl https://your-backend-url.run.app/health
```
