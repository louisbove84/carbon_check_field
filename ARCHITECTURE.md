# Architecture Overview 🏗️

This document describes the technical architecture of CarbonCheck Field.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Flutter Mobile App                       │
│                   (iOS + Android)                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ├──────────────────┐
                            ▼                  ▼
                    ┌───────────────┐  ┌─────────────┐
                    │  Google Maps  │  │   Service   │
                    │      SDK      │  │   Account   │
                    └───────────────┘  │    Auth     │
                                       └──────┬──────┘
                                              │
                    ┌─────────────────────────┼─────────────────┐
                    ▼                         ▼                 ▼
            ┌───────────────┐        ┌──────────────┐  ┌─────────────┐
            │ Earth Engine  │        │  Vertex AI   │  │  Sentinel-2 │
            │  REST API     │        │   Endpoint   │  │   Imagery   │
            └───────────────┘        └──────────────┘  └─────────────┘
                    │                        │
                    ▼                        ▼
            ┌───────────────┐        ┌──────────────┐
            │  NDVI Stats   │        │ Crop Type    │
            │  (17 features)│        │  Prediction  │
            └───────────────┘        └──────────────┘
```

## Layer Architecture

### 1. Presentation Layer (UI)

**Screens:**
- `home_screen.dart` - Landing page with branding
- `map_screen.dart` - Interactive polygon drawing
- `results_screen.dart` - Analysis results display

**Widgets (Reusable Components):**
- `area_display_card.dart` - Shows polygon area
- `map_instructions.dart` - User guidance overlay
- `loading_overlay.dart` - Loading state display
- `result_card.dart` - Formatted results card

### 2. Business Logic Layer (Services)

**Services:**
- `auth_service.dart` - OAuth2 token management
- `earth_engine_service.dart` - NDVI feature computation
- `vertex_ai_service.dart` - Crop prediction

### 3. Data Layer (Models)

**Models:**
- `field_data.dart` - Field polygon + geographic data
- `prediction_result.dart` - Crop prediction + CO₂ income

### 4. Utilities

- `constants.dart` - App-wide configuration
- `geo_utils.dart` - Geospatial calculations

## Data Flow

### User Journey: Drawing a Field

```
1. User taps "Draw Field on Map"
   └─> Navigate to MapScreen

2. User taps corners on satellite map
   ├─> Add point to _polygonPoints list
   ├─> Update markers (draggable)
   ├─> Update polygon shape
   └─> Calculate area in real-time (GeoUtils)

3. User taps "Analyze Field"
   ├─> Validate polygon (min/max points, size)
   ├─> Calculate centroid and bounds
   ├─> Create FieldData object
   └─> Navigate to ResultsScreen
```

### Analysis Pipeline: Computing Features

```
ResultsScreen._runAnalysis()
  │
  ├─> Step 1: Initialize AuthService
  │   └─> Load service-account.json from assets
  │   └─> Generate OAuth2 JWT token
  │   └─> Exchange for access token (cached 1hr)
  │
  ├─> Step 2: Compute NDVI Features (EarthEngineService)
  │   ├─> Build GeoJSON polygon from LatLng points
  │   ├─> Create Earth Engine expression:
  │   │   - Filter Sentinel-2 by date (2024)
  │   │   - Filter by cloud cover (<20%)
  │   │   - Compute NDVI = (NIR - Red) / (NIR + Red)
  │   │   - Reduce to statistics (mean, std, percentiles)
  │   ├─> POST to EE REST API: /computeValue
  │   └─> Parse result → 17 features:
  │       1. Basic stats: mean, std, min, max
  │       2. Percentiles: p25, p50, p75
  │       3. Temporal: early, late, change
  │       4. Location: lat, lng, elevation
  │       5. Derived: range, IQR, ratios
  │
  ├─> Step 3: Predict Crop Type (VertexAiService)
  │   ├─> Build request: {"instances": [[...17 features...]]}
  │   ├─> POST to Vertex AI endpoint
  │   ├─> Parse response: {"predictions": ["Corn"]}
  │   └─> Map crop → carbon rates
  │
  └─> Step 4: Display Results
      ├─> Show crop type + confidence
      ├─> Show field area (acres)
      ├─> Calculate CO₂ income range
      └─> Enable share functionality
```

## Key Technical Decisions

### 1. Why REST API instead of Earth Engine Dart SDK?

**Decision:** Use Earth Engine REST API directly

**Rationale:**
- No official Earth Engine Dart/Flutter package
- REST API is stable and well-documented
- Full control over requests and error handling
- Easier to debug network issues

### 2. Why Service Account instead of OAuth2 User Flow?

**Decision:** Service account with pre-authorized credentials

**Rationale:**
- No user login required (better UX)
- Consistent permissions across all app users
- Simpler auth flow (no redirect URLs)
- Works offline after initial token fetch

**Trade-off:** Private key in app (mitigated by Android/iOS app sandboxing)

### 3. Why 17 Features?

**Decision:** Compute comprehensive feature set from NDVI time series

**Rationale:**
- Model was trained on these exact features
- Captures spatial variation (mean, std, percentiles)
- Captures temporal variation (early vs late season)
- Includes location context (lat/lng, elevation)
- Provides robustness against noise

### 4. Why Client-Side Feature Computation?

**Decision:** Compute features on-device via Earth Engine API

**Rationale:**
- No backend server to maintain
- Real-time updates as user draws polygon
- Leverages Google's Earth Engine infrastructure
- Scales automatically with Earth Engine quotas

**Trade-off:** Slightly slower (5-10 sec) vs precomputed features

### 5. Why Flutter over Native iOS/Android?

**Decision:** Single Flutter codebase for both platforms

**Rationale:**
- 90% code reuse (iOS + Android)
- Faster development and iteration
- Easier maintenance (one codebase)
- Beautiful UI with Material Design
- Strong geospatial package ecosystem

## Security Considerations

### 1. Service Account Key Storage

**Current:** Bundled in `assets/service-account.json`

**Risk:** If app is decompiled, key could be extracted

**Mitigations:**
- Use API key restrictions (bundle ID / package name)
- Monitor GCP billing for abuse
- Rotate keys periodically

**Future Enhancement:** Move token generation to backend service

### 2. API Key Protection

**Current:** Hardcoded in AndroidManifest.xml and Info.plist

**Mitigations:**
- Use application restrictions (bundle ID / package name)
- Use API restrictions (limit to Maps SDK only)
- Monitor usage in Google Cloud Console

**Alternative:** Use Google Cloud Secrets Manager or Firebase Remote Config

### 3. Data Privacy

**Current:** No user data stored or transmitted (except polygon coordinates)

**Future:** If adding field history, use:
- Local SQLite encryption
- Or Firebase with security rules
- GDPR/CCPA compliance

## Performance Optimizations

### 1. Token Caching
- Cache OAuth2 token for 55 minutes (expires at 60 min)
- Reduces auth overhead on repeated API calls

### 2. Lazy Widget Building
- Use `const` constructors where possible
- Minimize rebuilds with proper state management

### 3. Debouncing Map Interactions
- Area calculation only triggers on marker drag end
- Prevents excessive computations during drawing

### 4. Polygon Simplification (Future)
- Could reduce vertex count before sending to Earth Engine
- Would speed up computation for complex polygons

## Error Handling Strategy

### 1. Network Errors
- Catch HTTP timeouts
- Display user-friendly message
- Offer retry button

### 2. Authentication Errors
- Detect 401/403 responses
- Clear cached token
- Attempt re-authentication
- Fallback to error screen with troubleshooting tips

### 3. Earth Engine Errors
- Handle "no data available" gracefully
- Suggest alternative date ranges
- Provide fallback to manual input

### 4. Validation Errors
- Prevent invalid polygons (too small, too few points)
- Show inline error messages
- Guide user to fix issues

## Testing Strategy

### 1. Unit Tests
- Test geospatial utilities (area, centroid, distance)
- Test carbon rate calculations
- Test feature vector construction

### 2. Widget Tests
- Test UI components in isolation
- Verify loading states
- Test error state displays

### 3. Integration Tests
- Test full analysis pipeline with mock data
- Test navigation flows
- Test map interactions

### 4. Manual Testing
- Test on real farms in different regions
- Verify crop predictions match ground truth
- Test edge cases (very small/large fields)

## Scalability Considerations

### Current Limitations
- Earth Engine quotas (requests per day)
- Vertex AI endpoint QPS limits
- No caching of historical analyses

### Future Enhancements
1. **Backend Service Layer**
   - Cache Earth Engine results by polygon hash
   - Batch process multiple fields
   - Store historical predictions

2. **Offline Mode**
   - Cache map tiles for offline viewing
   - Queue analyses for when network returns
   - Local feature computation fallback

3. **Multi-User Support**
   - User accounts (Firebase Auth)
   - Field library synced across devices
   - Sharing fields with team members

## Monitoring & Observability

### Recommended Tools
- Firebase Crashlytics (crash reporting)
- Firebase Performance Monitoring
- Google Analytics for Firebase (user behavior)
- Cloud Logging (backend errors)

### Key Metrics to Track
- Analysis success rate
- Average analysis duration
- Most common error types
- User retention (DAU/MAU)
- Fields analyzed per user

---

**Last Updated:** November 2024  
**Version:** 1.0.0

