# 🧪 Testing Guide for CarbonCheck Field

## Prerequisites

- ✅ Backend deployed: https://carboncheck-field-api-6by67xpgga-uc.a.run.app
- ✅ Firebase configured (iOS + Android)
- ✅ Flutter dependencies ready

---

## Quick Start Test

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field
flutter pub get
cd ios && pod install && cd ..
flutter run
```

---

## Test Scenarios

### Scenario 1: Iowa Cornfield (Real Data) 🌽

**Location:** Des Moines, Iowa (41.88°N, 93.09°W)

**Steps:**
1. Launch app
2. Tap "Draw Field on Map"
3. Zoom to Iowa (or search for Des Moines)
4. Tap these 4 corners to draw polygon:
   - Corner 1: (41.878, -93.097)
   - Corner 2: (41.880, -93.097)
   - Corner 3: (41.880, -93.094)
   - Corner 4: (41.878, -93.094)
5. Verify area shows ~47 acres
6. Tap "Analyze Field"
7. Wait 10-30 seconds

**Expected Result:**
- Crop: "Corn" (90-98% confidence)
- Area: 47.2 acres
- Income: $566-$850/year

---

### Scenario 2: California Alfalfa Field 🌾

**Location:** Central Valley, California

**Steps:**
1. Search for: "36.5°N, 119.8°W"
2. Draw polygon around visible agricultural field
3. Analyze

**Expected Result:**
- Crop: "Alfalfa" or "Corn"
- Higher CO₂ income for Alfalfa ($18-25/acre)

---

### Scenario 3: Small Test Field (Edge Case)

**Purpose:** Test minimum size validation

**Steps:**
1. Draw a very small polygon (< 0.1 acres)
2. Tap "Analyze Field"

**Expected Result:**
- Error: "Field is too small (minimum 0.1 acres)"

---

### Scenario 4: Large Field (Performance Test)

**Purpose:** Test backend performance with larger area

**Steps:**
1. Draw a large field (200+ acres)
2. Analyze

**Expected Result:**
- Takes 15-40 seconds (larger area = more data)
- Results still display correctly

---

## Verification Checklist

### App Startup
- [ ] App launches in < 3 seconds
- [ ] No Firebase errors in console
- [ ] Home screen displays correctly

### Map Functionality
- [ ] Satellite map loads
- [ ] Can zoom in/out
- [ ] Can pan around
- [ ] Current location works (if permissions granted)

### Drawing Polygon
- [ ] Tapping creates green markers
- [ ] Markers are draggable
- [ ] Polygon fills in green
- [ ] Area updates in real-time
- [ ] Can draw 3-50 points

### Analysis
- [ ] "Analyze Field" button appears
- [ ] Loading screen shows progress
- [ ] Loading messages update
- [ ] Takes 10-30 seconds for typical field
- [ ] No timeout errors

### Results Display
- [ ] Crop type appears
- [ ] Confidence percentage shows
- [ ] Area in acres displays
- [ ] CO₂ income range shows
- [ ] Average income displays
- [ ] Share button works
- [ ] "New Field" button returns to map

---

## Expected Timings

| Action | Expected Time |
|--------|---------------|
| App launch | 2-3 seconds |
| Map load | 3-5 seconds |
| Draw polygon | Instant |
| Firebase auth | 1-2 seconds |
| Backend analysis | 10-30 seconds |
| Results display | Instant |

---

## Testing Backend Directly

### Test Health Endpoint

```bash
curl https://carboncheck-field-api-6by67xpgga-uc.a.run.app/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "earth_engine": "initialized",
  "timestamp": "2024-11-21T..."
}
```

### Test with curl (Advanced)

You'll need a Firebase token, but you can test the backend structure:

```bash
# This will fail with 401 (expected - needs Firebase token)
curl -X POST https://carboncheck-field-api-6by67xpgga-uc.a.run.app/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "polygon": [
      {"lat": 41.878, "lng": -93.097},
      {"lat": 41.880, "lng": -93.097},
      {"lat": 41.880, "lng": -93.094},
      {"lat": 41.878, "lng": -93.094}
    ],
    "year": 2024
  }'
```

**Expected:** 401 error (authentication required) ✅ This is correct!

---

## Debug Console Output

### Successful Analysis Flow

Look for these logs in Flutter console:

```
✅ Firebase initialized successfully
✅ Signed in anonymously: uid:abc123...
✅ Backend connected: 200
✅ Analysis complete: Corn (0.98)
```

### Error Cases

**Backend unavailable:**
```
❌ Network error: Failed to connect
```

**Firebase error:**
```
❌ Firebase initialization failed
```

**Authentication error:**
```
❌ Token verification failed: 401
```

---

## View Backend Logs (Real-time)

While testing, open another terminal and watch backend logs:

```bash
gcloud run logs tail carboncheck-field-api --region us-central1
```

You should see:
```
✅ Firebase initialized
Computing NDVI features for 47.1 acre field...
Predicting crop type...
✅ Analysis complete: Corn (98%), $708/year
```

---

## Troubleshooting Tests

### "Map not loading"

**Check:**
```bash
grep -A 1 "GMSApiKey" ios/Runner/Info.plist
```

**Fix:** Ensure Google Maps API key is present

### "Backend unavailable"

**Check:**
```bash
curl https://carboncheck-field-api-6by67xpgga-uc.a.run.app/health
```

**Fix:** Backend may be cold-starting (wait 30 sec and retry)

### "Authentication failed"

**Check Firebase:**
1. Firebase Console → Authentication → Sign-in methods
2. Ensure "Anonymous" is ENABLED

### "Analysis takes too long"

**Normal behavior:**
- First request: 30-40 seconds (cold start)
- Subsequent requests: 10-20 seconds

**Check backend logs:**
```bash
gcloud run logs read carboncheck-field-api --region us-central1 --limit 20
```

---

## Performance Benchmarks

### Target Performance
- ✅ App launch: < 3 sec
- ✅ Map load: < 5 sec
- ✅ Analysis: < 30 sec (typical field)
- ✅ Results render: instant

### Cloud Run Metrics

View in Google Cloud Console:
1. Go to: https://console.cloud.google.com/run
2. Click: carboncheck-field-api
3. Tab: Metrics

**Expected:**
- Request latency: 15-25 seconds (p50)
- Memory usage: 500MB-1GB
- CPU utilization: 20-40%

---

## Test Coverage Matrix

| Feature | Test Case | Expected Result | Status |
|---------|-----------|-----------------|--------|
| Home Screen | App opens | Green/blue gradient | ⏹ |
| Map Loading | Tap button | Satellite view | ⏹ |
| Draw Polygon | Tap 4 corners | Green polygon | ⏹ |
| Area Calc | Draw field | Shows acres | ⏹ |
| Firebase Auth | Analyze field | No errors | ⏹ |
| Backend Call | Wait for results | 200 response | ⏹ |
| Crop Prediction | Iowa field | "Corn" result | ⏹ |
| CO₂ Calculation | Check income | $500-900/year | ⏹ |
| Share Feature | Tap share | System share sheet | ⏹ |
| Error Handling | Too small field | Error message | ⏹ |

---

## Success Criteria

Your app is working correctly if:

✅ All 3 screens load without errors  
✅ Can draw and edit polygon  
✅ Analysis completes in < 40 seconds  
✅ Results show crop + confidence + income  
✅ No authentication errors  
✅ Backend returns valid data  
✅ Share functionality works  

---

## Next Steps After Testing

### If Everything Works ✅
1. Test with real farm fields in your area
2. Try different crops (corn vs soy vs wheat)
3. Test edge cases (very small, very large fields)
4. Share with farmers for feedback

### If Issues Found ❌
1. Check logs: `flutter logs`
2. Check backend: `gcloud run logs tail...`
3. Verify Firebase config
4. Test backend health endpoint

---

## Real-World Testing Tips

1. **Test in Different Regions**
   - Iowa: Corn/Soybeans
   - California: Alfalfa/Cotton
   - Kansas: Winter Wheat
   - Each should predict correctly!

2. **Test Different Field Sizes**
   - Small: 5-10 acres (fast)
   - Medium: 50-100 acres (normal)
   - Large: 300+ acres (slower)

3. **Test Edge Cases**
   - Draw polygon with 3 points (minimum)
   - Draw polygon with 20+ points (complex shape)
   - Draw tiny field (should error)
   - Draw huge field (should still work)

4. **Test Network Conditions**
   - Good WiFi (fast)
   - Mobile data (slower)
   - Poor connection (should show error gracefully)

---

## Test Report Template

After testing, note:

```
Date: ___________
Device: iOS/Android
Version: ___________

✅ Tests Passed: ___/10
❌ Tests Failed: ___/10

Notes:
- App launch time: ___ seconds
- Analysis time: ___ seconds
- Predicted crop: ___________
- Confidence: ___%
- Issues found: ___________
```

---

**Ready to test? Run the app and follow Scenario 1!** 🚀

