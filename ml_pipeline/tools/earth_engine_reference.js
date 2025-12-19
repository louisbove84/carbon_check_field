// ============================================================
// EARTH ENGINE DATA COLLECTOR - VISUALIZATION & DEBUGGING
// ============================================================
// PURPOSE: Visualize what earth_engine_collector.py is doing
// MATCHES: Python collector logic exactly (feature extraction, sampling)
// USE: Debug data collection issues, visualize sample locations
// NO EXPORT: This script only visualizes - no BigQuery export
// ============================================================
//
// 🔧 TROUBLESHOOTING:
// If you see "API keys are not supported" error:
//   1. Hard refresh: Ctrl+Shift+R (Cmd+Shift+R on Mac)
//   2. Click your email in top-right corner → Re-authenticate
//   3. Code Editor uses OAuth2 automatically (no API keys needed)
//
// If a crop shows 0 samples:
//   - Check if the CDL code is correct (view CDL layer legend)
//   - Try a different CDL year (2023, 2022 instead of 2024)
//   - Add more county GEOIDs for that crop in the CROPS array
//   - County GEOIDs are 5-digit codes (e.g. 19169 = Story County, IA)
// ============================================================

// ---------------------------------------------------------------
// 1. CONFIGURATION - EDIT THESE!
// ---------------------------------------------------------------
var NUM_FIELDS_PER_CROP = 30;     // Number of fields per crop (increased for more diverse data)
var NUM_SAMPLES_PER_FIELD = 3;    // Number of samples per field (30*3*4 crops = 360 total samples)
var FIELD_RADIUS = 200;           // Buffer radius in meters (not used with new sampling)
var CDL_YEAR = 2024;              // Which CDL year to use
var COLLECT_OTHER = true;         // Set to false to skip non-crop sampling (matches Python collect_other flag)
var NUM_OTHER_FIELDS = NUM_FIELDS_PER_CROP;  // Number of non-crop fields to sample (match Python default)
var NON_CROP_BUFFER_METERS = 150;  // Minimum distance (meters) from crop areas for non-crop sampling
                                    // Prevents sampling near crop field edges to avoid confusion

// NOTE: Each sample point is CDL-verified to ensure it matches the target crop!
// NOTE: Using top-producing counties for fastest, most reliable sampling
// NOTE: Requests 100x samples per county to ensure we get enough valid matches
// NOTE: Current settings generate 120 total samples (10 fields * 3 samples * 4 crops)
// NOTE: Exports directly to BigQuery for immediate use in ML pipeline!
// TIP: If you get unbalanced samples (some crops have fewer), try:
//      - Adding more county GEOIDs for that crop
//      - Increase NUM_FIELDS_PER_CROP to 12 or 15

// Target crops - CDL codes from USDA Cropland Data Layer
// Using TOP 10 counties per crop for fastest sampling
var CROPS = [
  {
    name: 'Corn',
    code: 1,
    color: 'FFD300',
    counties: [
      '17113', '17019', '17093', '17161', '19099', '19113', '17095', '17031', '19155', '19133'
    ] // Top 10: McLean IL, Champaign IL, Grundy IL, etc.
  },
  {
    name: 'Soybeans',
    code: 5,
    color: '267300',
    counties: [
      '27131', '27133', '27043', '19153', '17019', '19099', '27103', '27123', '27087', '19133'
    ] // Top 10: Renville MN, Redwood MN, Faribault MN, etc.
  },
  {
    name: 'Winter_Wheat',
    code: 24,
    color: 'A57000',
    counties: [
      '20173', '20055', '20185', '20035', '40043', '20151', '20155', '20077', '20169', '20067'
    ] // Top 10: Sumner KS, Finney KS, Reno KS, Ford KS, Dewey OK, etc.
  },
  {
    name: 'Alfalfa',
    code: 36,
    color: 'FF00FF',
    counties: [
      '06037', '06029', '04013', '06065', '04027', '06019', '06059', '06041', '06071', '04015'
    ] // Top 10: Imperial CA, Kern CA, Maricopa AZ, Riverside CA, Yuma AZ, etc.
  }
];

print('═══════════════════════════════════════');
print('🌾 CROP SAMPLING PIPELINE (CDL-VERIFIED)');
print('═══════════════════════════════════════');
print('Samples per crop:', NUM_FIELDS_PER_CROP * NUM_SAMPLES_PER_FIELD);
var totalCropSamples = NUM_FIELDS_PER_CROP * NUM_SAMPLES_PER_FIELD * CROPS.length;
var totalOtherSamples = COLLECT_OTHER ? (NUM_OTHER_FIELDS * NUM_SAMPLES_PER_FIELD) : 0;
print('Total samples:', totalCropSamples + totalOtherSamples, '✅ (meets 100+ minimum!)');
print('CDL Year:', CDL_YEAR);
print('Mode: VISUALIZATION ONLY (no export)');
print('Sampling strategy: Top counties per crop, diverse counties for non-crop');
print('Verification: Samples are spatially filtered to match CDL crop type');
print('Non-crop sampling:', COLLECT_OTHER ? 'ENABLED' : 'DISABLED');
print('');

// ---------------------------------------------------------------
// 2. MAP SETUP (will be centered on 'Other' sample if available)
// ---------------------------------------------------------------
Map.setOptions('SATELLITE');

// Load CDL for background (needed for finding 'Other' samples)
var cdl = ee.Image('USDA/NASS/CDL/' + CDL_YEAR).select('cropland');

// Try to get multiple 'Other' sample locations to center the map and show markers
// This helps visualize what non-crop areas look like
var previewOtherSamples = ee.FeatureCollection([]);
if (COLLECT_OTHER) {
  print('📍 Finding "Other" sample locations to preview on map...');
  try {
    // Create non-crop mask with buffer exclusion
    var cropCodes = CROPS.map(function(crop) { return crop.code; });
    
    // Create crop mask and buffer
    var cropMask = ee.Image.constant(0);
    for (var i = 0; i < cropCodes.length; i++) {
      cropMask = cropMask.add(cdl.eq(cropCodes[i]));
    }
    cropMask = cropMask.gt(0);
    var bufferDistanceMeters = NON_CROP_BUFFER_METERS;
    var cropBuffer = cropMask.focalMax({
      radius: bufferDistanceMeters / 30,
      units: 'pixels'
    });
    
    // Create non-crop mask excluding crops and buffer zones
    var nonCropBinary = ee.Image.constant(1);
    for (var i = 0; i < cropCodes.length; i++) {
      nonCropBinary = nonCropBinary.multiply(cdl.neq(cropCodes[i]));
    }
    nonCropBinary = nonCropBinary.multiply(cropBuffer.eq(0));
    var nonCropMask = cdl.updateMask(nonCropBinary.eq(1));
    
    // Sample from multiple counties to get diverse non-crop areas
    var counties = ee.FeatureCollection('TIGER/2018/Counties');
    var previewSamples = ee.FeatureCollection([]);
    
    // Get samples from first 3 counties to show variety
    for (var i = 0; i < Math.min(3, CROPS[0].counties.length); i++) {
      var countyGEOID = CROPS[0].counties[i];
      var countyRegion = counties.filter(ee.Filter.eq('GEOID', countyGEOID)).geometry();
      
      // Sample 5 pixels per county for preview
      var countySamples = nonCropMask.sample({
        region: countyRegion,
        scale: 30,
        numPixels: 5000,
        seed: 999 + i,
        geometries: true,
        tileScale: 16
      }).limit(5);
      
      previewSamples = previewSamples.merge(countySamples);
    }
    
    previewOtherSamples = previewSamples;
    var previewCount = previewSamples.size().getInfo();
    
    if (previewCount > 0) {
      // Get first sample coordinates to center map
      var firstSample = previewSamples.first();
      var coords = firstSample.geometry().coordinates();
      var lon = coords.get(0).getInfo();
      var lat = coords.get(1).getInfo();
      print('✅ Found', previewCount, '"Other" preview samples');
      print('   Centering map on first sample:', lon.toFixed(6) + ', ' + lat.toFixed(6));
      Map.setCenter(lon, lat, 12);  // Zoom level 12 to see multiple markers
      
      // Add preview markers immediately (before main collection)
      Map.addLayer(
        previewSamples.style({
          color: 'FF0000',  // Red markers for preview
          pointSize: 15,
          pointShape: 'diamond',  // Diamond shape to distinguish from final samples
          width: 4,
          fillColor: 'FF0000'
        }),
        {},
        '🔍 Preview: Other Sample Locations (' + previewCount + ')',
        true
      );
    } else {
      print('⚠️  Could not find "Other" preview samples, using default center');
      Map.setCenter(-88.43055692640955, 44.409572652286904, 6);
    }
  } catch (e) {
    print('⚠️  Could not find "Other" sample location, using default center');
    print('   Error:', e.toString());
    Map.setCenter(-88.43055692640955, 44.409572652286904, 6);  // Default center
  }
} else {
  Map.setCenter(-88.43055692640955, 44.409572652286904, 6);  // Default center
}

// Create a simplified visualization (only show our target crops)
// Use expression to remap values: if crop code matches, map to 1-4, else mask out
var cdlVis = cdl.expression(
  '(cropland == 1) ? 1 : ' +  // Corn -> 1
  '(cropland == 5) ? 2 : ' +  // Soybeans -> 2
  '(cropland == 24) ? 3 : ' + // Winter Wheat -> 3
  '(cropland == 36) ? 4 : 0', // Alfalfa -> 4, else 0
  {'cropland': cdl}
).selfMask();

Map.addLayer(cdlVis, {
  min: 1,
  max: 4,
  palette: ['FFD300', '267300', 'A57000', 'FF00FF']  // Corn, Soy, Wheat, Alfalfa
}, 'CDL ' + CDL_YEAR + ' (Target Crops)', true, 0.7);

// Add buffer zone visualization if collecting 'Other' samples
if (COLLECT_OTHER) {
  // Create crop mask and buffer for visualization
  var cropCodes = CROPS.map(function(crop) { return crop.code; });
  var cropMask = ee.Image.constant(0);
  for (var i = 0; i < cropCodes.length; i++) {
    cropMask = cropMask.add(cdl.eq(cropCodes[i]));
  }
  cropMask = cropMask.gt(0);
  var cropBuffer = cropMask.focalMax({
    radius: NON_CROP_BUFFER_METERS / 30,
    units: 'pixels'
  });
  
  // Show buffer zone (areas excluded from non-crop sampling)
  var bufferZone = cropBuffer.subtract(cropMask);  // Buffer minus crops = just the buffer
  Map.addLayer(
    bufferZone.selfMask(),
    {palette: ['FFFF00'], min: 0, max: 1},  // Yellow for buffer zone
    '🛡️ Crop Buffer Zone (' + NON_CROP_BUFFER_METERS + 'm) - Excluded from "Other"',
    false  // Hidden by default, can be toggled
  );
  
  print('💡 TIP: Toggle "Crop Buffer Zone" layer to see areas excluded from non-crop sampling');
}

// ---------------------------------------------------------------
// 3. CREATE LEGEND
// ---------------------------------------------------------------
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px',
    backgroundColor: 'white',
    border: '2px solid black'
  }
});

var legendTitle = ui.Label({
  value: '🗺️ SAMPLE LOCATIONS',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '0 0 8px 0'}
});
legend.add(legendTitle);

legend.add(ui.Label({
  value: '▼ = Crop sample (bottom tip = exact location)',
  style: {fontSize: '11px', margin: '0 0 4px 0', color: '666'}
}));

legend.add(ui.Label({
  value: '● = Other (non-crop) sample',
  style: {fontSize: '11px', margin: '0 0 4px 0', color: '666'}
}));

legend.add(ui.Label({
  value: '🔴 = Preview Other locations',
  style: {fontSize: '11px', margin: '0 0 8px 0', color: '666'}
}));

function addLegendRow(color, name) {
  var colorBox = ui.Label({
    style: {
      backgroundColor: '#' + color,
      padding: '10px',
      margin: '0 8px 0 0',
      border: '1px solid black'
    },
    value: ''
  });
  
  var label = ui.Label({
    value: name,
    style: {margin: '0', fontSize: '13px'}
  });
  
  var panel = ui.Panel({
    widgets: [colorBox, label],
    layout: ui.Panel.Layout.Flow('horizontal'),
    style: {margin: '4px 0'}
  });
  
  legend.add(panel);
}

CROPS.forEach(function(crop) {
  addLegendRow(crop.color, crop.name);
});

// Add 'Other' category to legend if enabled
if (COLLECT_OTHER) {
  addLegendRow('808080', 'Other (Non-Crop)');
}

Map.add(legend);

// ---------------------------------------------------------------
// 4. COUNTY BOUNDARIES (for sampling regions)
// ---------------------------------------------------------------
var counties = ee.FeatureCollection('TIGER/2018/Counties');

function getCountyGeometry(countyGEOIDs) {
  return counties.filter(ee.Filter.inList('GEOID', countyGEOIDs)).geometry();
}

// ---------------------------------------------------------------
// 5. FEATURE EXTRACTION FUNCTION
// ---------------------------------------------------------------
function extractFeatures(point) {
  // Define growing season
  var startDate = ee.Date(CDL_YEAR + '-04-15');
  var endDate = ee.Date(CDL_YEAR + '-09-01');
  
  // Get Sentinel-2 data (MATCHES Python: 20% cloud filter)
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(startDate, endDate)
    .filterBounds(point)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));  // Changed from 30% to 20% to match Python
  
  // Calculate NDVI (MATCHES Python: map then select)
  var ndviCollection = s2.map(function(img) {
    return img.normalizedDifference(['B8', 'B4']).rename('NDVI')
      .copyProperties(img, ['system:time_start']);
  }).select('NDVI');
  
  // Compute statistics (MATCHES Python: use median composite, then reduceRegion)
  var ndviComposite = ndviCollection.median();
  
  var ndviStats = ndviComposite.reduceRegion({
    reducer: ee.Reducer.mean()
      .combine(ee.Reducer.stdDev(), '', true)
      .combine(ee.Reducer.min(), '', true)
      .combine(ee.Reducer.max(), '', true)
      .combine(ee.Reducer.percentile([25, 50, 75]), '', true),
    geometry: point.buffer(50),
    scale: 10,
    maxPixels: 1e9
  });
  
  // Early season (Apr-May) - MATCHES Python: use median, not mean
  var earlyNDVI = ndviCollection
    .filterDate(CDL_YEAR + '-04-15', CDL_YEAR + '-06-01')
    .median()
    .reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: point.buffer(50),
      scale: 10,
      maxPixels: 1e9
    });
  
  // Late season (Jul-Aug) - MATCHES Python: use median, not mean
  var lateNDVI = ndviCollection
    .filterDate(CDL_YEAR + '-07-01', CDL_YEAR + '-08-31')
    .median()
    .reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: point.buffer(50),
      scale: 10,
      maxPixels: 1e9
    });
  
  // REMOVED: elevation and lat/lon (MATCHES Python - these features removed from model)
  
  return ee.Feature(point, {
    ndvi_mean: ndviStats.get('NDVI_mean'),
    ndvi_std: ndviStats.get('NDVI_stdDev'),
    ndvi_min: ndviStats.get('NDVI_min'),
    ndvi_max: ndviStats.get('NDVI_max'),
    ndvi_p25: ndviStats.get('NDVI_p25'),
    ndvi_p50: ndviStats.get('NDVI_p50'),
    ndvi_p75: ndviStats.get('NDVI_p75'),
    ndvi_early: earlyNDVI.get('NDVI'),
    ndvi_late: lateNDVI.get('NDVI')
    // REMOVED: elevation_m, longitude, latitude (match Python feature set)
  });
}

// ---------------------------------------------------------------
// 6. SAMPLING FUNCTION WITH CDL VERIFICATION (SERVER-SIDE)
// ---------------------------------------------------------------
function sampleCropFields(cropInfo) {
  print('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  print('🌱', cropInfo.name.toUpperCase());
  print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  // Target number of samples
  var targetSamples = NUM_FIELDS_PER_CROP * NUM_SAMPLES_PER_FIELD;
  
  print('  🎯 Target:', targetSamples, 'samples for CDL code', cropInfo.code);
  print('  📡 Strategy: Request 100x per county to ensure valid matches');
  
  // SMART APPROACH: Try counties one by one until we have enough samples
  // NOTE: We request 100x samples per county because Earth Engine's sample()
  // function may not return all requested samples (sparse data, edge effects, etc.)
  var allSamples = ee.FeatureCollection([]);
  var samplesCollected = 0;
  var countiesUsed = 0;
  
  // Mask CDL to only show pixels of this crop
  var cropMask = cdl.select('cropland').updateMask(cdl.eq(cropInfo.code));
  
  // Try each county until we have enough samples
  for (var i = 0; i < cropInfo.counties.length && samplesCollected < targetSamples; i++) {
    var countyGEOID = cropInfo.counties[i];
    var countyRegion = getCountyGeometry([countyGEOID]);
    
    // How many more samples do we need?
    var samplesNeeded = targetSamples - samplesCollected;
    
    // Sample from this county (request 100x what we need to ensure we get matches!)
    var countySamples = cropMask.sample({
      region: countyRegion,
      scale: 30,
      numPixels: samplesNeeded * 100,  // 100x buffer for reliable sampling
      seed: 42 + cropInfo.code + i,
      geometries: true,
      tileScale: 16
    });
    
    var countyCount = countySamples.size().getInfo();
    
    if (countyCount > 0) {
      // Limit to what we need
      var samplesToAdd = countySamples.limit(samplesNeeded);
      allSamples = allSamples.merge(samplesToAdd);
      samplesCollected += Math.min(countyCount, samplesNeeded);
      countiesUsed++;
      
      if (i < 3) {  // Only print first 3 to avoid spam
        print('  📦 County', (i+1), '(GEOID:', countyGEOID + '):', 
              Math.min(countyCount, samplesNeeded), 'samples');
      }
    }
  }
  
  print('  ✅ Total collected:', samplesCollected, 'samples from', countiesUsed, 'counties');
  
  var finalSamples = allSamples;
  
  // Add crop info to each sample
  finalSamples = finalSamples.map(function(feature) {
    return feature.set({
      'crop': cropInfo.name,
      'crop_code': cropInfo.code
      // Note: cdl_code removed to match existing BigQuery schema
    });
  });
  
  // Extract NDVI features for each sample
  var allSamples = finalSamples.map(function(feature) {
    var point = feature.geometry();
    var sampleWithFeatures = extractFeatures(point);
    
    // Combine with crop labels
    return sampleWithFeatures.copyProperties(feature);
  });
  
  var sampleCount = finalSamples.size().getInfo();
  
  // Check if we got any samples at all
  if (sampleCount === 0) {
    print('  ❌ ERROR: Found 0 samples for', cropInfo.name);
    print('      This crop might not exist in', CDL_YEAR, 'CDL data!');
    print('      💡 SOLUTIONS:');
    print('         1. Check if CDL code', cropInfo.code, 'is correct');
    print('         2. Try a different CDL year (2023, 2022)');
    print('         3. Remove this crop from the CROPS array');
    return ee.FeatureCollection([]);  // Return empty collection
  }
  
  if (sampleCount < targetSamples) {
    print('  ⚠️  Got', sampleCount, '/', targetSamples, 'samples');
    print('      (Checked all', cropInfo.counties.length, 'counties - no more available)');
  } else {
    print('  ✅ SUCCESS: Got full quota of', targetSamples, 'samples!');
  }
  
  // Add markers to map ONLY if we have samples (upside-down triangles = map pins!)
  if (sampleCount > 0) {
    Map.addLayer(
      finalSamples.style({
        color: cropInfo.color,
        pointSize: 12,
        pointShape: 'triangle_down',  // Upside-down triangle (point at bottom)
        width: 3,
        fillColor: cropInfo.color
      }),
      {},
      cropInfo.name + ' Samples (' + sampleCount + ')',
      true
    );
  }
  
  return allSamples;
}

// ---------------------------------------------------------------
// 7. NON-CROP SAMPLING FUNCTION (MATCHES Python sample_non_crop_areas)
// ---------------------------------------------------------------
function sampleNonCropAreas(numFields, numSamplesPerField) {
  print('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  print('🌍 OTHER (NON-CROP AREAS)');
  print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  var targetSamples = numFields * numSamplesPerField;
  print('  🎯 Target:', targetSamples, 'samples for non-crop areas');
  print('  📡 Strategy: Exclude all crop codes, sample from remaining areas');
  
  // Get all crop codes to exclude
  var cropCodes = CROPS.map(function(crop) { return crop.code; });
  print('  🚫 Excluding crop codes:', cropCodes);
  
  // Load CDL
  var cdl = ee.Image('USDA/NASS/CDL/' + CDL_YEAR).select('cropland');
  
  // Create mask for crop areas (all crop codes combined)
  var cropMask = ee.Image.constant(0);
  for (var i = 0; i < cropCodes.length; i++) {
    cropMask = cropMask.add(cdl.eq(cropCodes[i]));
  }
  cropMask = cropMask.gt(0);  // 1 where any crop exists, 0 elsewhere
  
  // Create buffer around crop areas to exclude edge cases
  // Buffer distance in meters - ensures non-crop samples are clearly separated from crops
  var bufferDistanceMeters = NON_CROP_BUFFER_METERS;
  var cropBuffer = cropMask.focalMax({
    radius: bufferDistanceMeters / 30,
    units: 'pixels'
  });
  
  // Create mask for non-crop areas (MATCHES Python logic)
  // Build binary mask: 1 if pixel is NOT any crop code AND not in buffer zone
  var nonCropBinary = ee.Image.constant(1);
  for (var i = 0; i < cropCodes.length; i++) {
    nonCropBinary = nonCropBinary.multiply(cdl.neq(cropCodes[i]));
  }
  // Exclude buffered areas around crops
  nonCropBinary = nonCropBinary.multiply(cropBuffer.eq(0));
  
  // Apply mask: only pixels where binary = 1 (not any crop code and not in buffer)
  var nonCropMask = cdl.updateMask(nonCropBinary.eq(1));
  
  print('  🛡️  Added ' + bufferDistanceMeters + 'm buffer around crop areas to avoid edge cases');
  
  // Get unique counties from all crops for diversity
  var allCounties = [];
  CROPS.forEach(function(crop) {
    crop.counties.forEach(function(county) {
      if (allCounties.indexOf(county) === -1) {
        allCounties.push(county);
      }
    });
  });
  var uniqueCounties = allCounties.slice(0, 10);  // Use up to 10 counties
  print('  📍 Sampling from', uniqueCounties.length, 'counties');
  
  var allSamples = ee.FeatureCollection([]);
  var samplesCollected = 0;
  
  for (var i = 0; i < uniqueCounties.length && samplesCollected < targetSamples; i++) {
    var countyGEOID = uniqueCounties[i];
    var countyRegion = getCountyGeometry([countyGEOID]);
    
    var samplesNeeded = targetSamples - samplesCollected;
    
    // Sample from non-crop areas (1000x buffer - non-crop areas can be sparse)
    var countySamples = nonCropMask.sample({
      region: countyRegion,
      scale: 30,
      numPixels: samplesNeeded * 1000,  // 1000x buffer (matches Python)
      seed: 999 + i,  // Different seed for 'Other' category
      geometries: true,
      tileScale: 16
    });
    
    var countyCount = countySamples.size().getInfo();
    
    if (countyCount > 0) {
      var samplesToAdd = countySamples.limit(samplesNeeded);
      allSamples = allSamples.merge(samplesToAdd);
      samplesCollected += Math.min(countyCount, samplesNeeded);
      
      if (i < 3) {  // Only print first 3
        print('  📦 County', (i+1), '(GEOID:', countyGEOID + '):', 
              Math.min(countyCount, samplesNeeded), 'samples');
      }
    }
  }
  
  print('  ✅ Total collected:', samplesCollected, 'samples');
  
  // Add 'Other' category info
  var finalSamples = allSamples.map(function(feature) {
    return feature.set({
      'crop': 'Other',
      'crop_code': 0
    });
  });
  
  // Extract NDVI features for each sample
  var allSamplesWithFeatures = finalSamples.map(function(feature) {
    var point = feature.geometry();
    var sampleWithFeatures = extractFeatures(point);
    return sampleWithFeatures.copyProperties(feature);
  });
  
  var sampleCount = finalSamples.size().getInfo();
  
  if (sampleCount === 0) {
    print('  ❌ ERROR: Found 0 samples for non-crop areas');
    print('      💡 SOLUTIONS:');
    print('         1. Check if counties have non-crop areas');
    print('         2. Increase numPixels buffer (currently 1000x)');
    print('         3. Try different counties');
    return ee.FeatureCollection([]);
  }
  
  if (sampleCount < targetSamples) {
    print('  ⚠️  Got', sampleCount, '/', targetSamples, 'samples');
  } else {
    print('  ✅ SUCCESS: Got full quota of', targetSamples, 'samples!');
  }
  
  // Add visualization (gray color for 'Other' with larger, more visible markers)
  if (sampleCount > 0) {
    // Add main 'Other' samples layer with gray markers
    Map.addLayer(
      finalSamples.style({
        color: '000000',  // Black border for visibility
        pointSize: 18,    // Larger size to see clearly
        pointShape: 'circle',  // Circle shape for 'Other'
        width: 3,
        fillColor: '808080'  // Gray fill
      }),
      {},
      'Other (Non-Crop) Samples (' + sampleCount + ')',
      true
    );
    
    // Also add a layer with just outlines for better visibility
    Map.addLayer(
      finalSamples.style({
        color: 'FF0000',  // Red outline
        pointSize: 20,
        pointShape: 'circle',
        width: 2,
        fillColor: '00000000'  // Transparent fill
      }),
      {},
      'Other Samples (Outlines)',
      false  // Hidden by default, can be toggled
    );
    
    print('  📍 Added', sampleCount, '"Other" sample markers to map');
    print('   • Gray circles = Non-crop sample locations');
    print('   • Click markers to see coordinates');
    print('   • Toggle "Other Samples (Outlines)" layer for red outlines');
  }
  
  return allSamplesWithFeatures;
}

// ---------------------------------------------------------------
// 8. PROCESS ALL CROPS AND TRACK DISTRIBUTION
// ---------------------------------------------------------------
var allTrainingData = ee.FeatureCollection([]);
var sampleCounts = {};

CROPS.forEach(function(crop) {
  var cropSamples = sampleCropFields(crop);
  allTrainingData = allTrainingData.merge(cropSamples);
  
  // Track how many samples we got for this crop
  var count = cropSamples.size().getInfo();
  sampleCounts[crop.name] = count;
});

// Add 'Other' category if enabled (already defined in config section above)
if (COLLECT_OTHER) {
  var otherSamples = sampleNonCropAreas(NUM_OTHER_FIELDS, NUM_SAMPLES_PER_FIELD);
  allTrainingData = allTrainingData.merge(otherSamples);
  var otherCount = otherSamples.size().getInfo();
  sampleCounts['Other'] = otherCount;
}

// ---------------------------------------------------------------
// 9. DISPLAY SUMMARY TABLE
// ---------------------------------------------------------------
print('\n═══════════════════════════════════════');
print('📊 SUMMARY - SAMPLE DISTRIBUTION');
print('═══════════════════════════════════════');

var totalSamples = allTrainingData.size().getInfo();
var targetPerCrop = NUM_FIELDS_PER_CROP * NUM_SAMPLES_PER_FIELD;
var isBalanced = true;

print('Total samples:', totalSamples);
print('Target per crop:', targetPerCrop);
print('');
print('Category        | Samples | Status');
print('─────────────────────────────────────');

// Show all crops
CROPS.forEach(function(crop) {
  var count = sampleCounts[crop.name] || 0;
  var status = '✅';
  
  if (count === 0) {
    status = '❌ FAILED';
    isBalanced = false;
  } else if (count < targetPerCrop) {
    status = '⚠️  LOW';
    isBalanced = false;
  }
  
  var name = (crop.name + '               ').substring(0, 14);
  var countStr = ('       ' + count).slice(-7);
  print(name + '|' + countStr + ' | ' + status);
});

// Show 'Other' category if collected
if (COLLECT_OTHER && sampleCounts['Other'] !== undefined) {
  var otherCount = sampleCounts['Other'] || 0;
  var otherStatus = '✅';
  var otherTarget = NUM_OTHER_FIELDS * NUM_SAMPLES_PER_FIELD;
  
  if (otherCount === 0) {
    otherStatus = '❌ FAILED';
    isBalanced = false;
  } else if (otherCount < otherTarget) {
    otherStatus = '⚠️  LOW';
    isBalanced = false;
  }
  
  var otherName = ('Other            ').substring(0, 14);
  var otherCountStr = ('       ' + otherCount).slice(-7);
  print(otherName + '|' + otherCountStr + ' | ' + otherStatus);
}

print('');
if (isBalanced) {
  print('✅ BALANCED: All crops have equal samples!');
} else {
  print('⚠️  UNBALANCED: Some crops need more samples');
  print('');
  print('💡 SOLUTIONS:');
  print('  1. Remove failed crops (0 samples) from CROPS array');
  print('  2. Reduce NUM_SAMPLES_PER_FIELD (e.g., 3 instead of 5)');
  print('  3. Check CDL codes are correct for rare crops');
  print('  4. Add more county GEOIDs for low-sample crops in CROPS array');
}

print('');
print('═══════════════════════════════════════');
print('🗺️ VISUALIZATION GUIDE');
print('═══════════════════════════════════════');
print('Now check the map:');
print('  • CDL layer shows actual crop types');
print('  • Upside-down triangles (▼) show your sample points');
print('  • Bottom tip of each triangle = exact sample location');
print('  • Every sample is CDL-verified to be the correct crop/type');
print('  • Gray triangles = "Other" (non-crop) samples');
print('');

print('═══════════════════════════════════════');
print('🔍 DEBUGGING INFORMATION');
print('═══════════════════════════════════════');
print('This script matches earth_engine_collector.py exactly:');
print('  ✅ Feature extraction: median composite, 20% cloud filter');
print('  ✅ Sampling: same CDL verification logic');
print('  ✅ Non-crop sampling: binary mask excluding all crop codes');
print('  ✅ Feature set: 9 NDVI features (no elevation/lat/lon)');
print('');
print('Use this script to:');
print('  1. Visualize what the Python collector is doing');
print('  2. Debug why non-crop sampling might fail');
print('  3. Verify sample locations and distributions');
print('  4. Test different counties or CDL years');
print('');
print('⚠️  NO EXPORT: This script only visualizes - no BigQuery export!');
print('   Use earth_engine_collector.py for actual data collection.');
print('');

print('═══════════════════════════════════════');
print('🐛 TROUBLESHOOTING');
print('═══════════════════════════════════════');
print('  • Auth error? Hard refresh (Ctrl+Shift+R)');
print('  • 0 samples for a crop? Check CDL code or try different year');
print('  • 0 samples for "Other"? Non-crop areas might be sparse in these counties');
print('  • Unbalanced? All categories should have', targetPerCrop, 'samples');
print('  • Timeout? Reduce NUM_FIELDS_PER_CROP to 10');
print('  • Map layer error? Means that category has 0 samples');
print('  • Non-crop mask issue? Check if counties have enough non-crop pixels');
print('');

print('═══════════════════════════════════════');
print('📊 FEATURE SET (9 features)');
print('═══════════════════════════════════════');
print('  • ndvi_mean, ndvi_std, ndvi_min, ndvi_max');
print('  • ndvi_p25, ndvi_p50, ndvi_p75');
print('  • ndvi_early, ndvi_late');
print('  • REMOVED: elevation_m, longitude, latitude');
print('  • (Matches Python feature set exactly)');
print('');
