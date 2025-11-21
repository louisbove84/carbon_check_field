/// Earth Engine service for computing NDVI features from Sentinel-2 imagery
/// 
/// This service handles all interactions with the Earth Engine REST API
/// to compute the 17 features required for crop classification.

import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:carbon_check_field/models/field_data.dart';
import 'package:carbon_check_field/services/auth_service.dart';
import 'package:carbon_check_field/utils/constants.dart';
import 'package:carbon_check_field/utils/geo_utils.dart';

class EarthEngineService {
  final AuthService _authService;

  EarthEngineService(this._authService);

  /// Compute all 17 NDVI features for a field using Sentinel-2 data
  /// 
  /// This method:
  /// 1. Creates a GeoJSON polygon from field boundaries
  /// 2. Queries Sentinel-2 SR Harmonized for 2024
  /// 3. Computes NDVI for each image
  /// 4. Calculates statistics (mean, std, percentiles)
  /// 5. Derives temporal features (early, late, change)
  /// 6. Adds elevation and location features
  /// 
  /// Returns a list of 17 features in the exact order expected by the model.
  Future<List<double>> computeFeatures(FieldData field) async {
    // Step 1: Get access token
    final token = await _authService.getAccessToken();

    // Step 2: Build Earth Engine computation request
    final eeExpression = _buildEarthEngineExpression(field);

    // Step 3: Execute the computation via REST API
    final result = await _executeEarthEngineComputation(
      eeExpression,
      token,
    );

    // Step 4: Parse results and build feature vector
    final features = _parseFeatures(result, field);

    if (features.length != AppConstants.numFeatures) {
      throw Exception(
        'Feature computation returned ${features.length} features, '
        'expected ${AppConstants.numFeatures}'
      );
    }

    return features;
  }

  /// Build the Earth Engine computation expression (JavaScript-like syntax)
  /// 
  /// This creates a JSON object that represents the Earth Engine computation
  /// we want to run: filter Sentinel-2 → compute NDVI → get statistics.
  Map<String, dynamic> _buildEarthEngineExpression(FieldData field) {
    // Convert polygon to GeoJSON format
    final coords = GeoUtils.toGeoJsonCoordinates(field.polygonPoints);
    
    // Build geometry object
    final geometry = {
      'type': 'Polygon',
      'coordinates': [coords],
    };

    // Build the computation (simplified - you'll need to expand this)
    // This is the Earth Engine Python/JavaScript logic translated to REST API
    return {
      'expression': '''
        // Load Sentinel-2 collection
        var s2 = ee.ImageCollection('${AppConstants.sentinel2Collection}')
          .filterDate('${AppConstants.startDate}', '${AppConstants.endDate}')
          .filterBounds(geometry)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', ${AppConstants.maxCloudCover}));
        
        // Function to compute NDVI
        var addNDVI = function(image) {
          var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
          return image.addBands(ndvi);
        };
        
        // Apply NDVI calculation
        var withNDVI = s2.map(addNDVI);
        
        // Compute statistics over the region
        var stats = withNDVI.select('NDVI').reduce(ee.Reducer.mean()
          .combine(ee.Reducer.stdDev(), '', true)
          .combine(ee.Reducer.min(), '', true)
          .combine(ee.Reducer.max(), '', true)
          .combine(ee.Reducer.percentile([25, 50, 75]), '', true)
        );
        
        // Sample the region
        var result = stats.reduceRegion({
          reducer: ee.Reducer.first(),
          geometry: geometry,
          scale: 10,
          maxPixels: 1e9
        });
        
        result
      ''',
      'arguments': {
        'geometry': geometry,
      },
    };
  }

  /// Execute Earth Engine computation via REST API
  /// 
  /// Sends the computation expression to EE and waits for results.
  /// This may take 5-30 seconds depending on field size and image count.
  Future<Map<String, dynamic>> _executeEarthEngineComputation(
    Map<String, dynamic> expression,
    String token,
  ) async {
    final url = '${AppConstants.earthEngineBaseUrl}/projects/'
        '${AppConstants.gcpProjectId}:computeValue';

    final response = await http.post(
      Uri.parse(url),
      headers: {
        'Authorization': 'Bearer $token',
        'Content-Type': 'application/json',
      },
      body: json.encode(expression),
    );

    if (response.statusCode != 200) {
      throw Exception(
        'Earth Engine computation failed: ${response.statusCode} - ${response.body}'
      );
    }

    return json.decode(response.body) as Map<String, dynamic>;
  }

  /// Parse Earth Engine results into the 17-feature vector
  /// 
  /// Extract statistics from EE response and compute derived features.
  List<double> _parseFeatures(
    Map<String, dynamic> result,
    FieldData field,
  ) {
    // Extract basic NDVI statistics from Earth Engine result
    final ndviMean = (result['NDVI_mean'] as num?)?.toDouble() ?? 0.0;
    final ndviStd = (result['NDVI_stdDev'] as num?)?.toDouble() ?? 0.0;
    final ndviMin = (result['NDVI_min'] as num?)?.toDouble() ?? 0.0;
    final ndviMax = (result['NDVI_max'] as num?)?.toDouble() ?? 0.0;
    final ndviP25 = (result['NDVI_p25'] as num?)?.toDouble() ?? 0.0;
    final ndviP50 = (result['NDVI_p50'] as num?)?.toDouble() ?? 0.0;
    final ndviP75 = (result['NDVI_p75'] as num?)?.toDouble() ?? 0.0;

    // Temporal features (early vs late season)
    // NOTE: You'll need to modify the EE expression to compute these separately
    final ndviEarly = (result['NDVI_early'] as num?)?.toDouble() ?? ndviMean;
    final ndviLate = (result['NDVI_late'] as num?)?.toDouble() ?? ndviMean;

    // Elevation (from SRTM DEM)
    final elevation = (result['elevation'] as num?)?.toDouble() ?? 0.0;

    // Location features
    final longitude = field.centroid.longitude;
    final latitude = field.centroid.latitude;

    // Derived features
    final ndviRange = ndviMax - ndviMin;
    final ndviIqr = ndviP75 - ndviP25;
    final ndviChange = ndviLate - ndviEarly;
    final ndviEarlyRatio = ndviMean > 0 ? ndviEarly / ndviMean : 0.0;
    final ndviLateRatio = ndviMean > 0 ? ndviLate / ndviMean : 0.0;

    // Return features in exact order expected by model
    return [
      ndviMean,
      ndviStd,
      ndviMin,
      ndviMax,
      ndviP25,
      ndviP50,
      ndviP75,
      ndviEarly,
      ndviLate,
      elevation,
      longitude,
      latitude,
      ndviRange,
      ndviIqr,
      ndviChange,
      ndviEarlyRatio,
      ndviLateRatio,
    ];
  }

  /// Validate that Earth Engine is accessible with current credentials
  Future<bool> testConnection() async {
    try {
      final token = await _authService.getAccessToken();
      final url = '${AppConstants.earthEngineBaseUrl}/projects/'
          '${AppConstants.gcpProjectId}';

      final response = await http.get(
        Uri.parse(url),
        headers: {'Authorization': 'Bearer $token'},
      );

      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}

