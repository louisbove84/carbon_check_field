/// Application-wide constants for CarbonCheck Field
/// 
/// This file contains all hardcoded values, API endpoints, 
/// and configuration settings in one place for easy maintenance.

class AppConstants {
  // ============================================================
  // API ENDPOINTS
  // ============================================================
  
  /// Vertex AI model endpoint for crop prediction
  static const String vertexAiEndpoint =
      'https://us-central1-aiplatform.googleapis.com/v1/projects/ml-pipeline-477612/locations/us-central1/endpoints/7591968360607252480:predict';
  
  /// Earth Engine REST API base URL
  static const String earthEngineBaseUrl =
      'https://earthengine.googleapis.com/v1';
  
  /// Google Cloud project ID for Earth Engine
  static const String gcpProjectId = 'ml-pipeline-477612';
  
  /// OAuth2 token endpoint for service account
  static const String oauth2TokenUrl =
      'https://oauth2.googleapis.com/token';

  // ============================================================
  // EARTH ENGINE CONFIGURATION
  // ============================================================
  
  /// Sentinel-2 Surface Reflectance Harmonized collection
  static const String sentinel2Collection = 'COPERNICUS/S2_SR_HARMONIZED';
  
  /// Year to query for NDVI data
  static const int dataYear = 2024;
  
  /// Date range for analysis (full year)
  static const String startDate = '2024-01-01';
  static const String endDate = '2024-12-31';
  
  /// Cloud cover threshold (percentage)
  static const int maxCloudCover = 20;

  // ============================================================
  // MODEL FEATURE CONFIGURATION
  // ============================================================
  
  /// Number of features expected by the ML model
  static const int numFeatures = 17;
  
  /// Feature names in the exact order required by Vertex AI
  static const List<String> featureNames = [
    'ndvi_mean',
    'ndvi_std',
    'ndvi_min',
    'ndvi_max',
    'ndvi_p25',
    'ndvi_p50',
    'ndvi_p75',
    'ndvi_early',
    'ndvi_late',
    'elevation_m',
    'longitude',
    'latitude',
    'ndvi_range',
    'ndvi_iqr',
    'ndvi_change',
    'ndvi_early_ratio',
    'ndvi_late_ratio',
  ];

  // ============================================================
  // CARBON CREDIT RATES ($/acre/year)
  // ============================================================
  
  /// Carbon credit income rates per crop type
  /// Based on 2025 Indigo Ag and Truterra market rates
  static const Map<String, CarbonRate> carbonRates = {
    'Corn': CarbonRate(min: 12.0, max: 18.0),
    'Soybeans': CarbonRate(min: 15.0, max: 22.0),
    'Alfalfa': CarbonRate(min: 18.0, max: 25.0),
    'Winter Wheat': CarbonRate(min: 10.0, max: 15.0),
  };
  
  /// Default rate for unknown crops
  static const CarbonRate defaultCarbonRate = CarbonRate(min: 10.0, max: 20.0);

  // ============================================================
  // UI CONFIGURATION
  // ============================================================
  
  /// App primary color (farmer-friendly green)
  static const int primaryColorValue = 0xFF2E7D32;
  
  /// App accent color (sky blue)
  static const int accentColorValue = 0xFF1976D2;
  
  /// Default map camera position (center of US farmland)
  static const double defaultLatitude = 41.8781;
  static const double defaultLongitude = -93.0977;
  static const double defaultZoom = 15.0;
  
  /// Minimum polygon points required
  static const int minPolygonPoints = 3;
  
  /// Maximum polygon points allowed
  static const int maxPolygonPoints = 50;

  // ============================================================
  // CONVERSION FACTORS
  // ============================================================
  
  /// Square meters to acres conversion
  static const double sqMetersToAcres = 0.000247105;
  
  /// Meters to feet conversion
  static const double metersToFeet = 3.28084;

  // ============================================================
  // ERROR MESSAGES
  // ============================================================
  
  static const String errorNoSatelliteData =
      'No satellite data available for this location. Try a different field.';
  
  static const String errorNetworkFailure =
      'Network error. Please check your internet connection.';
  
  static const String errorAuthFailure =
      'Authentication failed. Please check service account credentials.';
  
  static const String errorInvalidPolygon =
      'Invalid field boundary. Please draw at least 3 points.';
  
  static const String errorPredictionFailed =
      'Crop prediction failed. Please try again.';
}

/// Carbon credit rate range for a specific crop
class CarbonRate {
  final double min;
  final double max;
  
  const CarbonRate({required this.min, required this.max});
  
  /// Calculate average rate
  double get average => (min + max) / 2;
}

