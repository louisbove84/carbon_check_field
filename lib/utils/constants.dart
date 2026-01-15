import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'web_config_stub.dart' if (dart.library.js_util) 'web_config.dart';

/// Application-wide constants for CarbonCheck Field
/// 
/// This file contains all hardcoded values, API endpoints, 
/// and configuration settings in one place for easy maintenance.

class AppConstants {
  // ============================================================
  // API KEYS & BACKEND CONFIGURATION
  // ============================================================
  
  /// Google Maps API key (for geocoding and maps)
  /// Loaded from .env on mobile/desktop, or window.GOOGLE_MAPS_API_KEY on web
  static String get googleMapsApiKey {
    if (kIsWeb) {
      final webKey = getMapsKeyFromWindow();
      if (webKey != null && webKey.isNotEmpty) {
        return webKey;
      }
    }
    final key = dotenv.env['GOOGLE_MAPS_API_KEY'];
    if (key == null || key.isEmpty) {
      throw Exception(
        'GOOGLE_MAPS_API_KEY not found. '
        'On web, ensure /api/carboncheck-config is reachable. '
        'On mobile, create .env with GOOGLE_MAPS_API_KEY=your_key.'
      );
    }
    return key;
  }
  
  /// Secure Cloud Run backend URL
  /// TODO: Update this after deploying backend to Cloud Run
  static const String backendApiUrl = 
      'https://carboncheck-field-api-XXXXXXXX-uc.a.run.app';
  
  /// API request timeout (seconds)
  static const int apiTimeout = 60;
  
  /// Maximum retry attempts for failed requests
  static const int maxRetries = 3;

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
  
  /// Default map camera position
  static const double defaultLatitude = 44.409438290384166;
  static const double defaultLongitude = -88.4304410977501;
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
      'Authentication failed. Please restart the app.';
  
  static const String errorInvalidPolygon =
      'Invalid field boundary. Please draw at least 3 points.';
  
  static const String errorAnalysisFailed =
      'Field analysis failed. Please try again.';
  
  static const String errorBackendUnavailable =
      'Backend service unavailable. Please try again later.';
}

/// Carbon credit rate range for a specific crop
class CarbonRate {
  final double min;
  final double max;
  
  const CarbonRate({required this.min, required this.max});
  
  /// Calculate average rate
  double get average => (min + max) / 2;
}

