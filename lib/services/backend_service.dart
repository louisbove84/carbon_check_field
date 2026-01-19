/// Backend API service for secure communication with Cloud Run
/// 
/// Handles all communication with the secure backend API:
/// - Firebase Auth token management
/// - Field analysis requests
/// - Error handling and retries

import 'dart:convert';
import 'dart:math';
import 'package:http/http.dart' as http;
import 'package:firebase_auth/firebase_auth.dart';
import 'package:carbon_check_field/models/field_data.dart';
import 'package:carbon_check_field/models/prediction_result.dart';
import 'package:carbon_check_field/models/crop_zone.dart';

class BackendService {
  /// Cloud Run backend URL
  static const String backendUrl = 'https://carboncheck-field-api-303566498201.us-central1.run.app';
  
  final FirebaseAuth _auth = FirebaseAuth.instance;
  
  /// Maximum number of retry attempts
  static const int maxRetries = 3;
  
  /// Delay between retries (in seconds)
  static const int retryDelay = 2;

  /// Get Firebase ID token for authentication
  Future<String> _getIdToken() async {
    try {
      final user = _auth.currentUser;
      
      if (user == null) {
        // Sign in anonymously if not already signed in
        await _auth.signInAnonymously();
      }
      
      // Get ID token
      final idToken = await _auth.currentUser?.getIdToken();
      
      if (idToken == null) {
        throw Exception('Failed to get Firebase ID token');
      }
      
      return idToken;
    } catch (e) {
      throw Exception('Firebase authentication failed: $e');
    }
  }

  /// Analyze a field by sending polygon to backend API
  /// 
  /// Returns PredictionResult with crop type, confidence, and CO₂ income
  Future<PredictionResult> analyzeField(FieldData field, {int? year}) async {
    try {
      // Get authentication token
      final idToken = await _getIdToken();
      
      // Build request body
      final requestBody = {
        'polygon': field.polygonPoints
            .map((p) => {'lat': p.latitude, 'lng': p.longitude})
            .toList(),
        'year': year ?? DateTime.now().year,
      };
      
      // Make API call with retry logic
      final response = await _makeRequestWithRetry(
        '/analyze',
        requestBody,
        idToken,
      );
      
      // Parse response
      return _parseAnalysisResponse(response, field.areaAcres);
      
    } catch (e) {
      throw Exception('Field analysis failed: $e');
    }
  }

  /// Make HTTP request with automatic retry logic
  Future<Map<String, dynamic>> _makeRequestWithRetry(
    String endpoint,
    Map<String, dynamic> body,
    String idToken,
  ) async {
    int attempts = 0;
    Exception? lastException;
    
    while (attempts < maxRetries) {
      attempts++;
      
      try {
        final url = Uri.parse('$backendUrl$endpoint');
        final requestId = '${DateTime.now().millisecondsSinceEpoch}-${Random().nextInt(100000)}';
        
        final response = await http.post(
          url,
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer $idToken',
            'X-Request-Id': requestId,
          },
          body: json.encode(body),
        ).timeout(
          const Duration(seconds: 60),
        );
        
        if (response.statusCode == 200) {
          return json.decode(response.body) as Map<String, dynamic>;
        } else if (response.statusCode == 401) {
          throw Exception('Authentication failed. Please restart the app.');
        } else if (response.statusCode >= 500 && attempts < maxRetries) {
          // Server error - retry
          lastException = Exception('Server error (${response.statusCode}). Retrying...');
          await Future.delayed(Duration(seconds: retryDelay * attempts));
          continue;
        } else {
          // Log for troubleshooting (shows up in browser console for web)
          // ignore: avoid_print
          print('API error ${response.statusCode} body=${response.body}');
          throw Exception('API error (${response.statusCode}): ${response.body}');
        }
      } on http.ClientException catch (e) {
        lastException = Exception('Network error: $e');
        if (attempts < maxRetries) {
          await Future.delayed(Duration(seconds: retryDelay * attempts));
          continue;
        }
      } catch (e) {
        lastException = Exception('Request failed: $e');
        if (attempts < maxRetries) {
          await Future.delayed(Duration(seconds: retryDelay * attempts));
          continue;
        }
      }
    }
    
    throw lastException ?? Exception('Request failed after $maxRetries attempts');
  }

  /// Parse analysis response from backend
  /// 
  /// Handles both formats:
  /// - Single prediction (small fields < 10 acres)
  /// - Grid-based (larger fields >= 10 acres)
  PredictionResult _parseAnalysisResponse(
    Map<String, dynamic> response,
    double areaAcres,
  ) {
    try {
      // Check if this is a grid-based response (new format)
      if (response.containsKey('crop_zones') && response['crop_zones'] != null) {
        return _parseGridResponse(response);
      }
      
      // Otherwise, parse as single prediction (legacy format)
      return PredictionResult(
        cropType: response['crop'] as String,
        confidence: (response['confidence'] as num?)?.toDouble() ?? -1,
        cdlCropType: response['cdl_crop'] as String?,
        cdlAgreement: response['cdl_agreement'] as bool? ?? false,
        areaAcres: areaAcres,
        carbonIncomeMin: (response['co2_income_min'] as num).toDouble(),
        carbonIncomeMax: (response['co2_income_max'] as num).toDouble(),
        carbonIncomeAverage: (response['co2_income_avg'] as num).toDouble(),
        predictedAt: DateTime.now(),
      );
    } catch (e) {
      throw Exception('Failed to parse analysis response: $e');
    }
  }
  
  /// Parse grid-based response (for fields >= 10 acres)
  /// 
  /// Includes all crop zones for map visualization
  PredictionResult _parseGridResponse(Map<String, dynamic> response) {
    try {
      final fieldSummary = response['field_summary'] as Map<String, dynamic>;
      final cropZonesJson = response['crop_zones'] as List<dynamic>;
      final co2Income = response['co2_income'] as Map<String, dynamic>;
      
      if (cropZonesJson.isEmpty) {
        final totalArea = (fieldSummary['total_area_acres'] as num).toDouble();
        return PredictionResult(
          cropType: 'No crops detected',
          confidence: 0.0,
          cdlCropType: null,
          cdlAgreement: false,
          areaAcres: totalArea,
          carbonIncomeMin: (co2Income['total_min'] as num).toDouble(),
          carbonIncomeMax: (co2Income['total_max'] as num).toDouble(),
          carbonIncomeAverage: (co2Income['total_avg'] as num).toDouble(),
          predictedAt: DateTime.now(),
          cropZones: const [],
        );
      }
      
      // Parse crop zones
      final zones = cropZonesJson
          .map((z) => CropZone.fromJson(z as Map<String, dynamic>))
          .toList();
      
      // Find dominant crop (largest area)
      var dominantZone = cropZonesJson.first;
      double maxArea = 0;
      for (var zone in cropZonesJson) {
        final zoneArea = (zone['area_acres'] as num?)?.toDouble() ?? 0.0;
        if (zoneArea > maxArea) {
          maxArea = zoneArea;
          dominantZone = zone;
        }
      }
      
      // Calculate weighted average confidence across all zones
      double totalArea = (fieldSummary['total_area_acres'] as num).toDouble();
      double weightedConfidence = 0;
      final confidenceAvailable = zones.every((z) => z.confidence != null);
      for (var zone in cropZonesJson) {
        final confidence = (zone['confidence'] as num?)?.toDouble() ?? 0.0;
        final percentage = (zone['percentage'] as num?)?.toDouble() ?? 0.0;
        weightedConfidence += confidence * (percentage / 100);
      }
      if (!confidenceAvailable) {
        weightedConfidence = -1;
      }
      
      // Count distinct crops
      final distinctCrops = zones.map((z) => z.crop).toSet().length;
      final cropSummary = distinctCrops == 1
          ? dominantZone['crop']
          : '$distinctCrops crop types detected';
      
      return PredictionResult(
        cropType: cropSummary,
        confidence: weightedConfidence,
        cdlCropType: null,  // Not available for grid analysis
        cdlAgreement: false,
        areaAcres: totalArea,
        carbonIncomeMin: (co2Income['total_min'] as num).toDouble(),
        carbonIncomeMax: (co2Income['total_max'] as num).toDouble(),
        carbonIncomeAverage: (co2Income['total_avg'] as num).toDouble(),
        predictedAt: DateTime.now(),
        cropZones: zones,  // Include zones for visualization
      );
    } catch (e) {
      throw Exception('Failed to parse grid response: $e');
    }
  }

  /// Test connection to backend API
  Future<bool> testConnection() async {
    try {
      final url = Uri.parse('$backendUrl/health');
      
      final response = await http.get(url).timeout(
        const Duration(seconds: 10),
      );
      
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  /// Sign out (if needed for debugging)
  Future<void> signOut() async {
    await _auth.signOut();
  }
}

