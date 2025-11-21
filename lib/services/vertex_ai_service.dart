/// Vertex AI service for crop type prediction
/// 
/// Takes 17 NDVI features and calls the deployed Vertex AI model
/// to predict crop type with confidence score.

import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:carbon_check_field/services/auth_service.dart';
import 'package:carbon_check_field/utils/constants.dart';
import 'package:carbon_check_field/models/prediction_result.dart';
import 'package:carbon_check_field/models/field_data.dart';

class VertexAiService {
  final AuthService _authService;

  VertexAiService(this._authService);

  /// Predict crop type from computed features
  /// 
  /// Takes the 17-feature vector and sends it to Vertex AI endpoint.
  /// Returns a PredictionResult with crop type, confidence, and CO₂ income.
  Future<PredictionResult> predictCropType(FieldData field) async {
    // Validate features are present
    if (field.features == null || field.features!.length != 17) {
      throw ArgumentError('Field must have 17 computed features');
    }

    // Step 1: Get access token
    final token = await _authService.getAccessToken();

    // Step 2: Build prediction request
    final requestBody = {
      'instances': [field.features],
    };

    // Step 3: Call Vertex AI endpoint
    final response = await http.post(
      Uri.parse(AppConstants.vertexAiEndpoint),
      headers: {
        'Authorization': 'Bearer $token',
        'Content-Type': 'application/json',
      },
      body: json.encode(requestBody),
    );

    if (response.statusCode != 200) {
      throw Exception(
        'Vertex AI prediction failed: ${response.statusCode} - ${response.body}'
      );
    }

    // Step 4: Parse response
    final responseData = json.decode(response.body) as Map<String, dynamic>;
    final predictions = responseData['predictions'] as List;
    
    if (predictions.isEmpty) {
      throw Exception('No predictions returned from model');
    }

    // Parse crop type and confidence
    // Response format: {"predictions": ["Corn"]} or {"predictions": [["Corn", 0.98]]}
    String cropType;
    double confidence = 0.95; // Default if not provided

    if (predictions[0] is String) {
      cropType = predictions[0] as String;
    } else if (predictions[0] is List) {
      final predList = predictions[0] as List;
      cropType = predList[0] as String;
      if (predList.length > 1) {
        confidence = (predList[1] as num).toDouble();
      }
    } else {
      throw Exception('Unexpected prediction format: ${predictions[0]}');
    }

    // Step 5: Calculate carbon credit income
    final carbonIncome = _calculateCarbonIncome(cropType, field.areaAcres);

    // Step 6: Build result object
    return PredictionResult(
      cropType: cropType,
      confidence: confidence,
      areaAcres: field.areaAcres,
      carbonIncomeMin: carbonIncome['min']!,
      carbonIncomeMax: carbonIncome['max']!,
      carbonIncomeAverage: carbonIncome['average']!,
      predictedAt: DateTime.now(),
    );
  }

  /// Calculate estimated CO₂ income based on crop type and acreage
  /// 
  /// Uses 2025 market rates from Indigo Ag and Truterra.
  /// Returns min, max, and average income per year.
  Map<String, double> _calculateCarbonIncome(String cropType, double acres) {
    // Get carbon rate for this crop (or default if unknown)
    final rate = AppConstants.carbonRates[cropType] ?? 
                  AppConstants.defaultCarbonRate;

    return {
      'min': rate.min * acres,
      'max': rate.max * acres,
      'average': rate.average * acres,
    };
  }

  /// Test connection to Vertex AI endpoint
  /// 
  /// Sends a dummy prediction request to verify authentication works.
  Future<bool> testConnection() async {
    try {
      final token = await _authService.getAccessToken();
      
      // Send dummy features (all zeros)
      final requestBody = {
        'instances': [List.filled(17, 0.0)],
      };

      final response = await http.post(
        Uri.parse(AppConstants.vertexAiEndpoint),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
        body: json.encode(requestBody),
      );

      // Even if prediction fails, a 200 means endpoint is accessible
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}

