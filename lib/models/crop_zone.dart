/// Data model for individual crop zones within a field
/// 
/// Used for grid-based classification results where a field
/// may contain multiple crop types in different areas.

import 'package:google_maps_flutter/google_maps_flutter.dart';

class CropZone {
  /// Crop type (e.g., "Corn", "Soybeans")
  final String crop;
  
  /// Confidence score for this zone (0.0 to 1.0)
  final double confidence;
  
  /// Area of this zone in acres
  final double areaAcres;
  
  /// Percentage of total field area
  final double percentage;
  
  /// Polygon boundary coordinates
  final List<LatLng> polygon;

  CropZone({
    required this.crop,
    required this.confidence,
    required this.areaAcres,
    required this.percentage,
    required this.polygon,
  });

  /// Format confidence as percentage
  String get confidencePercentage => '${(confidence * 100).toStringAsFixed(0)}%';

  /// Format area with 1 decimal
  String get areaFormatted => '${areaAcres.toStringAsFixed(1)} acres';

  /// Format percentage with 1 decimal
  String get percentageFormatted => '${percentage.toStringAsFixed(1)}%';

  /// Create from backend JSON
  factory CropZone.fromJson(Map<String, dynamic> json) {
    // Parse polygon coordinates
    final polygonCoords = (json['polygon'] as List<dynamic>)
        .map((coord) => LatLng(
              (coord[1] as num).toDouble(), // lat
              (coord[0] as num).toDouble(), // lng
            ))
        .toList();

    return CropZone(
      crop: json['crop'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      areaAcres: (json['area_acres'] as num).toDouble(),
      percentage: (json['percentage'] as num).toDouble(),
      polygon: polygonCoords,
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'crop': crop,
      'confidence': confidence,
      'area_acres': areaAcres,
      'percentage': percentage,
      'polygon': polygon
          .map((coord) => [coord.longitude, coord.latitude])
          .toList(),
    };
  }
}

