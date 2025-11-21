/// Data model representing a farmer's field with its geographic boundaries
/// and computed features for crop classification.

import 'package:google_maps_flutter/google_maps_flutter.dart';

/// Represents a single field with polygon boundaries and NDVI features
class FieldData {
  /// Unique identifier for this field
  final String id;
  
  /// User-drawn polygon points (latitude, longitude)
  final List<LatLng> polygonPoints;
  
  /// Field area in acres
  final double areaAcres;
  
  /// Centroid coordinates (for Earth Engine queries)
  final LatLng centroid;
  
  /// Bounding box for Earth Engine region filtering
  final LatLngBounds bounds;
  
  /// Timestamp when field was drawn
  final DateTime createdAt;
  
  /// 17 computed features for ML model (nullable until computed)
  List<double>? features;

  FieldData({
    required this.id,
    required this.polygonPoints,
    required this.areaAcres,
    required this.centroid,
    required this.bounds,
    required this.createdAt,
    this.features,
  });

  /// Create a copy with updated features
  FieldData copyWith({
    List<double>? features,
  }) {
    return FieldData(
      id: id,
      polygonPoints: polygonPoints,
      areaAcres: areaAcres,
      centroid: centroid,
      bounds: bounds,
      createdAt: createdAt,
      features: features ?? this.features,
    );
  }

  /// Convert to JSON for storage/sharing
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'polygonPoints': polygonPoints
          .map((p) => {'lat': p.latitude, 'lng': p.longitude})
          .toList(),
      'areaAcres': areaAcres,
      'centroid': {
        'lat': centroid.latitude,
        'lng': centroid.longitude,
      },
      'createdAt': createdAt.toIso8601String(),
      'features': features,
    };
  }

  /// Create from JSON
  factory FieldData.fromJson(Map<String, dynamic> json) {
    return FieldData(
      id: json['id'] as String,
      polygonPoints: (json['polygonPoints'] as List)
          .map((p) => LatLng(p['lat'] as double, p['lng'] as double))
          .toList(),
      areaAcres: json['areaAcres'] as double,
      centroid: LatLng(
        json['centroid']['lat'] as double,
        json['centroid']['lng'] as double,
      ),
      bounds: LatLngBounds(
        southwest: LatLng(
          json['bounds']['southwest']['lat'] as double,
          json['bounds']['southwest']['lng'] as double,
        ),
        northeast: LatLng(
          json['bounds']['northeast']['lat'] as double,
          json['bounds']['northeast']['lng'] as double,
        ),
      ),
      createdAt: DateTime.parse(json['createdAt'] as String),
      features: json['features'] != null
          ? List<double>.from(json['features'] as List)
          : null,
    );
  }

  /// Check if features have been computed
  bool get hasFeatures => features != null && features!.length == 17;
}

