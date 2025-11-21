/// Unit tests for geospatial utility functions
/// 
/// Run with: flutter test test/geo_utils_test.dart

import 'package:flutter_test/flutter_test.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:carbon_check_field/utils/geo_utils.dart';

void main() {
  group('GeoUtils - Area Calculation', () {
    test('calculates area of simple square correctly', () {
      // Create a ~1 square mile polygon (640 acres)
      final square = [
        const LatLng(41.0, -93.0),
        const LatLng(41.0, -92.9855),  // ~1 mile east
        const LatLng(40.9855, -92.9855), // ~1 mile south
        const LatLng(40.9855, -93.0),   // back west
      ];

      final area = GeoUtils.calculatePolygonAreaAcres(square);
      
      // Should be approximately 640 acres (within 10% tolerance)
      expect(area, inInclusiveRange(580, 700));
    });

    test('calculates area of triangle', () {
      final triangle = [
        const LatLng(41.0, -93.0),
        const LatLng(41.01, -93.0),
        const LatLng(41.005, -93.01),
      ];

      final area = GeoUtils.calculatePolygonAreaAcres(triangle);
      
      // Should be greater than 0
      expect(area, greaterThan(0));
    });

    test('returns zero for less than 3 points', () {
      final twoPoints = [
        const LatLng(41.0, -93.0),
        const LatLng(41.01, -93.0),
      ];

      final area = GeoUtils.calculatePolygonAreaAcres(twoPoints);
      expect(area, equals(0.0));
    });
  });

  group('GeoUtils - Centroid Calculation', () {
    test('calculates centroid of square correctly', () {
      final square = [
        const LatLng(41.0, -93.0),
        const LatLng(41.0, -92.0),
        const LatLng(40.0, -92.0),
        const LatLng(40.0, -93.0),
      ];

      final centroid = GeoUtils.calculateCentroid(square);
      
      // Centroid should be at (40.5, -92.5)
      expect(centroid.latitude, closeTo(40.5, 0.01));
      expect(centroid.longitude, closeTo(-92.5, 0.01));
    });

    test('throws error for empty polygon', () {
      expect(
        () => GeoUtils.calculateCentroid([]),
        throwsA(isA<ArgumentError>()),
      );
    });
  });

  group('GeoUtils - Bounds Calculation', () {
    test('calculates bounding box correctly', () {
      final points = [
        const LatLng(41.0, -93.0),
        const LatLng(41.5, -92.5),
        const LatLng(40.5, -93.5),
      ];

      final bounds = GeoUtils.calculateBounds(points);
      
      expect(bounds.southwest.latitude, equals(40.5));
      expect(bounds.southwest.longitude, equals(-93.5));
      expect(bounds.northeast.latitude, equals(41.5));
      expect(bounds.northeast.longitude, equals(-92.5));
    });
  });

  group('GeoUtils - Distance Calculation', () {
    test('calculates distance between two points', () {
      const point1 = LatLng(41.0, -93.0);
      const point2 = LatLng(41.0, -92.0);

      final distance = GeoUtils.calculateDistance(point1, point2);
      
      // ~1 degree longitude at 41°N ≈ 85 km
      expect(distance, inInclusiveRange(80000, 90000)); // meters
    });

    test('distance to same point is zero', () {
      const point = LatLng(41.0, -93.0);

      final distance = GeoUtils.calculateDistance(point, point);
      expect(distance, equals(0.0));
    });
  });

  group('GeoUtils - Polygon Validation', () {
    test('validates correct polygon', () {
      final validPolygon = [
        const LatLng(41.0, -93.0),
        const LatLng(41.01, -93.0),
        const LatLng(41.01, -93.01),
        const LatLng(41.0, -93.01),
      ];

      final error = GeoUtils.validatePolygon(validPolygon);
      expect(error, isNull);
    });

    test('rejects polygon with too few points', () {
      final tooFewPoints = [
        const LatLng(41.0, -93.0),
        const LatLng(41.01, -93.0),
      ];

      final error = GeoUtils.validatePolygon(tooFewPoints);
      expect(error, isNotNull);
      expect(error, contains('at least'));
    });

    test('rejects polygon that is too small', () {
      final tinyPolygon = [
        const LatLng(41.0, -93.0),
        const LatLng(41.0001, -93.0),
        const LatLng(41.0001, -93.0001),
      ];

      final error = GeoUtils.validatePolygon(tinyPolygon);
      expect(error, contains('too small'));
    });
  });

  group('GeoUtils - GeoJSON Conversion', () {
    test('converts LatLng to GeoJSON coordinates', () {
      final points = [
        const LatLng(41.0, -93.0),
        const LatLng(41.01, -93.0),
        const LatLng(41.01, -93.01),
      ];

      final coords = GeoUtils.toGeoJsonCoordinates(points);
      
      // Check first point (note: GeoJSON is [lng, lat])
      expect(coords[0][0], equals(-93.0));  // longitude
      expect(coords[0][1], equals(41.0));   // latitude
      
      // Should be closed (last point == first point)
      expect(coords.last, equals(coords.first));
    });
  });
}

