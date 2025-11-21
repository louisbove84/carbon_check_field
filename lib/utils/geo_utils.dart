/// Geospatial utility functions for polygon area calculation,
/// centroid computation, and coordinate transformations.

import 'dart:math' as math;
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:carbon_check_field/utils/constants.dart';

class GeoUtils {
  /// Calculate polygon area in acres using the Shoelace formula
  /// 
  /// This formula works for any simple polygon (no self-intersections).
  /// It computes area in square meters, then converts to acres.
  /// 
  /// Reference: https://en.wikipedia.org/wiki/Shoelace_formula
  static double calculatePolygonAreaAcres(List<LatLng> points) {
    if (points.length < 3) return 0.0;

    // Earth radius in meters
    const double earthRadius = 6378137.0;
    
    double area = 0.0;

    // Convert to radians and compute area
    for (int i = 0; i < points.length; i++) {
      final p1 = points[i];
      final p2 = points[(i + 1) % points.length];

      final lat1 = p1.latitude * math.pi / 180;
      final lat2 = p2.latitude * math.pi / 180;
      final lng1 = p1.longitude * math.pi / 180;
      final lng2 = p2.longitude * math.pi / 180;

      area += (lng2 - lng1) * (2 + math.sin(lat1) + math.sin(lat2));
    }

    area = area.abs() * earthRadius * earthRadius / 2.0;

    // Convert square meters to acres
    return area * AppConstants.sqMetersToAcres;
  }

  /// Calculate the centroid (geographic center) of a polygon
  /// 
  /// Uses the weighted average method for latitude and longitude.
  /// This is approximate but accurate for small agricultural fields.
  static LatLng calculateCentroid(List<LatLng> points) {
    if (points.isEmpty) {
      throw ArgumentError('Cannot calculate centroid of empty polygon');
    }

    double latSum = 0.0;
    double lngSum = 0.0;

    for (final point in points) {
      latSum += point.latitude;
      lngSum += point.longitude;
    }

    return LatLng(
      latSum / points.length,
      lngSum / points.length,
    );
  }

  /// Calculate bounding box (min/max lat/lng) for a set of points
  /// 
  /// Returns a LatLngBounds that can be used for Earth Engine region filtering.
  static LatLngBounds calculateBounds(List<LatLng> points) {
    if (points.isEmpty) {
      throw ArgumentError('Cannot calculate bounds of empty polygon');
    }

    double minLat = points[0].latitude;
    double maxLat = points[0].latitude;
    double minLng = points[0].longitude;
    double maxLng = points[0].longitude;

    for (final point in points) {
      if (point.latitude < minLat) minLat = point.latitude;
      if (point.latitude > maxLat) maxLat = point.latitude;
      if (point.longitude < minLng) minLng = point.longitude;
      if (point.longitude > maxLng) maxLng = point.longitude;
    }

    return LatLngBounds(
      southwest: LatLng(minLat, minLng),
      northeast: LatLng(maxLat, maxLng),
    );
  }

  /// Convert LatLng list to GeoJSON coordinates format
  /// 
  /// Earth Engine expects coordinates as [lng, lat] (note the order!)
  /// and closed polygons (first point == last point).
  static List<List<double>> toGeoJsonCoordinates(List<LatLng> points) {
    final coords = points.map((p) => [p.longitude, p.latitude]).toList();
    
    // Close the polygon if not already closed
    if (points.first.latitude != points.last.latitude ||
        points.first.longitude != points.last.longitude) {
      coords.add([points.first.longitude, points.first.latitude]);
    }
    
    return coords;
  }

  /// Calculate distance between two points in meters (Haversine formula)
  /// 
  /// Useful for validation (e.g., checking if polygon is too small/large).
  static double calculateDistance(LatLng point1, LatLng point2) {
    const double earthRadius = 6378137.0; // meters

    final lat1 = point1.latitude * math.pi / 180;
    final lat2 = point2.latitude * math.pi / 180;
    final dLat = (point2.latitude - point1.latitude) * math.pi / 180;
    final dLng = (point2.longitude - point1.longitude) * math.pi / 180;

    final a = math.sin(dLat / 2) * math.sin(dLat / 2) +
        math.cos(lat1) *
            math.cos(lat2) *
            math.sin(dLng / 2) *
            math.sin(dLng / 2);

    final c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a));

    return earthRadius * c;
  }

  /// Validate polygon is reasonable for a farm field
  /// 
  /// Returns error message if invalid, null if valid.
  static String? validatePolygon(List<LatLng> points) {
    if (points.length < AppConstants.minPolygonPoints) {
      return 'Field must have at least ${AppConstants.minPolygonPoints} points';
    }

    if (points.length > AppConstants.maxPolygonPoints) {
      return 'Field cannot have more than ${AppConstants.maxPolygonPoints} points';
    }

    final area = calculatePolygonAreaAcres(points);
    
    if (area < 0.1) {
      return 'Field is too small (minimum 0.1 acres)';
    }

    if (area > 10000) {
      return 'Field is too large (maximum 10,000 acres)';
    }

    return null; // Valid!
  }
}

