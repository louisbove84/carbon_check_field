/// Offline precomputed crop lookup
///
/// Loads a bundled grid of precomputed crop predictions (Fox Valley demo block)
/// and resolves a drawn field to a crop + carbon estimate entirely on-device —
/// no backend, no network, no cost. Fields outside the covered area return null
/// so the caller can fall back to the online backend.

import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:carbon_check_field/models/field_data.dart';
import 'package:carbon_check_field/models/prediction_result.dart';
import 'package:carbon_check_field/utils/constants.dart';
import 'package:carbon_check_field/utils/geo_utils.dart';

class _Cell {
  final double lat;
  final double lng;
  final String crop;
  final double conf;
  const _Cell(this.lat, this.lng, this.crop, this.conf);
}

class OfflineLookupService {
  static const String _assetPath = 'assets/precomputed_foxvalley.json';

  static List<_Cell>? _cells;
  static double _cellSizeM = 250;
  static double _minLat = 0, _maxLat = 0, _minLng = 0, _maxLng = 0;

  /// Load and cache the bundled grid. Safe to call multiple times.
  static Future<void> ensureLoaded() async {
    if (_cells != null) return;
    final raw = await rootBundle.loadString(_assetPath);
    final data = json.decode(raw) as Map<String, dynamic>;

    _cellSizeM = (data['cell_size_m'] as num?)?.toDouble() ?? 250;
    final bounds = data['bounds'] as Map<String, dynamic>?;
    if (bounds != null) {
      _minLat = (bounds['min_lat'] as num).toDouble();
      _maxLat = (bounds['max_lat'] as num).toDouble();
      _minLng = (bounds['min_lng'] as num).toDouble();
      _maxLng = (bounds['max_lng'] as num).toDouble();
    }

    _cells = (data['cells'] as List<dynamic>)
        .map((c) => _Cell(
              (c['lat'] as num).toDouble(),
              (c['lng'] as num).toDouble(),
              c['crop'] as String,
              (c['conf'] as num).toDouble(),
            ))
        .toList();
  }

  /// Whether a point falls within the precomputed coverage bounds.
  static bool isInCoverage(LatLng p) {
    return p.latitude >= _minLat &&
        p.latitude <= _maxLat &&
        p.longitude >= _minLng &&
        p.longitude <= _maxLng;
  }

  /// Resolve a field to a prediction using the nearest precomputed cell.
  ///
  /// Returns null if the field is outside the covered area, falls on a gap,
  /// or maps to a non-crop ("Other") cell — letting the caller fall back online.
  static Future<PredictionResult?> analyze(FieldData field) async {
    await ensureLoaded();
    final cells = _cells;
    if (cells == null || cells.isEmpty) return null;

    final centroid = GeoUtils.calculateCentroid(field.polygonPoints);
    if (!isInCoverage(centroid)) return null;

    _Cell? nearest;
    double bestMeters = double.infinity;
    for (final cell in cells) {
      final d = GeoUtils.calculateDistance(
        centroid,
        LatLng(cell.lat, cell.lng),
      );
      if (d < bestMeters) {
        bestMeters = d;
        nearest = cell;
      }
    }

    // Must be within one cell of a precomputed centroid to count as covered.
    if (nearest == null || bestMeters > _cellSizeM) return null;

    // Non-crop cell (urban/water): let the online path handle it.
    if (nearest.crop == 'Other') return null;

    final rate = AppConstants.carbonRates[nearest.crop] ??
        AppConstants.defaultCarbonRate;
    final area = field.areaAcres;

    return PredictionResult(
      cropType: nearest.crop,
      confidence: nearest.conf,
      cdlCropType: null,
      cdlAgreement: false,
      areaAcres: area,
      carbonIncomeMin: area * rate.min,
      carbonIncomeMax: area * rate.max,
      carbonIncomeAverage: area * rate.average,
      predictedAt: DateTime.now(),
    );
  }
}
