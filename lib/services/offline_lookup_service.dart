/// Offline precomputed crop lookup
///
/// Loads a bundled grid of precomputed crop predictions (Fox Valley demo block)
/// and resolves a drawn field to crop zones + carbon estimate entirely on-device
/// — no backend, no network, no cost. Each precomputed cell is reconstructed as
/// a square and clipped to the drawn field, producing the same colored crop-zone
/// overlay the backend used to return. Fields outside the covered area return
/// null so the caller can fall back to the online backend.

import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart' show rootBundle;
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:carbon_check_field/models/crop_zone.dart';
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

class _Slice {
  final String crop;
  final double conf;
  final double area;
  final List<LatLng> polygon;
  const _Slice(this.crop, this.conf, this.area, this.polygon);
}

class OfflineLookupService {
  static const String _assetPath = 'assets/precomputed_foxvalley.json';
  static const double _metersPerDegLat = 111320.0;

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

  /// Resolve a field to crop zones using the precomputed grid.
  ///
  /// Returns null if the field is outside the covered area so the caller can
  /// fall back to the online backend.
  static Future<PredictionResult?> analyze(FieldData field) async {
    await ensureLoaded();
    final cells = _cells;
    if (cells == null || cells.isEmpty) return null;

    final fieldPoly = field.polygonPoints;
    final centroid = GeoUtils.calculateCentroid(fieldPoly);
    if (!isInCoverage(centroid)) return null;

    // Field bounding box (fast reject for non-overlapping cells)
    double fMinLat = fieldPoly.first.latitude, fMaxLat = fieldPoly.first.latitude;
    double fMinLng = fieldPoly.first.longitude, fMaxLng = fieldPoly.first.longitude;
    for (final p in fieldPoly) {
      fMinLat = math.min(fMinLat, p.latitude);
      fMaxLat = math.max(fMaxLat, p.latitude);
      fMinLng = math.min(fMinLng, p.longitude);
      fMaxLng = math.max(fMaxLng, p.longitude);
    }

    final halfLat = (_cellSizeM / 2) / _metersPerDegLat;

    // First pass: clip overlapping cells to the field and gather raw slices
    final slices = <_Slice>[];
    final areaByCrop = <String, double>{};
    double totalArea = 0;
    double confTimesArea = 0;

    for (final cell in cells) {
      if (cell.crop == 'Other') continue;

      final halfLng = (_cellSizeM / 2) /
          (_metersPerDegLat * math.cos(cell.lat * math.pi / 180));
      final cMinLat = cell.lat - halfLat;
      final cMaxLat = cell.lat + halfLat;
      final cMinLng = cell.lng - halfLng;
      final cMaxLng = cell.lng + halfLng;

      // Fast reject: cell square vs field bbox
      if (cMaxLat < fMinLat ||
          cMinLat > fMaxLat ||
          cMaxLng < fMinLng ||
          cMinLng > fMaxLng) {
        continue;
      }

      final clipped = _clipToRect(fieldPoly, cMinLng, cMaxLng, cMinLat, cMaxLat);
      if (clipped.length < 3) continue;

      final area = GeoUtils.calculatePolygonAreaAcres(clipped);
      if (area <= 0.001) continue;

      slices.add(_Slice(cell.crop, cell.conf, area, clipped));
      areaByCrop[cell.crop] = (areaByCrop[cell.crop] ?? 0) + area;
      totalArea += area;
      confTimesArea += cell.conf * area;
    }

    // Field fell on a non-crop cell or a grid gap — let the online path handle it
    if (slices.isEmpty || totalArea <= 0) return null;

    // Second pass: build zones now that the total area is known
    final zones = slices
        .map((s) => CropZone(
              crop: s.crop,
              confidence: s.conf,
              areaAcres: s.area,
              percentage: (s.area / totalArea) * 100,
              polygon: s.polygon,
            ))
        .toList();

    final dominantCrop =
        areaByCrop.entries.reduce((a, b) => a.value >= b.value ? a : b).key;
    final confidence = confTimesArea / totalArea;
    final fieldAcres = field.areaAcres;

    double incomeMin = 0, incomeMax = 0, incomeAvg = 0;
    areaByCrop.forEach((crop, acres) {
      final rate =
          AppConstants.carbonRates[crop] ?? AppConstants.defaultCarbonRate;
      incomeMin += acres * rate.min;
      incomeMax += acres * rate.max;
      incomeAvg += acres * rate.average;
    });

    return PredictionResult(
      cropType: dominantCrop,
      confidence: confidence,
      cdlCropType: null,
      cdlAgreement: false,
      areaAcres: fieldAcres,
      carbonIncomeMin: incomeMin,
      carbonIncomeMax: incomeMax,
      carbonIncomeAverage: incomeAvg,
      predictedAt: DateTime.now(),
      cropZones: zones,
    );
  }

  /// Clip a polygon to an axis-aligned lat/lng rectangle (Sutherland–Hodgman).
  static List<LatLng> _clipToRect(
    List<LatLng> poly,
    double minLng,
    double maxLng,
    double minLat,
    double maxLat,
  ) {
    var out = poly;
    // x >= minLng
    out = _clipEdge(out, (p) => p.longitude >= minLng,
        (a, b) => _interpX(a, b, minLng));
    if (out.length < 3) return out;
    // x <= maxLng
    out = _clipEdge(out, (p) => p.longitude <= maxLng,
        (a, b) => _interpX(a, b, maxLng));
    if (out.length < 3) return out;
    // y >= minLat
    out = _clipEdge(out, (p) => p.latitude >= minLat,
        (a, b) => _interpY(a, b, minLat));
    if (out.length < 3) return out;
    // y <= maxLat
    out = _clipEdge(out, (p) => p.latitude <= maxLat,
        (a, b) => _interpY(a, b, maxLat));
    return out;
  }

  static List<LatLng> _clipEdge(
    List<LatLng> poly,
    bool Function(LatLng) inside,
    LatLng Function(LatLng, LatLng) intersect,
  ) {
    final out = <LatLng>[];
    if (poly.isEmpty) return out;
    for (int i = 0; i < poly.length; i++) {
      final cur = poly[i];
      final prev = poly[(i - 1 + poly.length) % poly.length];
      final curIn = inside(cur);
      final prevIn = inside(prev);
      if (curIn) {
        if (!prevIn) out.add(intersect(prev, cur));
        out.add(cur);
      } else if (prevIn) {
        out.add(intersect(prev, cur));
      }
    }
    return out;
  }

  static LatLng _interpX(LatLng a, LatLng b, double x) {
    final t = (x - a.longitude) / (b.longitude - a.longitude);
    return LatLng(a.latitude + t * (b.latitude - a.latitude), x);
  }

  static LatLng _interpY(LatLng a, LatLng b, double y) {
    final t = (y - a.latitude) / (b.latitude - a.latitude);
    return LatLng(y, a.longitude + t * (b.longitude - a.longitude));
  }
}
