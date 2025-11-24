/// Crop Zones Map Screen - Visualize field classification results
/// 
/// Displays an interactive map showing:
/// - Field boundary
/// - Crop zones with different colors
/// - Legend explaining crop types
/// - Zone details on tap

import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:carbon_check_field/models/prediction_result.dart';
import 'package:carbon_check_field/models/crop_zone.dart';
import 'dart:math' as math;

class CropZonesMapScreen extends StatefulWidget {
  final PredictionResult result;
  final List<LatLng> fieldBoundary;

  const CropZonesMapScreen({
    super.key,
    required this.result,
    required this.fieldBoundary,
  });

  @override
  State<CropZonesMapScreen> createState() => _CropZonesMapScreenState();
}

class _CropZonesMapScreenState extends State<CropZonesMapScreen> {
  GoogleMapController? _mapController;
  Set<Polygon> _polygons = {};
  bool _showLegend = true;
  CropZone? _selectedZone;

  // Crop type colors (consistent, vibrant colors)
  static const Map<String, Color> _cropColors = {
    'Corn': Color(0xFFFFD700),           // Gold
    'Soybeans': Color(0xFF32CD32),       // Lime Green
    'Alfalfa': Color(0xFF9370DB),        // Medium Purple
    'Winter Wheat': Color(0xFFFF8C00),   // Dark Orange
  };

  @override
  void initState() {
    super.initState();
    _buildPolygons();
  }

  /// Build polygons for crop zones and field boundary
  void _buildPolygons() {
    final polygons = <Polygon>{};
    
    // Add field boundary (outline only)
    polygons.add(Polygon(
      polygonId: const PolygonId('field_boundary'),
      points: widget.fieldBoundary,
      strokeColor: Colors.black,
      strokeWidth: 3,
      fillColor: Colors.transparent,
      consumeTapEvents: false,
    ));
    
    // Add crop zones
    if (widget.result.cropZones != null) {
      for (int i = 0; i < widget.result.cropZones!.length; i++) {
        final zone = widget.result.cropZones![i];
        final color = _getCropColor(zone.crop);
        
        polygons.add(Polygon(
          polygonId: PolygonId('zone_$i'),
          points: zone.polygon,
          fillColor: color.withOpacity(0.6),
          strokeColor: color.withOpacity(0.9),
          strokeWidth: 2,
          consumeTapEvents: true,
          onTap: () => _onZoneTapped(zone),
        ));
      }
    }
    
    setState(() {
      _polygons = polygons;
    });
  }

  /// Get color for crop type
  Color _getCropColor(String cropType) {
    return _cropColors[cropType] ?? _generateColor(cropType);
  }

  /// Generate consistent color for unknown crop types
  Color _generateColor(String cropType) {
    final hash = cropType.hashCode;
    final hue = (hash % 360).toDouble();
    return HSVColor.fromAHSV(1.0, hue, 0.7, 0.9).toColor();
  }

  /// Handle zone tap
  void _onZoneTapped(CropZone zone) {
    setState(() {
      _selectedZone = zone;
    });
    
    // Show bottom sheet with zone details
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => _buildZoneDetailsSheet(zone),
    );
  }

  /// Build zone details bottom sheet
  Widget _buildZoneDetailsSheet(CropZone zone) {
    final color = _getCropColor(zone.crop);
    
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header with crop icon
          Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      zone.crop,
                      style: const TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      'Crop Zone Details',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 24),
          const Divider(),
          const SizedBox(height: 16),
          
          // Zone stats
          _buildStatRow(Icons.crop, 'Area', zone.areaFormatted),
          const SizedBox(height: 12),
          _buildStatRow(Icons.pie_chart, 'Field %', zone.percentageFormatted),
          const SizedBox(height: 12),
          _buildStatRow(Icons.assessment, 'Confidence', zone.confidencePercentage),
          
          const SizedBox(height: 24),
          
          // Close button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () => Navigator.pop(context),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF2E7D32),
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
              child: const Text('Close'),
            ),
          ),
        ],
      ),
    );
  }

  /// Build stat row
  Widget _buildStatRow(IconData icon, String label, String value) {
    return Row(
      children: [
        Icon(icon, size: 20, color: Colors.grey[700]),
        const SizedBox(width: 12),
        Text(
          label,
          style: TextStyle(
            fontSize: 16,
            color: Colors.grey[700],
          ),
        ),
        const Spacer(),
        Text(
          value,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  /// Calculate map center from field boundary
  LatLng _calculateCenter() {
    double lat = 0;
    double lng = 0;
    
    for (final point in widget.fieldBoundary) {
      lat += point.latitude;
      lng += point.longitude;
    }
    
    return LatLng(
      lat / widget.fieldBoundary.length,
      lng / widget.fieldBoundary.length,
    );
  }

  @override
  Widget build(BuildContext context) {
    final center = _calculateCenter();
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Crop Classification Map'),
        backgroundColor: const Color(0xFF2E7D32),
        actions: [
          IconButton(
            icon: Icon(_showLegend ? Icons.visibility : Icons.visibility_off),
            onPressed: () {
              setState(() {
                _showLegend = !_showLegend;
              });
            },
            tooltip: _showLegend ? 'Hide legend' : 'Show legend',
          ),
        ],
      ),
      body: Stack(
        children: [
          // Map
          GoogleMap(
            onMapCreated: (controller) {
              _mapController = controller;
            },
            initialCameraPosition: CameraPosition(
              target: center,
              zoom: 15,
            ),
            polygons: _polygons,
            mapType: MapType.satellite,
            myLocationButtonEnabled: false,
            zoomControlsEnabled: true,
            mapToolbarEnabled: false,
          ),
          
          // Legend (top right)
          if (_showLegend) _buildLegend(),
          
          // Info card (bottom)
          _buildInfoCard(),
        ],
      ),
    );
  }

  /// Build legend overlay
  Widget _buildLegend() {
    final cropTypes = widget.result.cropZones
            ?.map((z) => z.crop)
            .toSet()
            .toList() ??
        [];
    
    return Positioned(
      top: 16,
      right: 16,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.2),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Crop Types',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 14,
              ),
            ),
            const SizedBox(height: 8),
            ...cropTypes.map((crop) => _buildLegendItem(crop)),
          ],
        ),
      ),
    );
  }

  /// Build legend item
  Widget _buildLegendItem(String crop) {
    final color = _getCropColor(crop);
    
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 20,
            height: 20,
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(4),
              border: Border.all(color: color.withOpacity(0.9), width: 1),
            ),
          ),
          const SizedBox(width: 8),
          Text(
            crop,
            style: const TextStyle(fontSize: 13),
          ),
        ],
      ),
    );
  }

  /// Build info card
  Widget _buildInfoCard() {
    return Positioned(
      bottom: 16,
      left: 16,
      right: 16,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.2),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              '${widget.result.distinctCropCount} Crop Types Detected',
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Total Area: ${widget.result.areaFormatted}',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[700],
              ),
            ),
            const SizedBox(height: 4),
            Text(
              'Tap any colored zone to see details',
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey[600],
                fontStyle: FontStyle.italic,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

