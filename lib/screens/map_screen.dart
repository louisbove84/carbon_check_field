/// Map screen - Interactive field drawing interface
/// 
/// Displays Google Maps satellite view where users can tap to create
/// polygon vertices. Shows real-time area calculation and analysis button.

import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:carbon_check_field/models/field_data.dart';
import 'package:carbon_check_field/utils/geo_utils.dart';
import 'package:carbon_check_field/utils/constants.dart';
import 'package:carbon_check_field/widgets/area_display_card.dart';
import 'package:carbon_check_field/widgets/map_instructions.dart';
import 'package:carbon_check_field/screens/results_screen.dart';
import 'package:uuid/uuid.dart';

class MapScreen extends StatefulWidget {
  const MapScreen({super.key});

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  GoogleMapController? _mapController;
  
  // User-drawn polygon points
  final List<LatLng> _polygonPoints = [];
  
  // Current polygon to display on map
  final Set<Polygon> _polygons = {};
  
  // Markers for polygon vertices
  final Set<Marker> _markers = {};
  
  // Calculated area in acres
  double _areaAcres = 0.0;
  
  // Whether user can analyze (needs at least 3 points)
  bool get _canAnalyze => _polygonPoints.length >= AppConstants.minPolygonPoints;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Draw Your Field'),
        backgroundColor: const Color(0xFF2E7D32),
        actions: [
          // Clear button
          if (_polygonPoints.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              onPressed: _clearPolygon,
              tooltip: 'Clear field',
            ),
        ],
      ),
      body: Stack(
        children: [
          // Google Maps widget
          GoogleMap(
            initialCameraPosition: const CameraPosition(
              target: LatLng(
                AppConstants.defaultLatitude,
                AppConstants.defaultLongitude,
              ),
              zoom: AppConstants.defaultZoom,
            ),
            mapType: MapType.satellite, // Satellite view for farm fields
            onMapCreated: _onMapCreated,
            onTap: _onMapTapped,
            polygons: _polygons,
            markers: _markers,
            myLocationEnabled: true,
            myLocationButtonEnabled: true,
            zoomControlsEnabled: true,
            mapToolbarEnabled: false,
          ),
          
          // Instructions overlay (top)
          const Positioned(
            top: 16,
            left: 16,
            right: 16,
            child: MapInstructions(),
          ),
          
          // Area display card (bottom)
          if (_polygonPoints.isNotEmpty)
            Positioned(
              bottom: 16,
              left: 16,
              right: 16,
              child: AreaDisplayCard(
                areaAcres: _areaAcres,
                pointCount: _polygonPoints.length,
              ),
            ),
        ],
      ),
      
      // Floating action button - Analyze field
      floatingActionButton: _canAnalyze
          ? FloatingActionButton.extended(
              onPressed: _analyzeField,
              backgroundColor: const Color(0xFF1976D2),
              icon: const Icon(Icons.analytics),
              label: const Text(
                'Analyze Field',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            )
          : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }

  /// Called when map is created
  void _onMapCreated(GoogleMapController controller) {
    _mapController = controller;
  }

  /// Handle map tap - add polygon vertex
  void _onMapTapped(LatLng position) {
    setState(() {
      // Add point to polygon
      _polygonPoints.add(position);
      
      // Update markers
      _updateMarkers();
      
      // Update polygon
      _updatePolygon();
      
      // Recalculate area
      _calculateArea();
    });
  }

  /// Update markers for each polygon vertex
  void _updateMarkers() {
    _markers.clear();
    
    for (int i = 0; i < _polygonPoints.length; i++) {
      _markers.add(
        Marker(
          markerId: MarkerId('point_$i'),
          position: _polygonPoints[i],
          icon: BitmapDescriptor.defaultMarkerWithHue(
            BitmapDescriptor.hueGreen,
          ),
          draggable: true,
          onDragEnd: (newPosition) => _onMarkerDragged(i, newPosition),
        ),
      );
    }
  }

  /// Handle marker drag - update polygon point
  void _onMarkerDragged(int index, LatLng newPosition) {
    setState(() {
      _polygonPoints[index] = newPosition;
      _updatePolygon();
      _calculateArea();
    });
  }

  /// Update the polygon shape on the map
  void _updatePolygon() {
    _polygons.clear();
    
    if (_polygonPoints.length >= 2) {
      _polygons.add(
        Polygon(
          polygonId: const PolygonId('field'),
          points: _polygonPoints,
          strokeColor: Colors.green,
          strokeWidth: 3,
          fillColor: Colors.green.withOpacity(0.2),
        ),
      );
    }
  }

  /// Calculate polygon area in acres
  void _calculateArea() {
    if (_polygonPoints.length >= 3) {
      _areaAcres = GeoUtils.calculatePolygonAreaAcres(_polygonPoints);
    } else {
      _areaAcres = 0.0;
    }
  }

  /// Clear the polygon and start over
  void _clearPolygon() {
    setState(() {
      _polygonPoints.clear();
      _markers.clear();
      _polygons.clear();
      _areaAcres = 0.0;
    });
  }

  /// Analyze the field - navigate to results screen
  void _analyzeField() {
    // Validate polygon
    final validationError = GeoUtils.validatePolygon(_polygonPoints);
    if (validationError != null) {
      _showError(validationError);
      return;
    }

    // Create FieldData object
    final fieldData = FieldData(
      id: const Uuid().v4(),
      polygonPoints: List.from(_polygonPoints),
      areaAcres: _areaAcres,
      centroid: GeoUtils.calculateCentroid(_polygonPoints),
      bounds: GeoUtils.calculateBounds(_polygonPoints),
      createdAt: DateTime.now(),
    );

    // Navigate to results screen (which will compute features & predict)
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ResultsScreen(fieldData: fieldData),
      ),
    );
  }

  /// Show error dialog
  void _showError(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Invalid Field'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }
}

