/// Map screen - Interactive field drawing interface
/// 
/// Displays Google Maps satellite view where users can tap to create
/// polygon vertices. Shows real-time area calculation and analysis button.
/// Includes address search for easy navigation to farm location.

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:http/http.dart' as http;
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
  final TextEditingController _searchController = TextEditingController();
  
  // User-drawn polygon points
  final List<LatLng> _polygonPoints = [];
  
  // Current polygon to display on map
  final Set<Polygon> _polygons = {};
  
  // Markers for polygon vertices
  final Set<Marker> _markers = {};
  
  // Calculated area in acres
  double _areaAcres = 0.0;
  
  // Search state
  bool _isSearching = false;
  
  // Whether user can analyze (needs at least 3 points)
  bool get _canAnalyze => _polygonPoints.length >= AppConstants.minPolygonPoints;

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

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
          
          // Address search bar (top)
          Positioned(
            top: 16,
            left: 16,
            right: 16,
            child: _buildSearchBar(),
          ),
          
          // Instructions overlay (below search)
          const Positioned(
            top: 80,
            left: 16,
            right: 16,
            child: MapInstructions(),
          ),
          
          // Bottom bar with area info and analyze button (always visible)
          Positioned(
            bottom: 16,
            left: 16,
            right: 16,
            child: Row(
              children: [
                // Area display card on the left
                Expanded(
                  flex: 3,
                  child: _polygonPoints.isNotEmpty
                      ? AreaDisplayCard(
                          areaAcres: _areaAcres,
                          pointCount: _polygonPoints.length,
                        )
                      : Card(
                          elevation: 8,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                          child: const Padding(
                            padding: EdgeInsets.symmetric(horizontal: 12.0, vertical: 20.0),
                            child: Center(
                              child: Text(
                                'Tap to draw',
                                style: TextStyle(
                                  fontSize: 14,
                                  color: Colors.grey,
                                ),
                              ),
                            ),
                          ),
                        ),
                ),
                
                const SizedBox(width: 12),
                
                // Analyze button on the right (always visible, disabled until ready)
                Expanded(
                  flex: 2,
                  child: ElevatedButton(
                    onPressed: _canAnalyze ? _analyzeField : null,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _canAnalyze ? const Color(0xFF1976D2) : Colors.grey[300],
                      foregroundColor: _canAnalyze ? Colors.white : Colors.grey[500],
                      padding: const EdgeInsets.symmetric(vertical: 20),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                      elevation: _canAnalyze ? 8 : 2,
                      disabledBackgroundColor: Colors.grey[300],
                      disabledForegroundColor: Colors.grey[500],
                    ),
                    child: const Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.analytics, size: 28),
                        SizedBox(height: 4),
                        Text(
                          'Analyze',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
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

  /// Build address search bar
  Widget _buildSearchBar() {
    return Card(
      elevation: 8,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(30),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16.0),
        child: Row(
          children: [
            const Icon(Icons.search, color: Colors.grey),
            const SizedBox(width: 12),
            Expanded(
              child: TextField(
                controller: _searchController,
                decoration: const InputDecoration(
                  hintText: 'Search address (e.g., "123 Farm Rd, Iowa")',
                  border: InputBorder.none,
                ),
                onSubmitted: (_) => _searchAddress(),
              ),
            ),
            if (_isSearching)
              const SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(strokeWidth: 2),
              )
            else
              IconButton(
                icon: const Icon(Icons.my_location, color: Color(0xFF2E7D32)),
                onPressed: _searchAddress,
                tooltip: 'Go to address',
              ),
          ],
        ),
      ),
    );
  }

  /// Search for address and move map using Google Geocoding API
  Future<void> _searchAddress() async {
    final address = _searchController.text.trim();
    
    if (address.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please enter an address'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }

    setState(() => _isSearching = true);

    try {
      // Use Google Geocoding API (works on web!)
      final encodedAddress = Uri.encodeComponent(address);
      final url = Uri.parse(
        'https://maps.googleapis.com/maps/api/geocode/json?address=$encodedAddress&key=${AppConstants.googleMapsApiKey}'
      );
      
      final response = await http.get(url).timeout(const Duration(seconds: 10));
      
      if (response.statusCode != 200) {
        throw Exception('Geocoding service error');
      }
      
      final data = json.decode(response.body);
      
      if (data['status'] != 'OK' || data['results'].isEmpty) {
        throw Exception('Address not found. Try being more specific.');
      }
      
      // Get the first result's location
      final location = data['results'][0]['geometry']['location'];
      final lat = location['lat'] as double;
      final lng = location['lng'] as double;
      final position = LatLng(lat, lng);
      
      // Get formatted address for display
      final formattedAddress = data['results'][0]['formatted_address'] as String;

      // Animate map to the location
      _mapController?.animateCamera(
        CameraUpdate.newLatLngZoom(position, 16), // Zoom level 16 for farm fields
      );

      // Show success message
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Found: $formattedAddress'),
            duration: const Duration(seconds: 3),
            backgroundColor: Colors.green,
          ),
        );
      }
      
      // Clear search text
      _searchController.clear();
      
      // Unfocus keyboard
      FocusScope.of(context).unfocus();
      
    } catch (e) {
      // Show error message
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Could not find address. Please try again.'),
            duration: const Duration(seconds: 3),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isSearching = false);
      }
    }
  }
}

