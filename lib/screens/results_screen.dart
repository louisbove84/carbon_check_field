/// Results screen - Display crop prediction and CO₂ income estimates
/// 
/// This screen orchestrates the entire analysis pipeline:
/// 1. Compute NDVI features via Earth Engine
/// 2. Send features to Vertex AI for prediction
/// 3. Display results with carbon credit income estimates

import 'package:flutter/material.dart';
import 'package:carbon_check_field/models/field_data.dart';
import 'package:carbon_check_field/models/prediction_result.dart';
import 'package:carbon_check_field/services/backend_service.dart';
import 'package:carbon_check_field/services/firebase_service.dart';
import 'package:carbon_check_field/widgets/loading_overlay.dart';
import 'package:carbon_check_field/widgets/result_card.dart';
import 'package:carbon_check_field/screens/crop_zones_map_screen.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:share_plus/share_plus.dart';

class ResultsScreen extends StatefulWidget {
  final FieldData fieldData;

  const ResultsScreen({
    super.key,
    required this.fieldData,
  });

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  // Backend service (secure - no keys in app!)
  late final BackendService _backendService;
  
  // State
  bool _isLoading = true;
  String _loadingMessage = 'Initializing...';
  PredictionResult? _result;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    
    // Initialize backend service (secure - uses Firebase Auth!)
    _backendService = BackendService();
    
    // Start analysis pipeline
    _runAnalysis();
  }

  /// Run the full analysis pipeline via secure backend
  Future<void> _runAnalysis() async {
    try {
      // Step 1: Ensure Firebase is initialized
      setState(() {
        _loadingMessage = 'Connecting securely...';
      });
      
      if (!FirebaseService.isSignedIn()) {
        await FirebaseService.signInAnonymously();
      }
      
      // Step 2: Send field to backend for analysis
      setState(() {
        _loadingMessage = 'Analyzing satellite imagery (2024)...\nThis may take 10-30 seconds.';
      });
      
      final result = await _backendService.analyzeField(widget.fieldData);
      
      // Step 3: Display results
      setState(() {
        _isLoading = false;
        _result = result;
      });
      
    } catch (e) {
      // Handle errors gracefully
      setState(() {
        _isLoading = false;
        _errorMessage = _parseErrorMessage(e.toString());
      });
    }
  }

  /// Parse error message into user-friendly text
  String _parseErrorMessage(String error) {
    if (error.contains('satellite') || error.contains('imagery')) {
      return 'No satellite data available for this location. '
          'Try a different field or check back later.';
    }
    if (error.contains('auth') || error.contains('401')) {
      return 'Authentication failed. Please check your service account credentials.';
    }
    if (error.contains('network') || error.contains('timeout')) {
      return 'Network error. Please check your internet connection and try again.';
    }
    return 'Analysis failed: $error\n\nPlease try again or contact support.';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Field Analysis'),
        backgroundColor: const Color(0xFF2E7D32),
        actions: [
          // Share button (only show when results are ready)
          if (_result != null)
            IconButton(
              icon: const Icon(Icons.share),
              onPressed: _shareResults,
              tooltip: 'Share results',
            ),
        ],
      ),
      body: _buildBody(),
    );
  }

  /// Build main body content based on current state
  Widget _buildBody() {
    if (_isLoading) {
      return _buildLoadingView();
    }
    
    if (_errorMessage != null) {
      return _buildErrorView();
    }
    
    if (_result != null) {
      return _buildResultsView();
    }
    
    return const Center(child: Text('Unknown state'));
  }

  /// Loading view with spinner and status message
  Widget _buildLoadingView() {
    return LoadingOverlay(
      message: _loadingMessage,
      child: const Center(
        child: SpinKitFadingCircle(
          color: Color(0xFF2E7D32),
          size: 60.0,
        ),
      ),
    );
  }

  /// Error view with retry button
  Widget _buildErrorView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(
              Icons.error_outline,
              size: 80,
              color: Colors.red,
            ),
            const SizedBox(height: 24),
            Text(
              _errorMessage!,
              style: const TextStyle(fontSize: 16),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            ElevatedButton.icon(
              onPressed: () {
                setState(() {
                  _isLoading = true;
                  _errorMessage = null;
                });
                _runAnalysis();
              },
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF2E7D32),
                padding: const EdgeInsets.symmetric(
                  horizontal: 32,
                  vertical: 16,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Results view with prediction and CO₂ income
  Widget _buildResultsView() {
    return SingleChildScrollView(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Success icon
            const Icon(
              Icons.check_circle,
              size: 80,
              color: Colors.green,
            ),
            
            const SizedBox(height: 24),
            
            // Results card
            ResultCard(result: _result!),
            
            const SizedBox(height: 24),
            
            // Show crop zones button (only for multi-zone results)
            if (_result!.hasMultipleZones)
              Column(
                children: [
                  ElevatedButton.icon(
                    onPressed: _showCropZonesMap,
                    icon: const Icon(Icons.map),
                    label: Text('View ${_result!.distinctCropCount} Crop Zones on Map'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF2E7D32),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                      minimumSize: const Size(double.infinity, 50),
                    ),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Tap to see how your field was divided into crop zones',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[600],
                      fontStyle: FontStyle.italic,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 24),
                ],
              ),
            
            // Action buttons
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.arrow_back),
                    label: const Text('New Field'),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _shareResults,
                    icon: const Icon(Icons.share),
                    label: const Text('Share'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF1976D2),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  /// Share results via system share sheet
  void _shareResults() {
    if (_result != null) {
      Share.share(_result!.shareableText);
    }
  }
  
  /// Show crop zones on interactive map
  void _showCropZonesMap() {
    if (_result != null && _result!.hasMultipleZones) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => CropZonesMapScreen(
            result: _result!,
            fieldBoundary: widget.fieldData.polygonPoints,
          ),
        ),
      );
    }
  }
}

