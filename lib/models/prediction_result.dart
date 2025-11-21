/// Data model for crop prediction results and carbon credit estimates

/// Contains the predicted crop type, confidence, and financial projections
class PredictionResult {
  /// Predicted crop type (e.g., "Corn", "Soybeans")
  final String cropType;
  
  /// Model confidence score (0.0 to 1.0)
  final double confidence;
  
  /// Field area in acres
  final double areaAcres;
  
  /// Minimum estimated CO₂ income per year (USD)
  final double carbonIncomeMin;
  
  /// Maximum estimated CO₂ income per year (USD)
  final double carbonIncomeMax;
  
  /// Average estimated CO₂ income per year (USD)
  final double carbonIncomeAverage;
  
  /// Timestamp of prediction
  final DateTime predictedAt;

  PredictionResult({
    required this.cropType,
    required this.confidence,
    required this.areaAcres,
    required this.carbonIncomeMin,
    required this.carbonIncomeMax,
    required this.carbonIncomeAverage,
    required this.predictedAt,
  });

  /// Format confidence as percentage string
  String get confidencePercentage => '${(confidence * 100).toStringAsFixed(0)}%';

  /// Format income range as currency string
  String get incomeRangeFormatted =>
      '\$${carbonIncomeMin.toStringAsFixed(0)} – \$${carbonIncomeMax.toStringAsFixed(0)}';

  /// Format average income as currency string
  String get incomeAverageFormatted =>
      '\$${carbonIncomeAverage.toStringAsFixed(0)}';

  /// Format area with 1 decimal place
  String get areaFormatted => '${areaAcres.toStringAsFixed(1)} acres';

  /// Generate shareable text summary
  String get shareableText => '''
🌾 CarbonCheck Field Results

Crop Type: $cropType ($confidencePercentage confidence)
Field Area: $areaFormatted
Estimated CO₂ Income: $incomeRangeFormatted/year
Average: $incomeAverageFormatted/year

Analyzed on ${predictedAt.toString().split(' ')[0]}
  ''';

  /// Convert to JSON for storage
  Map<String, dynamic> toJson() {
    return {
      'cropType': cropType,
      'confidence': confidence,
      'areaAcres': areaAcres,
      'carbonIncomeMin': carbonIncomeMin,
      'carbonIncomeMax': carbonIncomeMax,
      'carbonIncomeAverage': carbonIncomeAverage,
      'predictedAt': predictedAt.toIso8601String(),
    };
  }

  /// Create from JSON
  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      cropType: json['cropType'] as String,
      confidence: json['confidence'] as double,
      areaAcres: json['areaAcres'] as double,
      carbonIncomeMin: json['carbonIncomeMin'] as double,
      carbonIncomeMax: json['carbonIncomeMax'] as double,
      carbonIncomeAverage: json['carbonIncomeAverage'] as double,
      predictedAt: DateTime.parse(json['predictedAt'] as String),
    );
  }
}

