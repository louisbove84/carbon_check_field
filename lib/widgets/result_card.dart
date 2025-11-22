/// Result card widget displaying crop prediction and CO₂ income
/// 
/// Shows the final analysis results in a beautiful, shareable format.

import 'package:flutter/material.dart';
import 'package:carbon_check_field/models/prediction_result.dart';

class ResultCard extends StatelessWidget {
  final PredictionResult result;

  const ResultCard({
    super.key,
    required this.result,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 8,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
      ),
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            const Text(
              'Field Analysis Results',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Color(0xFF2E7D32),
              ),
            ),
            
            const SizedBox(height: 24),
            
            // Crop type (Model Prediction)
            _buildResultRow(
              icon: Icons.psychology,
              label: 'Model Prediction',
              value: result.cropType,
              valueColor: const Color(0xFF2E7D32),
            ),
            
            const SizedBox(height: 16),
            
            // Confidence
            _buildResultRow(
              icon: Icons.insights,
              label: 'Confidence',
              value: result.confidencePercentage,
              valueColor: Colors.blue,
            ),
            
            const SizedBox(height: 16),
            
            // CDL Ground Truth (if available)
            if (result.cdlCropType != null) ...[
              _buildCDLRow(),
              const SizedBox(height: 16),
            ],
            
            // Field area
            _buildResultRow(
              icon: Icons.square_foot,
              label: 'Field Area',
              value: result.areaFormatted,
              valueColor: Colors.grey[700]!,
            ),
            
            const Divider(height: 32),
            
            // CO₂ Income section
            const Row(
              children: [
                Icon(
                  Icons.eco,
                  color: Colors.green,
                  size: 28,
                ),
                SizedBox(width: 12),
                Text(
                  'Estimated CO₂ Income',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 16),
            
            // Income range
            Center(
              child: Column(
                children: [
                  Text(
                    result.incomeRangeFormatted,
                    style: const TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF2E7D32),
                    ),
                  ),
                  const Text(
                    'per year',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.grey,
                    ),
                  ),
                  
                  const SizedBox(height: 12),
                  
                  // Average
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.green[50],
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      'Average: ${result.incomeAverageFormatted}/year',
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                        color: Color(0xFF2E7D32),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Disclaimer
            Text(
              '* Rates based on 2025 Indigo Ag and Truterra carbon credit markets. '
              'Actual income depends on farming practices, verification, and market conditions.',
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

  /// Build CDL ground truth row with agreement indicator
  Widget _buildCDLRow() {
    final agreementIcon = result.cdlAgreement 
        ? const Icon(Icons.check_circle, color: Colors.green, size: 24)
        : const Icon(Icons.info_outline, color: Colors.orange, size: 24);
    
    final agreementText = result.cdlAgreement 
        ? 'Matches model ✓'
        : 'Different from model';
    
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: result.cdlAgreement ? Colors.green[50] : Colors.orange[50],
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: result.cdlAgreement ? Colors.green[200]! : Colors.orange[200]!,
          width: 2,
        ),
      ),
      child: Row(
        children: [
          Icon(Icons.satellite_alt, color: Colors.grey[700], size: 24),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'CDL Ground Truth (USDA)',
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey[700],
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Text(
                      result.cdlCropType!,
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF1976D2),
                      ),
                    ),
                    const SizedBox(width: 8),
                    agreementIcon,
                  ],
                ),
                const SizedBox(height: 2),
                Text(
                  agreementText,
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey[600],
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Build a result row with icon, label, and value
  Widget _buildResultRow({
    required IconData icon,
    required String label,
    required String value,
    required Color valueColor,
  }) {
    return Row(
      children: [
        Icon(icon, color: Colors.grey[600], size: 24),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey[600],
                ),
              ),
              const SizedBox(height: 4),
              Text(
                value,
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: valueColor,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

