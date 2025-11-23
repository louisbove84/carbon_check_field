/// Widget displaying the calculated field area in acres
/// 
/// Shows at the bottom of the map screen as user draws the polygon.

import 'package:flutter/material.dart';

class AreaDisplayCard extends StatelessWidget {
  final double areaAcres;
  final int pointCount;

  const AreaDisplayCard({
    super.key,
    required this.areaAcres,
    required this.pointCount,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 8,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 16.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Area value
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.baseline,
              textBaseline: TextBaseline.alphabetic,
              children: [
                Text(
                  areaAcres.toStringAsFixed(1),
                  style: const TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF2E7D32),
                  ),
                ),
                const SizedBox(width: 6),
                const Text(
                  'acres',
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.grey,
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 4),
            
            // Point count
            Text(
              '$pointCount points',
              style: const TextStyle(
                fontSize: 12,
                color: Colors.grey,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

