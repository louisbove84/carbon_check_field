/// Instructional overlay for the map screen
/// 
/// Shows users how to draw their field polygon.

import 'package:flutter/material.dart';

class MapInstructions extends StatelessWidget {
  const MapInstructions({super.key});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Padding(
        padding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
        child: Row(
          children: [
            Icon(
              Icons.touch_app,
              color: Color(0xFF2E7D32),
              size: 24,
            ),
            SizedBox(width: 12),
            Expanded(
              child: Text(
                'Tap corners of your field to draw boundary',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

