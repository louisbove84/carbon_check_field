/// Loading overlay widget with status message
/// 
/// Displays a semi-transparent overlay with loading spinner and text.

import 'package:flutter/material.dart';

class LoadingOverlay extends StatelessWidget {
  final String message;
  final Widget? child;

  const LoadingOverlay({
    super.key,
    required this.message,
    this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // Background (optional child widget)
        if (child != null) child!,
        
        // Overlay
        Container(
          color: Colors.black54,
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Loading indicator
                child ?? const SizedBox.shrink(),
                
                const SizedBox(height: 24),
                
                // Status message
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24,
                    vertical: 16,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    message,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

