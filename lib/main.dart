/// CarbonCheck Field - Main entry point
/// 
/// A Flutter mobile app for farmers to analyze crop types and estimate
/// carbon credit income by drawing field boundaries on a satellite map.
/// 
/// Security: Uses Firebase Auth + secure Cloud Run backend (no keys in app!)

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:carbon_check_field/screens/home_screen.dart';
import 'package:carbon_check_field/services/firebase_service.dart';

void main() async {
  // Ensure Flutter is initialized
  WidgetsFlutterBinding.ensureInitialized();
  
  // Load environment variables from .env file
  try {
    await dotenv.load(fileName: ".env");
    print('✅ Environment variables loaded');
  } catch (e) {
    print('⚠️ Failed to load .env file: $e');
    print('⚠️ Make sure .env file exists in the project root');
    // App can still run, but API keys won't be available
  }
  
  // Initialize Firebase
  try {
    await FirebaseService.initialize();
  } catch (e) {
    print('⚠️ Firebase initialization failed: $e');
    // App can still run, but backend calls will fail
  }
  
  // Set preferred orientations (portrait only for better UX)
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  
  runApp(const CarbonCheckApp());
}

class CarbonCheckApp extends StatelessWidget {
  const CarbonCheckApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CarbonCheck Field',
      debugShowCheckedModeBanner: false,
      
      // Theme configuration
      theme: ThemeData(
        // Primary color: Farmer-friendly green
        primarySwatch: Colors.green,
        primaryColor: const Color(0xFF2E7D32),
        
        // Accent color: Sky blue
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2E7D32),
          secondary: const Color(0xFF1976D2),
        ),
        
        // Typography
        textTheme: const TextTheme(
          displayLarge: TextStyle(
            fontSize: 32,
            fontWeight: FontWeight.bold,
          ),
          headlineMedium: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.w600,
          ),
          bodyLarge: TextStyle(
            fontSize: 16,
          ),
        ),
        
        // Card theme
        cardTheme: CardThemeData(
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        
        // Button theme
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            elevation: 4,
            padding: const EdgeInsets.symmetric(
              horizontal: 24,
              vertical: 14,
            ),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
        
        // AppBar theme
        appBarTheme: const AppBarTheme(
          elevation: 0,
          centerTitle: true,
          backgroundColor: Color(0xFF2E7D32),
          foregroundColor: Colors.white,
        ),
        
        // Use Material 3
        useMaterial3: true,
      ),
      
      // Start with home screen
      home: const HomeScreen(),
    );
  }
}

