/// Firebase initialization service
/// 
/// Handles Firebase app initialization and authentication setup.

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:carbon_check_field/firebase_options.dart';

class FirebaseService {
  static bool _initialized = false;

  /// Initialize Firebase
  /// 
  /// This should be called once at app startup before any Firebase services are used.
  static Future<void> initialize() async {
    if (_initialized) return;
    
    try {
      await Firebase.initializeApp(
        options: DefaultFirebaseOptions.currentPlatform,
      );
      _initialized = true;
      print('✅ Firebase initialized successfully');
    } catch (e) {
      print('❌ Firebase initialization failed: $e');
      throw Exception('Failed to initialize Firebase: $e');
    }
  }

  /// Sign in anonymously
  /// 
  /// This creates an anonymous user account for accessing the backend API.
  /// The user doesn't need to provide any credentials.
  static Future<User?> signInAnonymously() async {
    try {
      final userCredential = await FirebaseAuth.instance.signInAnonymously();
      print('✅ Signed in anonymously: ${userCredential.user?.uid}');
      return userCredential.user;
    } catch (e) {
      print('❌ Anonymous sign-in failed: $e');
      throw Exception('Anonymous sign-in failed: $e');
    }
  }

  /// Get current user
  static User? getCurrentUser() {
    return FirebaseAuth.instance.currentUser;
  }

  /// Check if user is signed in
  static bool isSignedIn() {
    return FirebaseAuth.instance.currentUser != null;
  }

  /// Sign out
  static Future<void> signOut() async {
    await FirebaseAuth.instance.signOut();
    print('✅ Signed out');
  }
}

