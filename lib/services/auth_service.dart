/// Authentication service for Google Cloud APIs
/// 
/// Handles service account JWT generation and OAuth2 token retrieval
/// for both Earth Engine and Vertex AI API calls.

import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:carbon_check_field/utils/constants.dart';
import 'package:flutter/services.dart' show rootBundle;

class AuthService {
  /// Cached access token
  String? _cachedToken;
  
  /// Token expiration time
  DateTime? _tokenExpiry;
  
  /// Service account credentials loaded from assets
  Map<String, dynamic>? _serviceAccount;

  /// Load service account JSON from assets folder
  /// 
  /// This should be called once at app startup.
  /// The JSON file contains private_key, client_email, etc.
  Future<void> initialize() async {
    try {
      final jsonString = await rootBundle.loadString('assets/service-account.json');
      _serviceAccount = json.decode(jsonString) as Map<String, dynamic>;
    } catch (e) {
      throw Exception('Failed to load service account credentials: $e');
    }
  }

  /// Get a valid OAuth2 access token
  /// 
  /// Returns cached token if still valid, otherwise requests a new one.
  /// Tokens typically expire after 1 hour.
  Future<String> getAccessToken() async {
    // Return cached token if still valid (with 5-minute buffer)
    if (_cachedToken != null &&
        _tokenExpiry != null &&
        DateTime.now().isBefore(_tokenExpiry!.subtract(const Duration(minutes: 5)))) {
      return _cachedToken!;
    }

    // Request new token
    _cachedToken = await _requestNewToken();
    _tokenExpiry = DateTime.now().add(const Duration(hours: 1));
    
    return _cachedToken!;
  }

  /// Request a new access token using service account credentials
  /// 
  /// This uses the OAuth2 service account flow with JWT assertion.
  /// Scopes include Earth Engine and AI Platform.
  Future<String> _requestNewToken() async {
    if (_serviceAccount == null) {
      await initialize();
    }

    // Build JWT assertion
    final jwt = _createJwt();

    // Request token from Google OAuth2 endpoint
    final response = await http.post(
      Uri.parse(AppConstants.oauth2TokenUrl),
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: {
        'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
        'assertion': jwt,
      },
    );

    if (response.statusCode != 200) {
      throw Exception('OAuth2 token request failed: ${response.body}');
    }

    final data = json.decode(response.body) as Map<String, dynamic>;
    return data['access_token'] as String;
  }

  /// Create JWT (JSON Web Token) for service account authentication
  /// 
  /// This is a simplified implementation. In production, use a proper
  /// JWT library like `dart_jsonwebtoken` or `jose` package.
  String _createJwt() {
    final now = DateTime.now().millisecondsSinceEpoch ~/ 1000;
    
    // JWT Header
    final header = {
      'alg': 'RS256',
      'typ': 'JWT',
    };

    // JWT Claim Set
    final claimSet = {
      'iss': _serviceAccount!['client_email'],
      'scope': 'https://www.googleapis.com/auth/earthengine '
          'https://www.googleapis.com/auth/cloud-platform',
      'aud': AppConstants.oauth2TokenUrl,
      'exp': now + 3600, // 1 hour from now
      'iat': now,
    };

    // Base64url encode header and claim set
    final encodedHeader = _base64UrlEncode(json.encode(header));
    final encodedClaim = _base64UrlEncode(json.encode(claimSet));

    // Create signature (simplified - in production use a crypto library)
    final message = '$encodedHeader.$encodedClaim';
    final signature = _signMessage(message, _serviceAccount!['private_key']);

    return '$message.$signature';
  }

  /// Base64url encode (without padding)
  String _base64UrlEncode(String input) {
    return base64Url.encode(utf8.encode(input)).replaceAll('=', '');
  }

  /// Sign message with private key (RSA-SHA256)
  /// 
  /// NOTE: This is a placeholder. In production, you MUST use a proper
  /// crypto library like `pointycastle` or call a backend service to sign.
  /// 
  /// For MVP, you can temporarily use Firebase Functions or Cloud Run
  /// to sign the JWT server-side and return the token.
  String _signMessage(String message, String privateKey) {
    // TODO: Implement RSA-SHA256 signing with pointycastle
    // For now, this is a placeholder that will fail authentication.
    // You have two options:
    // 1. Add pointycastle dependency and implement proper RSA signing
    // 2. Create a simple Cloud Function to handle token generation
    
    throw UnimplementedError(
      'JWT signing not implemented. Please use a backend service '
      'or add pointycastle library for RSA signing.'
    );
  }

  /// Clear cached token (useful for testing or logout)
  void clearCache() {
    _cachedToken = null;
    _tokenExpiry = null;
  }
}

