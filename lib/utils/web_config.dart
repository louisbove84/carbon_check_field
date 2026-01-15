import 'dart:js_util' as js_util;

String? getMapsKeyFromWindow() {
  final value = js_util.getProperty(js_util.globalThis, 'GOOGLE_MAPS_API_KEY');
  if (value is String && value.isNotEmpty) {
    return value;
  }
  return null;
}
