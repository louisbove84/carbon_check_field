#!/bin/bash
# Generate web/config.js from .env file

if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "   Please create .env file from .env.example"
    exit 1
fi

# Extract Google Maps API key from .env
API_KEY=$(grep "^GOOGLE_MAPS_API_KEY=" .env | cut -d '=' -f2)

if [ -z "$API_KEY" ]; then
    echo "❌ Error: GOOGLE_MAPS_API_KEY not found in .env"
    exit 1
fi

# Generate config.js
cat > web/config.js << JS_EOF
// CarbonCheck Field - Web Configuration
// ======================================
// This file is auto-generated from .env
// DO NOT edit manually - run generate_web_config.sh instead

var GOOGLE_MAPS_API_KEY = '$API_KEY';
JS_EOF

echo "✅ Generated web/config.js from .env"
