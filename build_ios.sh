#!/bin/bash

# iOS Build Script - Prepares Flutter app for Xcode Archive
# After running this, use Xcode GUI to create archive and upload to App Store

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🍎 iOS Build Script - Preparing for Xcode Archive${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Flutter is installed
if ! command -v flutter &> /dev/null; then
    echo -e "${RED}❌ Flutter not found. Please install Flutter first.${NC}"
    exit 1
fi

# Parse command line arguments
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            echo "Usage: $0 [--clean]"
            exit 1
            ;;
    esac
done

# Step 1: Clean (if requested)
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}🧹 Cleaning build artifacts...${NC}"
    flutter clean
    rm -rf build/ios
    rm -rf ios/build
    echo -e "${GREEN}✅ Clean complete${NC}"
    echo ""
fi

# Step 2: Get dependencies
echo -e "${BLUE}📦 Getting Flutter dependencies...${NC}"
flutter pub get
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Step 3: Install CocoaPods dependencies
echo -e "${BLUE}📦 Installing CocoaPods dependencies...${NC}"
cd ios
if [ ! -d "Pods" ]; then
    pod install
else
    pod install --repo-update
fi
cd ..
echo -e "${GREEN}✅ CocoaPods installed${NC}"
echo ""

# Step 4: Clean extended attributes before build
echo -e "${BLUE}🧹 Cleaning extended attributes...${NC}"
if [ -d "build/ios" ]; then
    xattr -rc build/ios 2>/dev/null || true
fi
if [ -d "ios/build" ]; then
    xattr -rc ios/build 2>/dev/null || true
fi

# Step 5: Build Flutter app for iOS release
echo -e "${BLUE}🔨 Building Flutter app for iOS release...${NC}"
flutter build ios --release --no-codesign

# Step 6: Clean extended attributes after build (fixes code signing issues)
echo -e "${BLUE}🧹 Cleaning extended attributes from Flutter framework...${NC}"
if [ -d "build/ios/Release-iphoneos" ]; then
    xattr -rc build/ios/Release-iphoneos 2>/dev/null || true
    if [ -d "build/ios/Release-iphoneos/Flutter.framework" ]; then
        xattr -rc build/ios/Release-iphoneos/Flutter.framework 2>/dev/null || true
        if [ -f "build/ios/Release-iphoneos/Flutter.framework/Flutter" ]; then
            xattr -c build/ios/Release-iphoneos/Flutter.framework/Flutter 2>/dev/null || true
        fi
    fi
fi

echo -e "${GREEN}✅ Flutter build complete${NC}"
echo ""

# Step 7: Open Xcode for archiving
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "  1. Xcode should open automatically"
echo "  2. Select 'Any iOS Device' or 'Generic iOS Device'"
echo "  3. Product → Archive"
echo "  4. In Organizer, click 'Distribute App' → 'App Store Connect' → 'Upload'"
echo ""

if command -v open &> /dev/null; then
    echo -e "${BLUE}🚀 Opening Xcode workspace...${NC}"
    open ios/Runner.xcworkspace
    echo -e "${GREEN}✅ Ready to archive in Xcode!${NC}"
else
    echo -e "${YELLOW}⚠️  Please open Xcode manually:${NC}"
    echo "  open ios/Runner.xcworkspace"
fi
