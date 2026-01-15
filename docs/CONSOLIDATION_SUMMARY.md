# Documentation Consolidation Summary

## Overview

Markdown files have been consolidated into a clearer structure for easier navigation and maintenance.

## Changes Made

### Consolidated Files

1. **iOS Development** → `docs/IOS_DEVELOPMENT.md`
   - Merged `IOS_BUILD_GUIDE.md` (App Store builds)
   - Merged `IOS_SIMULATOR_GUIDE.md` (Simulator testing)
   - Single comprehensive guide for all iOS development needs

2. **Installation** → `docs/INSTALLATION.md`
   - Merged `INSTALL.md` (Installation guide)
   - Merged `SETUP_PYTHON313.md` (Python 3.13 setup)
   - Merged `README_VENV.md` (Virtual environment setup)
   - Single comprehensive installation guide

### Archived Files

- `CLEANUP_SUMMARY.md` → `docs/archive/CLEANUP_SUMMARY.md`
  - Historical cleanup documentation moved to archive

### Deleted Files

- `IOS_BUILD_GUIDE.md` (merged into `docs/IOS_DEVELOPMENT.md`)
- `IOS_SIMULATOR_GUIDE.md` (merged into `docs/IOS_DEVELOPMENT.md`)
- `INSTALL.md` (merged into `docs/INSTALLATION.md`)
- `SETUP_PYTHON313.md` (merged into `docs/INSTALLATION.md`)
- `README_VENV.md` (merged into `docs/INSTALLATION.md`)

## New Structure

```
carbon_check_field/
├── README.md                    # Main project overview
├── DEPLOYMENT.md               # GCP deployment structure
├── docs/
│   ├── IOS_DEVELOPMENT.md      # iOS simulator + build guide (NEW)
│   ├── INSTALLATION.md         # Python/ML setup guide (NEW)
│   ├── archive/
│   │   └── CLEANUP_SUMMARY.md  # Historical docs
│   └── [other docs...]
```

## Updated References

- `README.md` now links to consolidated docs
- `DEPLOYMENT.md` references installation guide
- All broken links have been fixed

## Benefits

1. **Single source of truth** - Each topic has one comprehensive guide
2. **Easier navigation** - Clear structure in `docs/` folder
3. **Less duplication** - Related content consolidated
4. **Better maintenance** - Fewer files to update

## Migration Notes

If you have bookmarks or references to old files:
- `IOS_BUILD_GUIDE.md` → `docs/IOS_DEVELOPMENT.md#building-for-app-store`
- `IOS_SIMULATOR_GUIDE.md` → `docs/IOS_DEVELOPMENT.md#running-on-ios-simulator`
- `INSTALL.md` → `docs/INSTALLATION.md`
- `SETUP_PYTHON313.md` → `docs/INSTALLATION.md#python-setup`
- `README_VENV.md` → `docs/INSTALLATION.md#using-the-virtual-environment`
