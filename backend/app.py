"""
CarbonCheck Field - Secure Backend API
======================================

A FastAPI backend for secure crop classification and carbon credit estimation.

Features:
- Uses Application Default Credentials (no service account keys!)
- Calls Earth Engine for NDVI feature computation
- Calls Vertex AI for crop prediction
- Firebase Auth token verification
- CORS enabled for Flutter mobile app

Author: CarbonCheck Team
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict
import ee
from google.cloud import aiplatform
from google.auth.transport import requests
from google.oauth2 import id_token
import uvicorn
import os
from datetime import datetime
import math
import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

# Initialize FastAPI
app = FastAPI(
    title="CarbonCheck Field API",
    description="Secure backend for crop classification and carbon credit estimation",
    version="1.0.0"
)

# CORS - Allow Flutter app to call this API
# Note: When allow_credentials=True, cannot use allow_origins=["*"]
# Must explicitly list allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:3000",
        # Add your production domain here when deploying
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize Earth Engine with Application Default Credentials
try:
    ee.Initialize()
    print("✅ Earth Engine initialized successfully")
except Exception as e:
    print(f"⚠️ Earth Engine initialization failed: {e}")
    print("ℹ️ Will attempt to initialize on first request")

# Configuration
GCP_PROJECT_ID = "ml-pipeline-477612"  # Vertex AI is in this project
FIREBASE_PROJECT_ID = "carbon-check-field"  # Firebase Auth is in this project
VERTEX_AI_ENDPOINT = (
    "projects/ml-pipeline-477612/locations/us-central1/"
    "endpoints/2450616804754587648"  # crop-endpoint (active)
)
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

# CO₂ carbon credit rates ($/acre/year) - 2025 market rates
CARBON_RATES = {
    "Corn": {"min": 12.0, "max": 18.0},
    "Soybeans": {"min": 15.0, "max": 22.0},
    "Alfalfa": {"min": 18.0, "max": 25.0},
    "Winter Wheat": {"min": 10.0, "max": 15.0},
}
DEFAULT_RATE = {"min": 10.0, "max": 20.0}


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class LatLng(BaseModel):
    """Geographic coordinate (latitude, longitude)"""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lng: float = Field(..., ge=-180, le=180, description="Longitude")


class AnalyzeFieldRequest(BaseModel):
    """Request to analyze a farm field"""
    polygon: List[LatLng] = Field(..., min_items=3, max_items=100)
    year: Optional[int] = Field(2024, ge=2015, le=2030)


class CropZone(BaseModel):
    """Individual crop zone within a field"""
    crop: str
    confidence: float
    area_acres: float
    percentage: float
    polygon: List[List[float]]  # [[lng, lat], ...]


class CO2IncomeByCrop(BaseModel):
    """CO₂ income breakdown by crop type"""
    crop: str
    min: float
    max: float
    avg: float


class CO2IncomeTotal(BaseModel):
    """Total CO₂ income across all crop zones"""
    total_min: float
    total_max: float
    total_avg: float
    by_crop: List[CO2IncomeByCrop]


class FieldSummary(BaseModel):
    """Summary of field analysis"""
    total_area_acres: float
    grid_cell_size_meters: Optional[int] = None
    total_cells: Optional[int] = None


class AnalyzeFieldResponse(BaseModel):
    """Analysis results with crop prediction and CO₂ income"""
    # Legacy fields for backward compatibility (single prediction)
    crop: Optional[str] = None
    confidence: Optional[float] = None
    cdl_crop: Optional[str] = None  # CDL ground truth crop type
    cdl_agreement: Optional[bool] = False  # Whether model and CDL agree
    area_acres: float
    co2_income_min: float
    co2_income_max: float
    co2_income_avg: float
    features: Optional[List[float]] = None
    timestamp: str
    
    # New grid-based fields
    field_summary: Optional[FieldSummary] = None
    crop_zones: Optional[List[CropZone]] = None
    co2_income: Optional[CO2IncomeTotal] = None


# ============================================================================
# AUTHENTICATION
# ============================================================================

async def verify_firebase_token(authorization: str = Header(None)) -> dict:
    """
    Verify Firebase ID token from Authorization header.
    
    Header format: "Bearer <token>"
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    try:
        # Extract token from "Bearer <token>"
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid Authorization format")
        
        token = authorization.split("Bearer ")[1]
        
        # Verify the token
        decoded_token = id_token.verify_firebase_token(
            token, 
            requests.Request(),
            audience=FIREBASE_PROJECT_ID
        )
        
        return decoded_token
    
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")


# ============================================================================
# GEOSPATIAL UTILITIES
# ============================================================================

def calculate_polygon_area_acres(coords: List[Tuple[float, float]]) -> float:
    """
    Calculate polygon area in acres using the Shoelace formula.
    
    Args:
        coords: List of (lng, lat) tuples
    
    Returns:
        Area in acres
    """
    import math
    
    if len(coords) < 3:
        return 0.0
    
    # Earth radius in meters
    earth_radius = 6378137.0
    
    area = 0.0
    
    for i in range(len(coords)):
        lng1, lat1 = coords[i]
        lng2, lat2 = coords[(i + 1) % len(coords)]
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lng1_rad = math.radians(lng1)
        lng2_rad = math.radians(lng2)
        
        area += (lng2_rad - lng1_rad) * (2 + math.sin(lat1_rad) + math.sin(lat2_rad))
    
    area = abs(area) * earth_radius * earth_radius / 2.0
    
    # Convert square meters to acres
    return area * 0.000247105


# CDL Crop Code Mapping (USDA Cropland Data Layer)
CDL_CROP_MAPPING = {
    1: "Corn",
    5: "Soybeans",
    36: "Alfalfa",
    24: "Winter Wheat",
    23: "Spring Wheat",
    4: "Sorghum",
    28: "Oats",
    # Add more as needed
}

# ============================================================================
# EARTH ENGINE - CDL CROP TYPE LOOKUP
# ============================================================================

def get_cdl_crop_type(polygon_coords: List[Tuple[float, float]], year: int) -> Optional[str]:
    """
    Get the dominant crop type from USDA Cropland Data Layer (CDL).
    
    Args:
        polygon_coords: List of (lng, lat) tuples
        year: Year for CDL data (e.g., 2024)
    
    Returns:
        Crop name from CDL or None if unavailable
    """
    try:
        # Ensure Earth Engine is initialized
        if not ee.data._initialized:
            ee.Initialize()
        
        # Create polygon
        ee_polygon = ee.Geometry.Polygon([polygon_coords])
        
        # Load CDL dataset for the year
        # CDL is available from 2008-2023 (2024 may not be available yet)
        cdl_year = min(year, 2023)  # Use latest available
        cdl = ee.Image(f'USDA/NASS/CDL/{cdl_year}').select('cropland')
        
        # Get the most common crop type in the polygon
        stats = cdl.reduceRegion(
            reducer=ee.Reducer.mode(),  # Most common value
            geometry=ee_polygon,
            scale=30,
            maxPixels=1e9
        ).getInfo()
        
        crop_code = stats.get('cropland')
        
        if crop_code:
            crop_name = CDL_CROP_MAPPING.get(int(crop_code), f"Crop Code {crop_code}")
            return crop_name
        
        return None
    
    except Exception as e:
        print(f"Warning: CDL lookup failed: {e}")
        return None  # Don't fail the whole request if CDL unavailable


# ============================================================================
# EARTH ENGINE - NDVI FEATURE COMPUTATION
# ============================================================================

def compute_ndvi_features(polygon_coords: List[Tuple[float, float]], year: int) -> List[float]:
    """
    Compute 17 NDVI features from Sentinel-2 imagery for a given polygon.
    
    Features (in order):
    1. ndvi_mean, ndvi_std, ndvi_min, ndvi_max
    2. ndvi_p25, ndvi_p50, ndvi_p75
    3. ndvi_early, ndvi_late, elevation_m
    4. longitude, latitude
    5. ndvi_range, ndvi_iqr, ndvi_change
    6. ndvi_early_ratio, ndvi_late_ratio
    
    Args:
        polygon_coords: List of (lng, lat) tuples forming the field boundary
        year: Year for analysis (e.g., 2024)
    
    Returns:
        List of 17 float features
    """
    try:
        # Ensure Earth Engine is initialized
        if not ee.data._initialized:
            ee.Initialize()
        
        # Create Earth Engine polygon
        ee_polygon = ee.Geometry.Polygon([polygon_coords])
        
        # Date range for the growing season
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        early_end = f"{year}-06-30"
        late_start = f"{year}-07-01"
        
        # Load Sentinel-2 collection
        s2_collection = (
            ee.ImageCollection(SENTINEL2_COLLECTION)
            .filterBounds(ee_polygon)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        )
        
        # Compute NDVI
        def add_ndvi(image):
            ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
            return image.addBands(ndvi)
        
        ndvi_collection = s2_collection.map(add_ndvi).select("NDVI")
        
        # Overall statistics
        ndvi_composite = ndvi_collection.median()
        
        stats = ndvi_composite.reduceRegion(
            reducer=ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), "", True)
                .combine(ee.Reducer.min(), "", True)
                .combine(ee.Reducer.max(), "", True)
                .combine(ee.Reducer.percentile([25, 50, 75]), "", True),
            geometry=ee_polygon,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        # Early season vs late season
        early_ndvi = (
            ndvi_collection
            .filterDate(start_date, early_end)
            .median()
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_polygon,
                scale=10,
                maxPixels=1e9
            )
            .getInfo()
        )
        
        late_ndvi = (
            ndvi_collection
            .filterDate(late_start, end_date)
            .median()
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_polygon,
                scale=10,
                maxPixels=1e9
            )
            .getInfo()
        )
        
        # Elevation from SRTM
        elevation = (
            ee.Image("USGS/SRTMGL1_003")
            .select("elevation")
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_polygon,
                scale=30,
                maxPixels=1e9
            )
            .getInfo()
        )
        
        # Extract values with defaults
        ndvi_mean = stats.get("NDVI_mean", 0.5)
        ndvi_std = stats.get("NDVI_stdDev", 0.1)
        ndvi_min = stats.get("NDVI_min", 0.0)
        ndvi_max = stats.get("NDVI_max", 1.0)
        ndvi_p25 = stats.get("NDVI_p25", 0.4)
        ndvi_p50 = stats.get("NDVI_p50", 0.5)
        ndvi_p75 = stats.get("NDVI_p75", 0.6)
        
        ndvi_early = early_ndvi.get("NDVI", ndvi_mean)
        ndvi_late = late_ndvi.get("NDVI", ndvi_mean)
        elevation_m = elevation.get("elevation", 0.0)
        
        # Calculate centroid for location features
        centroid = ee_polygon.centroid().coordinates().getInfo()
        longitude = centroid[0]
        latitude = centroid[1]
        
        # Derived features
        ndvi_range = ndvi_max - ndvi_min
        ndvi_iqr = ndvi_p75 - ndvi_p25
        ndvi_change = ndvi_late - ndvi_early
        ndvi_early_ratio = ndvi_early / ndvi_mean if ndvi_mean > 0 else 1.0
        ndvi_late_ratio = ndvi_late / ndvi_mean if ndvi_mean > 0 else 1.0
        
        # Return 17 features in exact order
        features = [
            ndvi_mean,
            ndvi_std,
            ndvi_min,
            ndvi_max,
            ndvi_p25,
            ndvi_p50,
            ndvi_p75,
            ndvi_early,
            ndvi_late,
            elevation_m,
            longitude,
            latitude,
            ndvi_range,
            ndvi_iqr,
            ndvi_change,
            ndvi_early_ratio,
            ndvi_late_ratio,
        ]
        
        return features
    
    except Exception as e:
        print(f"Error computing NDVI features: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Earth Engine computation failed: {str(e)}"
        )


# ============================================================================
# VERTEX AI - CROP PREDICTION
# ============================================================================

def predict_crop_type(features: List[float]) -> Tuple[str, float]:
    """
    Call Vertex AI endpoint to predict crop type from features.
    
    Args:
        features: List of 17 NDVI features
    
    Returns:
        Tuple of (crop_name, confidence_score)
    """
    try:
        # Initialize Vertex AI client
        aiplatform.init(project=GCP_PROJECT_ID, location="us-central1")
        
        # Get endpoint
        endpoint = aiplatform.Endpoint(VERTEX_AI_ENDPOINT)
        
        # Make prediction
        prediction = endpoint.predict(instances=[features])
        
        # Parse response
        if prediction.predictions:
            pred = prediction.predictions[0]
            
            # Handle different response formats
            if isinstance(pred, str):
                crop = pred
                confidence = 0.95  # Default confidence
            elif isinstance(pred, list) and len(pred) >= 1:
                crop = pred[0]
                confidence = pred[1] if len(pred) > 1 else 0.95
            else:
                crop = str(pred)
                confidence = 0.95
            
            return crop, confidence
        else:
            raise HTTPException(status_code=500, detail="No predictions returned")
    
    except Exception as e:
        print(f"Error predicting crop: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Vertex AI prediction failed: {str(e)}"
        )


# ============================================================================
# CARBON CREDIT CALCULATION
# ============================================================================

def calculate_carbon_income(crop: str, area_acres: float) -> dict:
    """
    Calculate estimated CO₂ income based on crop type and acreage.
    
    Args:
        crop: Predicted crop type (e.g., "Corn", "Soybeans")
        area_acres: Field area in acres
    
    Returns:
        Dict with min, max, and average income
    """
    rates = CARBON_RATES.get(crop, DEFAULT_RATE)
    
    return {
        "min": rates["min"] * area_acres,
        "max": rates["max"] * area_acres,
        "avg": ((rates["min"] + rates["max"]) / 2) * area_acres,
    }


# ============================================================================
# GRID-BASED CLASSIFICATION
# ============================================================================

import math
import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

MAX_FIELD_SIZE_ACRES = 2000
MIN_FIELD_SIZE_FOR_GRID = 10  # Use grid for fields >= 10 acres (larger fields only)
MAX_GRID_CELLS = 25  # Reduced to 25 for faster processing
PRACTICAL_CELL_SIZES = [50, 100, 200, 300, 500]  # meters (larger cells for speed)


def calculate_optimal_grid_size(area_acres: float) -> Tuple[int, int]:
    """
    Calculate optimal grid cell size to target max cells.
    Conservative settings for faster processing.
    
    Args:
        area_acres: Field area in acres
    
    Returns:
        Tuple of (cell_size_meters, estimated_cell_count)
    """
    # Convert acres to square meters
    area_m2 = area_acres * 4046.86
    
    # Target cell area for MAX_GRID_CELLS
    target_cell_area = area_m2 / MAX_GRID_CELLS
    
    # Cell size (square cells)
    cell_size = math.sqrt(target_cell_area)
    
    # Round to nearest practical size
    cell_size = min(PRACTICAL_CELL_SIZES, key=lambda x: abs(x - cell_size))
    
    # Ensure within bounds (larger cells for speed)
    cell_size = max(50, min(500, cell_size))
    
    # Estimate actual cell count (with 20% buffer for irregular shapes)
    estimated_cells = int(area_m2 / (cell_size * cell_size) * 1.2)
    
    # If still too many cells, increase size
    while estimated_cells > MAX_GRID_CELLS and cell_size < 300:
        # Find next larger size
        idx = PRACTICAL_CELL_SIZES.index(cell_size)
        if idx < len(PRACTICAL_CELL_SIZES) - 1:
            cell_size = PRACTICAL_CELL_SIZES[idx + 1]
            estimated_cells = int(area_m2 / (cell_size * cell_size) * 1.2)
        else:
            break
    
    return cell_size, estimated_cells


def generate_grid_cells(coords: List[Tuple[float, float]], cell_size_meters: int) -> List[Polygon]:
    """
    Generate grid cells over the polygon.
    
    Args:
        coords: List of (lng, lat) tuples forming polygon
        cell_size_meters: Size of each grid cell in meters
    
    Returns:
        List of Shapely Polygon objects representing grid cells
    """
    # Create polygon and fix any invalid geometries (self-intersections, etc.)
    poly = Polygon(coords)
    if not poly.is_valid:
        print(f"⚠️  Invalid polygon detected (self-intersection), attempting to fix...")
        poly = poly.buffer(0)  # Fix self-intersections and invalid geometries
        if not poly.is_valid:
            raise ValueError("Polygon is invalid and cannot be fixed automatically. Please redraw the field boundary without crossing edges.")
    
    # Get bounding box
    minx, miny, maxx, maxy = poly.bounds
    
    # Convert cell size from meters to degrees (approximate)
    # At equator: 1 degree ≈ 111,320 meters
    # This is approximate; more accurate conversion would use projection
    lat_center = (miny + maxy) / 2
    cell_size_lng = cell_size_meters / (111320 * math.cos(math.radians(lat_center)))
    cell_size_lat = cell_size_meters / 111320
    
    # Generate grid
    cells = []
    lng = minx
    while lng < maxx:
        lat = miny
        while lat < maxy:
            # Create cell polygon
            cell = box(lng, lat, lng + cell_size_lng, lat + cell_size_lat)
            
            # Check if cell intersects field (>50% coverage)
            intersection = cell.intersection(poly)
            if intersection.area > 0:
                coverage = intersection.area / cell.area
                if coverage > 0.5:
                    cells.append(cell)
            
            lat += cell_size_lat
        lng += cell_size_lng
    
    return cells


def group_adjacent_cells(cells: List[Dict], threshold_distance: float = 0.001) -> List[Dict]:
    """
    Group adjacent cells with the same crop type.
    
    Args:
        cells: List of cell dicts with 'crop', 'confidence', 'polygon'
        threshold_distance: Distance threshold for adjacency (degrees)
    
    Returns:
        List of grouped crop zones
    """
    if not cells:
        return []
    
    # Group by crop type
    crops = {}
    for cell in cells:
        crop = cell['crop']
        if crop not in crops:
            crops[crop] = []
        crops[crop].append(cell)
    
    # For each crop, merge adjacent polygons
    zones = []
    for crop, crop_cells in crops.items():
        # Convert to Shapely polygons
        polygons = [Polygon(cell['polygon']) for cell in crop_cells]
        
        # Merge all polygons for this crop
        merged = unary_union(polygons)
        
        # Calculate total area and average confidence
        total_area_m2 = sum(Polygon(cell['polygon']).area * 111320 * 111320 for cell in crop_cells)
        total_area_acres = total_area_m2 * 0.000247105
        avg_confidence = sum(cell['confidence'] for cell in crop_cells) / len(crop_cells)
        
        # Extract coordinates from merged polygon
        if hasattr(merged, 'geoms'):  # MultiPolygon
            # If fragmented, take the largest polygon
            largest = max(merged.geoms, key=lambda p: p.area)
            coords = list(largest.exterior.coords)
        else:  # Polygon
            coords = list(merged.exterior.coords)
        
        zones.append({
            'crop': crop,
            'confidence': avg_confidence,
            'area_acres': total_area_acres,
            'polygon': coords
        })
    
    return zones


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "CarbonCheck Field API",
        "version": "1.0.0",
        "earth_engine": "initialized" if ee.data._initialized else "not initialized"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    ee_status = "ok" if ee.data._initialized else "not initialized"
    
    return {
        "status": "healthy",
        "earth_engine": ee_status,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/analyze", response_model=AnalyzeFieldResponse)
async def analyze_field(
    request: AnalyzeFieldRequest,
    user: dict = Depends(verify_firebase_token)
):
    """
    Analyze a farm field and return crop prediction + CO₂ income estimate.
    
    Uses grid-based classification for fields >= 2 acres to detect multiple crop types.
    Small fields (< 2 acres) use single prediction for speed.
    
    Requires:
    - Authorization header with Firebase ID token
    - Polygon coordinates (3-100 points)
    - Optional year (default: 2024)
    
    Returns:
    - Crop type and confidence (or multiple crop zones for large fields)
    - Area in acres
    - CO₂ income estimates (min, max, avg)
    """
    try:
        # Convert polygon to (lng, lat) tuples
        coords = [(point.lng, point.lat) for point in request.polygon]
        
        # Calculate area
        area_acres = calculate_polygon_area_acres(coords)
        
        # Validate field size
        if area_acres < 0.1:
            raise HTTPException(status_code=400, detail="Field is too small (minimum 0.1 acres)")
        
        if area_acres > MAX_FIELD_SIZE_ACRES:
            raise HTTPException(
                status_code=400, 
                detail=f"Field too large ({area_acres:.1f} acres). "
                       f"Maximum size is {MAX_FIELD_SIZE_ACRES} acres (about 3 square miles). "
                       f"Please draw a smaller area or split into multiple fields."
            )
        
        # Choose analysis method based on field size
        if area_acres < MIN_FIELD_SIZE_FOR_GRID:
            # Small field: use single prediction (fast)
            return await analyze_field_single(coords, area_acres, request.year)
        else:
            # Large field: use grid-based classification (detailed)
            return await analyze_field_grid(coords, area_acres, request.year)

    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_field: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def analyze_field_single(coords: List[Tuple[float, float]], area_acres: float, year: int) -> AnalyzeFieldResponse:
    """
    Single prediction for small fields (< 2 acres).
    Uses existing logic for backward compatibility.
    """
    print(f"Single prediction mode for {area_acres:.1f} acre field...")
    
    # Compute NDVI features via Earth Engine
    features = compute_ndvi_features(coords, year)
    
    # Predict crop type via Vertex AI
    crop, confidence = predict_crop_type(features)
    
    # Get CDL ground truth crop type
    cdl_crop = get_cdl_crop_type(coords, year)
    
    # Check if model and CDL agree
    cdl_agreement = False
    if cdl_crop:
        crop_normalized = crop.lower().replace(" ", "")
        cdl_normalized = cdl_crop.lower().replace(" ", "")
        cdl_agreement = crop_normalized == cdl_normalized
    
    # Calculate carbon income
    income = calculate_carbon_income(crop, area_acres)
    
    # Build response (legacy format)
    response = AnalyzeFieldResponse(
        crop=crop,
        confidence=confidence,
        cdl_crop=cdl_crop,
        cdl_agreement=cdl_agreement,
        area_acres=round(area_acres, 2),
        co2_income_min=round(income["min"], 2),
        co2_income_max=round(income["max"], 2),
        co2_income_avg=round(income["avg"], 2),
        features=features,
        timestamp=datetime.utcnow().isoformat()
    )
    
    print(f"✅ Single prediction: {crop} ({confidence:.0%}), ${income['avg']:.0f}/year")
    
    return response


async def analyze_field_grid(coords: List[Tuple[float, float]], area_acres: float, year: int) -> AnalyzeFieldResponse:
    """
    Grid-based classification for larger fields (>= 2 acres).
    Detects multiple crop types within the field.
    """
    print(f"Grid-based classification for {area_acres:.1f} acre field...")
    
    # Calculate optimal grid size
    cell_size_meters, estimated_cells = calculate_optimal_grid_size(area_acres)
    print(f"Using {cell_size_meters}m grid cells (estimated: {estimated_cells} cells)")
    
    # Generate grid
    cells = generate_grid_cells(coords, cell_size_meters)
    print(f"Generated {len(cells)} grid cells")
    
    if len(cells) == 0:
        raise HTTPException(status_code=500, detail="Failed to generate grid cells")
    
    # Process each cell (batch NDVI + batch predictions)
    cell_results = []
    
    # Batch process cells in groups of 10 for efficiency
    batch_size = 10
    for i in range(0, len(cells), batch_size):
        batch = cells[i:i+batch_size]
        
        for cell in batch:
            try:
                # Get cell coordinates
                cell_coords = list(cell.exterior.coords)
                
                # Compute NDVI features for this cell
                features = compute_ndvi_features(cell_coords, year)
                
                # Predict crop type for this cell
                crop, confidence = predict_crop_type(features)
                
                cell_results.append({
                    'crop': crop,
                    'confidence': confidence,
                    'polygon': cell_coords
                })
            except Exception as e:
                print(f"Warning: Cell processing failed: {e}")
                # Skip failed cells
                continue
    
    if not cell_results:
        raise HTTPException(status_code=500, detail="All grid cells failed processing")
    
    print(f"Processed {len(cell_results)}/{len(cells)} cells successfully")
    
    # Group adjacent cells by crop type
    zones = group_adjacent_cells(cell_results)
    print(f"Grouped into {len(zones)} crop zones")
    
    # Calculate total area and percentages
    total_area = sum(zone['area_acres'] for zone in zones)
    for zone in zones:
        zone['percentage'] = (zone['area_acres'] / total_area) * 100 if total_area > 0 else 0
    
    # Calculate CO₂ income by crop
    co2_by_crop = []
    total_co2_min = 0
    total_co2_max = 0
    total_co2_avg = 0
    
    for zone in zones:
        income = calculate_carbon_income(zone['crop'], zone['area_acres'])
        co2_by_crop.append(CO2IncomeByCrop(
            crop=zone['crop'],
            min=round(income['min'], 2),
            max=round(income['max'], 2),
            avg=round(income['avg'], 2)
        ))
        total_co2_min += income['min']
        total_co2_max += income['max']
        total_co2_avg += income['avg']
    
    # Build response (grid format)
    response = AnalyzeFieldResponse(
        area_acres=round(total_area, 2),
        co2_income_min=round(total_co2_min, 2),
        co2_income_max=round(total_co2_max, 2),
        co2_income_avg=round(total_co2_avg, 2),
        timestamp=datetime.utcnow().isoformat(),
        field_summary=FieldSummary(
            total_area_acres=round(total_area, 2),
            grid_cell_size_meters=cell_size_meters,
            total_cells=len(cell_results)
        ),
        crop_zones=[CropZone(
            crop=zone['crop'],
            confidence=round(zone['confidence'], 3),
            area_acres=round(zone['area_acres'], 2),
            percentage=round(zone['percentage'], 1),
            polygon=zone['polygon']
        ) for zone in zones],
        co2_income=CO2IncomeTotal(
            total_min=round(total_co2_min, 2),
            total_max=round(total_co2_max, 2),
            total_avg=round(total_co2_avg, 2),
            by_crop=co2_by_crop
        )
    )
    
    # Print summary
    print(f"✅ Grid analysis complete:")
    for zone in zones:
        print(f"   - {zone['crop']}: {zone['area_acres']:.1f} acres ({zone['percentage']:.0f}%)")
    print(f"   Total income: ${total_co2_avg:.0f}/year")
    
    return response


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

