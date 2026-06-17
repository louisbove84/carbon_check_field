"""
Batch NDVI feature extraction via Earth Engine reduceRegions.

Builds statewide season composites once, then samples many grid cells per batch.
"""

from __future__ import annotations

from typing import Dict, List

import ee

from feature_engineering import engineer_features_from_raw


def build_season_ndvi_images(region: ee.Geometry, year: int) -> Dict[str, ee.Image]:
    """Build full-season, early, and late NDVI median composites."""
    start_date = f"{year}-04-15"
    end_date = f"{year}-09-01"
    early_end = f"{year}-06-01"
    late_start = f"{year}-07-01"

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    def add_ndvi(image):
        return image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    ndvi_col = s2.map(add_ndvi).select("NDVI")

    def _safe_median(collection: ee.ImageCollection, default: float = 0.5) -> ee.Image:
        """Avoid 'Image has no bands' when a date window has no clear scenes."""
        size = collection.size()
        return ee.Image(
            ee.Algorithms.If(
                size.gt(0),
                collection.median(),
                ee.Image.constant(default).rename("NDVI"),
            )
        )

    full = _safe_median(ndvi_col).clip(region)
    early = _safe_median(ndvi_col.filterDate(start_date, early_end)).clip(region)
    late = _safe_median(ndvi_col.filterDate(late_start, end_date)).clip(region)
    return {"full": full, "early": early, "late": late}


def cells_to_feature_collection(cells: List[Dict]) -> ee.FeatureCollection:
    features = []
    for cell in cells:
        geom = ee.Geometry.Polygon([cell["polygon_coords"]])
        features.append(
            ee.Feature(
                geom,
                {
                    "cell_id": cell["cell_id"],
                    "centroid_lng": cell["centroid_lng"],
                    "centroid_lat": cell["centroid_lat"],
                },
            )
        )
    return ee.FeatureCollection(features)


def _default(val, fallback: float) -> float:
    if val is None:
        return fallback
    try:
        return float(val)
    except (TypeError, ValueError):
        return fallback


def parse_feature_properties(props: Dict) -> List[float]:
    """Convert reduceRegions properties to engineered feature vector.

    For a single-band image, reduceRegions names outputs by the reducer
    (mean, stdDev, min, max, p25, p50, p75) with no band-name prefix.
    """
    raw = {
        "ndvi_mean": _default(props.get("mean"), 0.5),
        "ndvi_std": _default(props.get("stdDev"), 0.1),
        "ndvi_min": _default(props.get("min"), 0.0),
        "ndvi_max": _default(props.get("max"), 1.0),
        "ndvi_p25": _default(props.get("p25"), 0.4),
        "ndvi_p50": _default(props.get("p50"), 0.5),
        "ndvi_p75": _default(props.get("p75"), 0.6),
        "ndvi_early": _default(props.get("early_NDVI"), 0.5),
        "ndvi_late": _default(props.get("late_NDVI"), 0.5),
    }
    return engineer_features_from_raw(**raw)


def extract_ndvi_features_batch(
    cells: List[Dict],
    region: ee.Geometry,
    year: int,
    scale: int = 30,
) -> List[Dict]:
    """
    Extract NDVI features for a batch of grid cells.

    Returns list of dicts with cell metadata + features list.
    """
    if not cells:
        return []

    images = build_season_ndvi_images(region, year)
    fc = cells_to_feature_collection(cells)

    full_reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.min(), "", True)
        .combine(ee.Reducer.max(), "", True)
        .combine(ee.Reducer.percentile([25, 50, 75]), "", True)
    )

    full_fc = images["full"].reduceRegions(
        collection=fc,
        reducer=full_reducer,
        scale=scale,
        tileScale=4,
    )
    early_fc = images["early"].reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=4,
    )
    late_fc = images["late"].reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=4,
    )

    full_list = full_fc.getInfo()["features"]
    early_by_id = {
        f["properties"]["cell_id"]: f["properties"].get("mean")
        for f in early_fc.getInfo()["features"]
    }
    late_by_id = {
        f["properties"]["cell_id"]: f["properties"].get("mean")
        for f in late_fc.getInfo()["features"]
    }

    results = []
    cell_by_id = {c["cell_id"]: c for c in cells}
    for feat in full_list:
        props = feat["properties"]
        cell_id = props["cell_id"]
        props["early_NDVI"] = early_by_id.get(cell_id)
        props["late_NDVI"] = late_by_id.get(cell_id)
        features = parse_feature_properties(props)
        cell = cell_by_id[cell_id]
        results.append(
            {
                **cell,
                "features": features,
            }
        )
    return results
