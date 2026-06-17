"""
Wisconsin grid generation for precomputed crop predictions.

Uses Wisconsin Transverse Mercator (EPSG:3071) for accurate 250m cells,
then converts cell corners to WGS84 for Earth Engine and BigQuery.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

# Wisconsin approximate bounds (WGS84) for fast centroid checks
WI_BOUNDS = {
    "min_lng": -92.92,
    "max_lng": -86.75,
    "min_lat": 42.45,
    "max_lat": 47.15,
}

WTM = "EPSG:3071"
WGS84 = "EPSG:4326"

_TO_WTM = Transformer.from_crs(WGS84, WTM, always_xy=True)
_TO_WGS84 = Transformer.from_crs(WTM, WGS84, always_xy=True)

# Dane County pilot (approximate WGS84 envelope)
DANE_COUNTY_BOUNDS = {
    "min_lng": -89.85,
    "max_lng": -89.00,
    "min_lat": 42.93,
    "max_lat": 43.30,
}

# Madison demo block (~11x11 km dense block over the city for a usable test area)
MADISON_BOUNDS = {
    "min_lng": -89.468,
    "max_lng": -89.332,
    "min_lat": 43.025,
    "max_lat": 43.123,
}

# Fox Valley farmland demo block (~11x11 km near Wrightstown, WI — real cropland)
FOXVALLEY_BOUNDS = {
    "min_lng": -88.500,
    "max_lng": -88.361,
    "min_lat": 44.360,
    "max_lat": 44.459,
}


def _to_wtm(poly: Polygon) -> Polygon:
    return transform(_TO_WTM.transform, poly)


def _to_wgs84(poly: Polygon) -> Polygon:
    return transform(_TO_WGS84.transform, poly)


def polygon_centroid(coords: List[Tuple[float, float]]) -> Tuple[float, float]:
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    c = poly.centroid
    return c.x, c.y


def is_in_wisconsin(coords: List[Tuple[float, float]]) -> bool:
    lng, lat = polygon_centroid(coords)
    return (
        WI_BOUNDS["min_lng"] <= lng <= WI_BOUNDS["max_lng"]
        and WI_BOUNDS["min_lat"] <= lat <= WI_BOUNDS["max_lat"]
    )


def bounds_for_region(region: str) -> Dict[str, float]:
    region = region.lower()
    if region in ("foxvalley", "fox_valley", "wrightstown"):
        return FOXVALLEY_BOUNDS.copy()
    if region in ("madison", "madison_demo"):
        return MADISON_BOUNDS.copy()
    if region in ("dane", "pilot", "dane_county"):
        return DANE_COUNTY_BOUNDS.copy()
    if region in ("wisconsin", "wi", "state"):
        return WI_BOUNDS.copy()
    raise ValueError(
        f"Unknown region: {region}. Use 'wisconsin', 'dane', 'madison', or 'foxvalley'."
    )


def region_polygon_wgs84(region: str) -> Polygon:
    b = bounds_for_region(region)
    return Polygon(
        [
            (b["min_lng"], b["min_lat"]),
            (b["max_lng"], b["min_lat"]),
            (b["max_lng"], b["max_lat"]),
            (b["min_lng"], b["max_lat"]),
            (b["min_lng"], b["min_lat"]),
        ]
    )


def generate_grid_cells(
    region: str,
    cell_size_m: int = 250,
    max_cells: Optional[int] = None,
) -> List[Dict]:
    """
    Generate square grid cells covering a region.

    Returns list of dicts:
      cell_id, row, col, polygon_coords [(lng,lat),...], centroid (lng,lat), area_m2
    """
    region_poly_wgs84 = region_polygon_wgs84(region)
    region_poly_wtm = _to_wtm(region_poly_wgs84)
    minx, miny, maxx, maxy = region_poly_wtm.bounds

    cells: List[Dict] = []
    row = 0
    y = miny
    while y < maxy:
        col = 0
        x = minx
        while x < maxx:
            cell_wtm = Polygon(
                [
                    (x, y),
                    (x + cell_size_m, y),
                    (x + cell_size_m, y + cell_size_m),
                    (x, y + cell_size_m),
                    (x, y),
                ]
            )
            if cell_wtm.intersects(region_poly_wtm):
                intersection = cell_wtm.intersection(region_poly_wtm)
                if intersection.is_empty:
                    col += 1
                    x += cell_size_m
                    continue

                cell_wgs84 = _to_wgs84(cell_wtm)
                coords = list(cell_wgs84.exterior.coords)
                centroid = cell_wgs84.centroid
                cells.append(
                    {
                        "cell_id": f"{region}_{row}_{col}",
                        "row": row,
                        "col": col,
                        "polygon_coords": [(lng, lat) for lng, lat in coords],
                        "centroid_lng": centroid.x,
                        "centroid_lat": centroid.y,
                        "area_m2": float(cell_wtm.area),
                    }
                )
                if max_cells and len(cells) >= max_cells:
                    return cells
            col += 1
            x += cell_size_m
        row += 1
        y += cell_size_m

    return cells


def iter_cell_batches(cells: List[Dict], batch_size: int) -> Iterator[List[Dict]]:
    for i in range(0, len(cells), batch_size):
        yield cells[i : i + batch_size]


def polygon_to_wkt(coords: List[Tuple[float, float]]) -> str:
    """WKT POLYGON for BigQuery ST_GEOGFROMTEXT (lon lat order)."""
    if coords[0] != coords[-1]:
        coords = list(coords) + [coords[0]]
    parts = ", ".join(f"{lng} {lat}" for lng, lat in coords)
    return f"POLYGON(({parts}))"


def cell_to_wkt(cell: Dict) -> str:
    return polygon_to_wkt(cell["polygon_coords"])
