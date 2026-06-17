"""
BigQuery lookup for precomputed Wisconsin crop grid cells.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from google.cloud import bigquery
from shapely.geometry import Polygon, shape

from wisconsin_grid import is_in_wisconsin, polygon_to_wkt

DEFAULT_PROJECT = os.getenv("GCP_PROJECT_ID", "ml-pipeline-477612")
DEFAULT_DATASET = os.getenv("PRECOMPUTE_BQ_DATASET", "crop_ml")
DEFAULT_TABLE = os.getenv("PRECOMPUTE_BQ_TABLE", "wi_crop_grid")


def full_table_id(project: str = DEFAULT_PROJECT) -> str:
    return f"{project}.{DEFAULT_DATASET}.{DEFAULT_TABLE}"


def precomputed_years() -> List[int]:
    raw = os.getenv("PRECOMPUTE_YEARS", "2024")
    return [int(y.strip()) for y in raw.split(",") if y.strip()]


def is_precompute_enabled() -> bool:
    return os.getenv("PRECOMPUTE_ENABLED", "true").lower() in ("1", "true", "yes")


def ensure_table(client: bigquery.Client, table_id: str) -> None:
    """Create wi_crop_grid table if missing."""
    schema = [
        bigquery.SchemaField("cell_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("year", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("crop", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("confidence", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("model_version", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("cell_size_m", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("centroid_lat", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("centroid_lng", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("geom", "GEOGRAPHY", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    ]
    table = bigquery.Table(table_id, schema=schema)
    try:
        client.get_table(table_id)
    except Exception:
        client.create_table(table)
        print(f"Created table {table_id}")


def lookup_intersecting_cells(
    coords: List[Tuple[float, float]],
    year: int,
    client: Optional[bigquery.Client] = None,
    min_coverage_ratio: float = 0.5,
) -> Optional[List[Dict]]:
    """
    Query precomputed cells intersecting the user polygon.

    Returns None if outside Wisconsin, year not precomputed, or coverage too low.
    """
    if not is_precompute_enabled():
        return None
    years = precomputed_years()
    if not years:
        return None
    # The app requests the current year, but precomputed data lags (crop
    # imagery is only complete after a growing season). Fall back to the most
    # recent precomputed year so lookups still hit instead of failing live.
    lookup_year = year if year in years else max(years)
    if not is_in_wisconsin(coords):
        return None

    client = client or bigquery.Client(project=DEFAULT_PROJECT)
    table_id = full_table_id()
    wkt = polygon_to_wkt(coords)

    query = f"""
    WITH user_poly AS (
      SELECT ST_GEOGFROMTEXT(@polygon_wkt) AS geom
    ),
    hits AS (
      SELECT
        g.cell_id,
        g.crop,
        g.confidence,
        g.centroid_lat,
        g.centroid_lng,
        ST_AsGeoJSON(g.geom) AS geom_json,
        ST_AREA(ST_INTERSECTION(g.geom, u.geom)) AS overlap_m2
      FROM `{table_id}` g
      CROSS JOIN user_poly u
      WHERE g.year = @year
        AND ST_INTERSECTS(g.geom, u.geom)
    )
    SELECT * FROM hits WHERE overlap_m2 > 0
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("year", "INT64", lookup_year),
            bigquery.ScalarQueryParameter("polygon_wkt", "STRING", wkt),
        ]
    )

    try:
        rows = list(client.query(query, job_config=job_config).result())
    except Exception as e:
        print(f"Precomputed lookup failed: {e}")
        return None

    if not rows:
        return None

    field_poly = Polygon(coords)
    if not field_poly.is_valid:
        field_poly = field_poly.buffer(0)
    field_area_m2 = _polygon_area_m2_approx(field_poly)

    total_overlap = sum(float(r.overlap_m2) for r in rows)
    if field_area_m2 <= 0 or total_overlap / field_area_m2 < min_coverage_ratio:
        return None

    results = []
    for r in rows:
        geom = json.loads(r.geom_json)
        poly = shape(geom)
        coords_list = list(poly.exterior.coords)
        results.append(
            {
                "cell_id": r.cell_id,
                "crop": r.crop,
                "confidence": float(r.confidence) if r.confidence is not None else None,
                "polygon": [(lng, lat) for lng, lat in coords_list],
                "overlap_m2": float(r.overlap_m2),
            }
        )
    return results


def _polygon_area_m2_approx(poly: Polygon) -> float:
    """Rough area in m² using degree-to-meter at polygon centroid."""
    import math

    c = poly.centroid
    lat = c.y
    m_per_deg_lat = 111_320.0
    m_per_deg_lng = 111_320.0 * math.cos(math.radians(lat))
    ring = list(poly.exterior.coords)

    def to_m(xy):
        return xy[0] * m_per_deg_lng, xy[1] * m_per_deg_lat

    m_ring = [to_m(p) for p in ring]
    area = 0.0
    for i in range(len(m_ring) - 1):
        x1, y1 = m_ring[i]
        x2, y2 = m_ring[i + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0
