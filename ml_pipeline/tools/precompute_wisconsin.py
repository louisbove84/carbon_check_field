#!/usr/bin/env python3
"""
Precompute Wisconsin crop predictions on a fixed grid.

Usage:
  # Pilot: Dane County envelope (~200 cells, fast test)
  python precompute_wisconsin.py --region dane --year 2024 --max-cells 200

  # Full Wisconsin (long-running; use tmux)
  python precompute_wisconsin.py --region wisconsin --year 2024 --upload-bq

Requires: earthengine authenticate, gcloud ADC, model in gs://carboncheck-data/models/crop_classifier_latest
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import ee
import joblib
import numpy as np
from google.cloud import bigquery, storage

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.join(_SCRIPT_DIR, "..", "shared")
sys.path.insert(0, _SHARED_DIR)

from batch_ndvi_extract import extract_ndvi_features_batch  # noqa: E402
from precomputed_lookup import ensure_table, full_table_id  # noqa: E402
from wisconsin_grid import (  # noqa: E402
    cell_to_wkt,
    generate_grid_cells,
    iter_cell_batches,
    region_polygon_wgs84,
)

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "ml-pipeline-477612")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "carboncheck-data")
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "models/crop_classifier_latest")


def load_rf_model() -> tuple:
    """Download and load Random Forest model from GCS."""
    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET)
    local_dir = "/tmp/precompute_models"
    os.makedirs(local_dir, exist_ok=True)

    blobs = bucket.list_blobs(prefix=MODEL_PREFIX)
    for blob in blobs:
        rel = blob.name[len(MODEL_PREFIX) :].lstrip("/")
        if not rel:
            continue
        path = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        blob.download_to_filename(path)

    rf_path = os.path.join(local_dir, "model_rf.pkl")
    legacy = os.path.join(local_dir, "model.joblib")
    path = rf_path if os.path.exists(rf_path) else legacy
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at gs://{MODEL_BUCKET}/{MODEL_PREFIX}")

    data = joblib.load(path)
    if isinstance(data, dict):
        pipeline = data["pipeline"]
        classes = data.get("classes", getattr(pipeline, "classes_", None))
    else:
        pipeline = data
        classes = getattr(pipeline, "classes_", None)
    version = datetime.now(timezone.utc).strftime("%Y%m%d")
    return pipeline, classes, version


def predict_batch(pipeline, classes, rows: List[Dict]) -> List[Dict]:
    X = np.array([r["features"] for r in rows])
    probs = pipeline.predict_proba(X)
    out = []
    for i, row in enumerate(rows):
        idx = int(np.argmax(probs[i]))
        crop = str(classes[idx])
        confidence = float(probs[i][idx])
        out.append({**row, "crop": crop, "confidence": confidence})
    return out


def region_ee_geometry(region: str) -> ee.Geometry:
    poly = region_polygon_wgs84(region)
    coords = list(poly.exterior.coords)
    return ee.Geometry.Polygon([coords])


def upload_rows(
    client: bigquery.Client,
    table_id: str,
    rows: List[Dict],
    year: int,
    cell_size_m: int,
    model_version: str,
) -> None:
    """Insert rows using parameterized queries (GEOGRAPHY via ST_GEOGFROMTEXT)."""
    sql = f"""
    INSERT INTO `{table_id}`
      (cell_id, year, crop, confidence, model_version, cell_size_m,
       centroid_lat, centroid_lng, geom, created_at)
    VALUES
      (@cell_id, @year, @crop, @confidence, @model_version, @cell_size_m,
       @centroid_lat, @centroid_lng, ST_GEOGFROMTEXT(@wkt), CURRENT_TIMESTAMP())
    """
    for r in rows:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("cell_id", "STRING", r["cell_id"]),
                bigquery.ScalarQueryParameter("year", "INT64", year),
                bigquery.ScalarQueryParameter("crop", "STRING", r["crop"]),
                bigquery.ScalarQueryParameter("confidence", "FLOAT64", r["confidence"]),
                bigquery.ScalarQueryParameter("model_version", "STRING", model_version),
                bigquery.ScalarQueryParameter("cell_size_m", "INT64", cell_size_m),
                bigquery.ScalarQueryParameter("centroid_lat", "FLOAT64", r["centroid_lat"]),
                bigquery.ScalarQueryParameter("centroid_lng", "FLOAT64", r["centroid_lng"]),
                bigquery.ScalarQueryParameter("wkt", "STRING", cell_to_wkt(r)),
            ]
        )
        client.query(sql, job_config=job_config).result()


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Wisconsin crop grid")
    parser.add_argument(
        "--region",
        choices=["foxvalley", "madison", "dane", "wisconsin"],
        default="dane",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--cell-size", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--upload-bq", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only generate grid, no EE")
    args = parser.parse_args()

    if args.region == "dane" and args.max_cells is None:
        args.max_cells = 200

    print(f"Generating {args.cell_size}m grid for region={args.region}...")
    cells = generate_grid_cells(args.region, args.cell_size, max_cells=args.max_cells)
    print(f"Grid: {len(cells)} cells")

    if args.dry_run:
        print("Dry run complete.")
        return

    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Initialize()
    region_geom = region_ee_geometry(args.region)
    pipeline, classes, model_version = load_rf_model()
    print(f"Loaded model, version tag={model_version}")

    bq_client = None
    table_id = full_table_id(PROJECT_ID)
    if args.upload_bq:
        bq_client = bigquery.Client(project=PROJECT_ID)
        ensure_table(bq_client, table_id)
        print(f"Upload target: {table_id}")

    processed = 0
    for batch_idx, batch in enumerate(iter_cell_batches(cells, args.batch_size)):
        print(f"Batch {batch_idx + 1}: {len(batch)} cells (EE extract)...")
        extracted = extract_ndvi_features_batch(batch, region_geom, args.year)
        predicted = predict_batch(pipeline, classes, extracted)

        for r in predicted:
            print(f"  {r['cell_id']}: {r['crop']} ({r['confidence']:.1%})")

        if args.upload_bq and bq_client:
            upload_rows(
                bq_client,
                table_id,
                predicted,
                args.year,
                args.cell_size,
                model_version,
            )
            print(f"  Uploaded {len(predicted)} rows to BigQuery")

        processed += len(predicted)
        print(f"Progress: {processed}/{len(cells)}")

    print(f"Done. Processed {processed} cells.")


if __name__ == "__main__":
    main()
