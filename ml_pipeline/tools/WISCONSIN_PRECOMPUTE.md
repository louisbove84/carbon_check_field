# Wisconsin Precomputed Crop Grid

Instant crop lookups for Wisconsin fields via BigQuery — no per-request Earth Engine calls.

## Architecture

1. **Offline batch** (`precompute_wisconsin.py`): grid → Earth Engine NDVI → RF model → BigQuery
2. **Online API** (`/analyze`): polygon intersects `crop_ml.wi_crop_grid` → instant response

## One-time setup

```bash
# Authenticate
earthengine authenticate
gcloud auth application-default login
gcloud config set project ml-pipeline-477612

# Create BigQuery table
bq query --use_legacy_sql=false < setup_wisconsin_grid_table.sql

# Install deps (from repo root)
pip install -r ml_pipeline/requirements.txt
```

## Pilot run (Dane County, ~200 cells, ~30 min)

```bash
cd ml_pipeline/tools
python precompute_wisconsin.py \
  --region dane \
  --year 2024 \
  --max-cells 200 \
  --upload-bq
```

## Full Wisconsin (long-running — use tmux)

```bash
tmux new -s wi-precompute
python precompute_wisconsin.py \
  --region wisconsin \
  --year 2024 \
  --batch-size 100 \
  --upload-bq
```

Expect ~2M cells at 250m resolution. Run in stages or increase `--batch-size` if EE quotas allow.

## Backend env vars (Cloud Run)

| Variable | Default | Description |
|----------|---------|-------------|
| `PRECOMPUTE_ENABLED` | `true` | Use BigQuery lookup for WI fields |
| `PRECOMPUTE_YEARS` | `2024` | Comma-separated years with data |
| `PRECOMPUTE_BQ_DATASET` | `crop_ml` | BigQuery dataset |
| `PRECOMPUTE_BQ_TABLE` | `wi_crop_grid` | Table name |
| `PRECOMPUTE_CELL_SIZE_M` | `250` | Grid resolution (for response metadata) |

## Verify lookup

```bash
bq query --use_legacy_sql=false \
  'SELECT crop, COUNT(*) n FROM `ml-pipeline-477612.crop_ml.wi_crop_grid` WHERE year=2024 GROUP BY crop'
```

Draw a field in Dane County in the app — `/analyze` should log `mode=precomputed`.

## Refresh annually

Re-run `precompute_wisconsin.py` with `--year 2025` after growing season imagery is available, then add `2025` to `PRECOMPUTE_YEARS`.
