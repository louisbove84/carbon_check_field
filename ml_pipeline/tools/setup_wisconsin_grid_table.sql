-- Wisconsin precomputed crop grid table
-- Run: bq query --use_legacy_sql=false < setup_wisconsin_grid_table.sql

CREATE SCHEMA IF NOT EXISTS `ml-pipeline-477612.crop_ml`;

CREATE TABLE IF NOT EXISTS `ml-pipeline-477612.crop_ml.wi_crop_grid` (
  cell_id STRING NOT NULL,
  year INT64 NOT NULL,
  crop STRING NOT NULL,
  confidence FLOAT64,
  model_version STRING,
  cell_size_m INT64,
  centroid_lat FLOAT64,
  centroid_lng FLOAT64,
  geom GEOGRAPHY NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY RANGE_BUCKET(year, GENERATE_ARRAY(2015, 2031, 1))
CLUSTER BY crop, cell_id;
