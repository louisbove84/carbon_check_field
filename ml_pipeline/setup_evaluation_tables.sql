-- ============================================================
-- BigQuery Tables for Model Evaluation & Tracking
-- ============================================================
-- Run this script to create tables for the ML evaluation pipeline

-- Table 1: Holdout Test Set (permanent 20% of data for unbiased evaluation)
CREATE TABLE IF NOT EXISTS `ml-pipeline-477612.crop_ml.holdout_test_set` (
  field_id STRING,
  crop STRING,
  crop_code INT64,
  sample_id STRING,
  collection_date TIMESTAMP,
  cdl_code INT64,
  -- NDVI features
  ndvi_mean FLOAT64,
  ndvi_std FLOAT64,
  ndvi_min FLOAT64,
  ndvi_max FLOAT64,
  ndvi_p25 FLOAT64,
  ndvi_p50 FLOAT64,
  ndvi_p75 FLOAT64,
  ndvi_early FLOAT64,
  ndvi_late FLOAT64,
  elevation_m FLOAT64,
  longitude FLOAT64,
  latitude FLOAT64,
  -- Holdout metadata
  reserved_date TIMESTAMP
)
PARTITION BY DATE(reserved_date)
OPTIONS(
  description="Permanent holdout test set for unbiased model evaluation (20% of all data)"
);

-- Table 2: Model Performance Metrics (tracks all model evaluations)
CREATE TABLE IF NOT EXISTS `ml-pipeline-477612.crop_ml.model_performance` (
  model_type STRING,  -- "champion" or "challenger"
  model_name STRING,
  accuracy FLOAT64,
  test_samples INT64,
  evaluation_time TIMESTAMP,
  -- Per-crop F1 scores
  corn_f1 FLOAT64,
  soybeans_f1 FLOAT64,
  alfalfa_f1 FLOAT64,
  winter_wheat_f1 FLOAT64,
  -- Full metrics as JSON
  metrics_json STRING
)
PARTITION BY DATE(evaluation_time)
OPTIONS(
  description="Model performance metrics for champion/challenger comparison"
);

-- Table 3: Deployment History (tracks which models were deployed and when)
CREATE TABLE IF NOT EXISTS `ml-pipeline-477612.crop_ml.deployment_history` (
  deployment_time TIMESTAMP,
  model_gcs_path STRING,
  model_id STRING,
  endpoint_id STRING,
  deployment_decision STRING,  -- "deployed", "blocked", or "rollback"
  -- Metrics snapshot
  accuracy FLOAT64,
  training_samples INT64,
  -- Decision details
  gates_passed STRING,  -- JSON array
  gates_failed STRING,  -- JSON array
  reasoning STRING,  -- JSON array
  -- Full decision as JSON
  decision_json STRING
)
PARTITION BY DATE(deployment_time)
OPTIONS(
  description="History of model deployments and deployment decisions"
);

-- Note: BigQuery automatically indexes partitioned columns and clustering keys
-- No explicit CREATE INDEX needed

-- Sample query to compare recent champion vs challenger
-- ============================================================
-- SELECT 
--   model_type,
--   model_name,
--   accuracy,
--   corn_f1,
--   soybeans_f1,
--   alfalfa_f1,
--   winter_wheat_f1,
--   evaluation_time
-- FROM `ml-pipeline-477612.crop_ml.model_performance`
-- WHERE evaluation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
-- ORDER BY evaluation_time DESC;

