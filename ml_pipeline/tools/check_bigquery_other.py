#!/usr/bin/env python3
"""Quick script to check if 'Other' samples exist in BigQuery"""

import sys
from pathlib import Path
import yaml
from google.cloud import bigquery

# Load config
config_path = Path(__file__).parent / 'ml_pipeline' / 'orchestrator' / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

project_id = config['project']['id']
dataset = config['bigquery']['dataset']
table = config['bigquery']['tables']['training']

client = bigquery.Client(project=project_id)

# Check crop distribution
query = f"""
    SELECT crop, COUNT(*) as count
    FROM `{project_id}.{dataset}.{table}`
    WHERE crop IS NOT NULL
    GROUP BY crop
    ORDER BY count DESC
"""

print("Checking BigQuery for crop distribution...")
print(f"Table: {project_id}.{dataset}.{table}\n")

results = client.query(query).to_dataframe()

if len(results) == 0:
    print("❌ No data found in BigQuery table!")
    print("   You need to run the Earth Engine collector first.")
else:
    print("✅ Found crops in BigQuery:")
    print(results.to_string(index=False))
    print()
    
    if 'Other' in results['crop'].values:
        other_count = results[results['crop'] == 'Other']['count'].values[0]
        print(f"✅ 'Other' category found: {other_count} samples")
    else:
        print("❌ 'Other' category NOT found in BigQuery")
        print("\nTo add 'Other' samples:")
        print("1. Run the Earth Engine collector (orchestrator)")
        print("2. Make sure collect_other: true in config.yaml")
        print("3. Wait for BigQuery export to complete")
