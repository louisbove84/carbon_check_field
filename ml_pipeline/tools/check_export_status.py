#!/usr/bin/env python3
"""Check Earth Engine export task status and BigQuery data"""

import sys
from pathlib import Path
import yaml
import ee
from google.cloud import bigquery

# Load config
config_path = Path(__file__).parent / 'ml_pipeline' / 'orchestrator' / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

project_id = config['project']['id']
dataset = config['bigquery']['dataset']
table = config['bigquery']['tables']['training']

# Initialize Earth Engine
try:
    ee.Initialize(project=project_id)
except Exception:
    ee.Initialize()

print("=" * 70)
print("🔍 CHECKING EXPORT STATUS")
print("=" * 70)
print()

# Check recent tasks
print("📋 Recent Earth Engine Export Tasks:")
print("-" * 70)
tasks = ee.batch.Task.list()
recent_tasks = [t for t in tasks if 'bigquery' in t.config.get('description', '').lower()][:5]

if not recent_tasks:
    print("   No recent BigQuery export tasks found")
else:
    for task in recent_tasks:
        state = task.state
        task_id = task.id
        desc = task.config.get('description', 'N/A')
        
        status_icon = {
            'COMPLETED': '✅',
            'FAILED': '❌',
            'RUNNING': '🔄',
            'READY': '⏳',
            'CANCELLED': '⚠️'
        }.get(state, '❓')
        
        print(f"   {status_icon} Task: {task_id}")
        print(f"      State: {state}")
        print(f"      Description: {desc}")
        
        if state == 'FAILED':
            try:
                error_msg = getattr(task, 'error_message', None) or getattr(task, 'error', None)
                if error_msg:
                    print(f"      Error: {error_msg}")
            except:
                pass
        
        print()

# Check BigQuery data
print("=" * 70)
print("📊 CHECKING BIGQUERY DATA")
print("=" * 70)
print()

client = bigquery.Client(project=project_id)
table_ref = f'{project_id}.{dataset}.{table}'

try:
    # Get total row count
    count_query = f"SELECT COUNT(*) as total FROM `{table_ref}`"
    result = client.query(count_query).to_dataframe()
    total_rows = result['total'].iloc[0]
    
    print(f"Table: {table_ref}")
    print(f"Total rows: {total_rows}")
    print()
    
    if total_rows > 0:
        # Get crop distribution
        query = f"""
            SELECT crop, COUNT(*) as count
            FROM `{table_ref}`
            WHERE crop IS NOT NULL
            GROUP BY crop
            ORDER BY count DESC
        """
        
        results = client.query(query).to_dataframe()
        print("Crop distribution:")
        print(results.to_string(index=False))
        print()
        
        if 'Other' in results['crop'].values:
            other_count = results[results['crop'] == 'Other']['count'].values[0]
            print(f"✅ 'Other' category found: {other_count} samples")
        else:
            print("❌ 'Other' category NOT found")
    else:
        print("❌ Table is empty - export may have failed")
        print()
        print("Next steps:")
        print("1. Check Earth Engine Tasks console: https://code.earthengine.google.com/tasks")
        print("2. Look for task ID in recent tasks above")
        print("3. Re-run data collection if needed")
        
except Exception as e:
    print(f"❌ Error checking BigQuery: {e}")
    print("   Table may not exist yet")

print()
print("=" * 70)
