#!/usr/bin/env python3
"""
Clean up old TensorBoard Experiments to reduce storage costs.

Vertex AI TensorBoard charges $10/GB/month for storage.
This script helps delete outdated experiments.

Usage:
    # List all experiments
    python cleanup_tensorboard_experiments.py --list

    # Delete experiments older than a date
    python cleanup_tensorboard_experiments.py --delete-before 2024-12-01

    # Delete all experiments (WARNING: destructive!)
    python cleanup_tensorboard_experiments.py --delete-all --confirm
"""

import argparse
from datetime import datetime
from google.cloud import aiplatform

# Configuration
PROJECT_ID = "ml-pipeline-477612"
LOCATION = "us-central1"
TENSORBOARD_INSTANCE_ID = "1461173987500359680"


def list_experiments():
    """List all TensorBoard experiments with their timestamps."""
    print("=" * 70)
    print("🔍 LISTING TENSORBOARD EXPERIMENTS")
    print("=" * 70)
    print(f"Project: {PROJECT_ID}")
    print(f"TensorBoard ID: {TENSORBOARD_INSTANCE_ID}")
    print()
    
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    tensorboard = aiplatform.Tensorboard(
        tensorboard_name=TENSORBOARD_INSTANCE_ID
    )
    
    experiments = aiplatform.TensorboardExperiment.list(
        tensorboard_name=tensorboard.resource_name,
        order_by="create_time desc"
    )
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"Found {len(experiments)} experiments:\n")
    
    total_size = 0
    for i, exp in enumerate(experiments, 1):
        create_time = exp.create_time
        update_time = exp.update_time
        exp_name = exp.display_name or exp.name.split('/')[-1]
        
        print(f"{i}. {exp_name}")
        print(f"   Created: {create_time}")
        print(f"   Updated: {update_time}")
        print(f"   Resource: {exp.resource_name}")
        print()
    
    print("=" * 70)
    print(f"Total: {len(experiments)} experiments")
    print("💰 Storage cost: ~$10/GB/month")
    print()


def delete_experiments_before(cutoff_date, dry_run=True):
    """Delete experiments created before the cutoff date."""
    print("=" * 70)
    if dry_run:
        print("🔍 DRY RUN - No experiments will be deleted")
    else:
        print("🗑️  DELETING EXPERIMENTS")
    print("=" * 70)
    print(f"Cutoff date: {cutoff_date}")
    print()
    
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    tensorboard = aiplatform.Tensorboard(
        tensorboard_name=TENSORBOARD_INSTANCE_ID
    )
    
    experiments = aiplatform.TensorboardExperiment.list(
        tensorboard_name=tensorboard.resource_name,
        order_by="create_time"
    )
    
    cutoff = datetime.fromisoformat(cutoff_date)
    deleted_count = 0
    kept_count = 0
    
    for exp in experiments:
        create_time = exp.create_time
        exp_name = exp.display_name or exp.name.split('/')[-1]
        
        # Convert to offset-naive for comparison
        create_time_naive = create_time.replace(tzinfo=None)
        
        if create_time_naive < cutoff:
            print(f"{'[DRY RUN] ' if dry_run else ''}Deleting: {exp_name}")
            print(f"  Created: {create_time}")
            if not dry_run:
                try:
                    exp.delete()
                    print(f"  ✅ Deleted")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
            deleted_count += 1
        else:
            kept_count += 1
            print(f"Keeping: {exp_name} (created {create_time})")
    
    print()
    print("=" * 70)
    print(f"{'Would delete' if dry_run else 'Deleted'}: {deleted_count} experiments")
    print(f"Keeping: {kept_count} experiments")
    print()


def delete_all_experiments(confirm=False):
    """Delete ALL experiments (WARNING: destructive!)."""
    if not confirm:
        print("❌ ERROR: Must use --confirm flag to delete all experiments")
        return
    
    print("=" * 70)
    print("🗑️  DELETING ALL EXPERIMENTS")
    print("=" * 70)
    print("⚠️  WARNING: This will delete ALL experiments!")
    print()
    
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    tensorboard = aiplatform.Tensorboard(
        tensorboard_name=TENSORBOARD_INSTANCE_ID
    )
    
    experiments = aiplatform.TensorboardExperiment.list(
        tensorboard_name=tensorboard.resource_name
    )
    
    total = len(experiments)
    deleted = 0
    
    for i, exp in enumerate(experiments, 1):
        exp_name = exp.display_name or exp.name.split('/')[-1]
        print(f"[{i}/{total}] Deleting: {exp_name}")
        try:
            exp.delete()
            print(f"  ✅ Deleted")
            deleted += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print()
    print("=" * 70)
    print(f"✅ Deleted {deleted}/{total} experiments")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old TensorBoard experiments to reduce storage costs"
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all experiments'
    )
    parser.add_argument(
        '--delete-before',
        type=str,
        metavar='DATE',
        help='Delete experiments created before DATE (format: YYYY-MM-DD)'
    )
    parser.add_argument(
        '--delete-all',
        action='store_true',
        help='Delete ALL experiments (requires --confirm)'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm destructive operations'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Show what would be deleted without actually deleting (default: True)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually execute deletions (turns off dry-run)'
    )
    
    args = parser.parse_args()
    
    # Handle dry-run logic
    dry_run = args.dry_run and not args.execute
    
    if args.list:
        list_experiments()
    elif args.delete_before:
        delete_experiments_before(args.delete_before, dry_run=dry_run)
    elif args.delete_all:
        if dry_run:
            print("❌ ERROR: --delete-all cannot be used with --dry-run")
            print("   Use: --delete-all --confirm --execute")
        else:
            delete_all_experiments(confirm=args.confirm)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

