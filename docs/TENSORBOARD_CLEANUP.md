# TensorBoard Cleanup Guide

## Understanding TensorBoard Costs

Vertex AI TensorBoard charges **$10/GB/month** for storage. Each training experiment stores TensorBoard logs (metrics, visualizations, etc.), which accumulate over time.

## What's "Running" vs What to Delete

### TensorBoard INSTANCE (keep this!)
- Status: "Running" ← **NORMAL and EXPECTED**
- This is the TensorBoard service itself
- **DO NOT DELETE** - you need this for future training runs
- ID: `8764605208211750912`

### TensorBoard EXPERIMENTS (clean these up periodically)
- Individual training runs
- Each experiment stores data and costs money
- Safe to delete old ones you no longer need

## Current Status

As of Dec 6, 2024, we have:
- **14 experiments** total
- Most from Dec 5-6 (recent testing)
- 3 older ones from Nov 26-27

## Cleanup Script

Located at: `ml_pipeline/scripts/cleanup_tensorboard_experiments.py`

### List all experiments

```bash
cd ml_pipeline/scripts
python3 cleanup_tensorboard_experiments.py --list
```

### Delete experiments older than a date (DRY RUN first!)

```bash
# Dry run - see what would be deleted
python3 cleanup_tensorboard_experiments.py --delete-before 2024-12-01

# Actually delete (removes --dry-run)
python3 cleanup_tensorboard_experiments.py --delete-before 2024-12-01 --execute
```

### Delete experiments older than 7 days

```bash
# Calculate date 7 days ago
CUTOFF_DATE=$(date -v-7d +%Y-%m-%d)

# Dry run
python3 cleanup_tensorboard_experiments.py --delete-before $CUTOFF_DATE

# Execute
python3 cleanup_tensorboard_experiments.py --delete-before $CUTOFF_DATE --execute
```

### Delete ALL experiments (⚠️ DESTRUCTIVE!)

```bash
# This requires --confirm and --execute flags
python3 cleanup_tensorboard_experiments.py --delete-all --confirm --execute
```

## Recommended Cleanup Schedule

### Option 1: Manual periodic cleanup
- Keep last 7 days of experiments
- Run cleanup script monthly
- Estimated cost savings: ~$5-15/month (depending on data size)

### Option 2: Keep only important experiments
- Delete test/debugging experiments immediately
- Keep only experiments with significant results
- Manually delete after each training session

## Example Workflow

1. **After a training session:**
   ```bash
   # List all experiments
   python3 cleanup_tensorboard_experiments.py --list
   ```

2. **Identify which ones to keep:**
   - Latest successful run
   - Runs with best metrics
   - Runs with important insights

3. **Delete old experiments:**
   ```bash
   # Delete experiments older than Dec 1
   python3 cleanup_tensorboard_experiments.py --delete-before 2024-12-01 --execute
   ```

## Automation (Optional)

You can add experiment cleanup to the ML pipeline orchestrator:

```python
# In orchestrator.py, after training completes
def cleanup_old_experiments(days_to_keep=7):
    """Delete experiments older than N days."""
    from datetime import datetime, timedelta
    cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
    
    # Call cleanup script
    # ... (implementation)
```

## Cost Estimation

Based on our current setup:
- **~14 experiments** with TensorBoard data
- Each experiment: ~50-200 MB (varies based on images/metrics logged)
- Total storage: ~1-3 GB
- **Estimated monthly cost: $10-30**

### After cleanup (keeping last 7 days):
- ~3-5 recent experiments
- Total storage: ~0.3-1 GB
- **Estimated monthly cost: $3-10**

**Savings: $7-20/month**

## Important Notes

1. **Experiments are independent** - deleting old experiments doesn't affect your trained models (models are stored separately in GCS)

2. **TensorBoard instance stays "running"** - this is normal and required for viewing experiments

3. **Deletion is permanent** - make sure you've reviewed/downloaded any important visualizations before deleting

4. **Consider archiving** - if you want to keep experiment data but reduce costs, you can:
   - Download TensorBoard logs locally
   - Archive to cheaper GCS storage (Nearline/Coldline)
   - Delete from TensorBoard instance

## Troubleshooting

### "Unable to delete experiments"
- Check IAM permissions - you need `aiplatform.tensorboardExperiments.delete`
- Verify you're using the correct project ID and TensorBoard instance ID

### "Experiments still show as running"
- This is the TensorBoard **INSTANCE** status, not experiments
- Instance should remain "running"
- Check experiments with `--list` to see actual experiment status

### "Storage costs still high after deletion"
- Billing updates can take 24-48 hours
- Check actual storage usage in GCS bucket
- Some metadata may be retained temporarily

