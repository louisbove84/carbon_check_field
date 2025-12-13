#!/bin/bash
#
# View TensorBoard Results Locally
# =================================
# This script launches TensorBoard pointing to your training logs in GCS.
# Works perfectly for viewing all confusion matrices, feature importance, etc.
#

set -e

# Configuration
BUCKET="cloud-ai-platform-e1b347d9-9a9e-445e-b937-34089ed1d347"
TENSORBOARD_ID="3503556418512879616"
PORT=6007

echo "🚀 TensorBoard Viewer"
echo "===================="
echo ""

# Check if specific run is provided
if [ -n "$1" ]; then
    RUN_DIR="$1"
    echo "📊 Viewing specific run: $RUN_DIR"
    LOGDIR="gs://${BUCKET}/tensorboard-${TENSORBOARD_ID}/${RUN_DIR}/default/"
else
    # View all runs for this TensorBoard instance
    echo "📊 Viewing all runs in TensorBoard instance"
    LOGDIR="gs://${BUCKET}/tensorboard-${TENSORBOARD_ID}/"
fi

echo "   Log directory: $LOGDIR"
echo "   Port: $PORT"
echo ""

# Kill any existing TensorBoard on this port
pkill -f "tensorboard.*${PORT}" 2>/dev/null || true

echo "🌐 Starting TensorBoard..."
echo "   Open in browser: http://localhost:${PORT}"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

# Start TensorBoard
python3 -m tensorboard.main --logdir="$LOGDIR" --port=$PORT --bind_all

echo ""
echo "✅ TensorBoard stopped"

