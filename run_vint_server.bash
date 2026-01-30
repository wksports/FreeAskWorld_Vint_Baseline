#!/bin/bash

# Start this script first, then launch Unity

# Auto-detect project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv_py310"
BASELINE_NAME="vint_baseline"
EPISODES=${EPISODES:-46}
TCP_PORT=10000

# ===== User Configuration =====
# RUN_MODE: 'full' (all episodes) or 'single' (one episode)
RUN_MODE="single"

# Full mode: total episode count
FULL_RUN_EPISODE_COUNT=46

# Single mode: which episode to use
SINGLE_EPISODE_NUMBER=15

# Activate Python 3.10 virtualenv (must be before sourcing ROS2)
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Activated Python 3.10 virtualenv at $VENV_PATH"
else
    echo "Error: Virtualenv not found at $VENV_PATH"
    exit 1
fi

# Source ROS2 environment
source /opt/ros/humble/setup.bash
source "$PROJECT_ROOT/install/setup.bash"

# ROS2 DDS: use CycloneDDS for better performance
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Disable multicast to reduce network spam
export CYCLONEDDS_URI="<Disc><DefaultMulticastAddress>0.0.0.0</></>"

# Set PYTHONPATH
export PYTHONPATH=$VENV_PATH/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/visualnav-transformer-main/train:$PYTHONPATH

# ViNT model path
export MODEL_PATH="$PROJECT_ROOT/model_weights/vint.pth"

# Topomap base directory
export TOPOMAP_BASE_DIR="$PROJECT_ROOT/topomaps/images"
ORIGINAL_TOPOMAP_BASE_DIR="$TOPOMAP_BASE_DIR"

# Configure run mode
case "$RUN_MODE" in
    full)
        EPISODES=$FULL_RUN_EPISODE_COUNT
        export TOTAL_CYCLES=1
        export EPISODES_PER_CYCLE=$EPISODES
        export TOTAL_EPISODES=$EPISODES
        RUN_MODE_DESC="Full loop (${FULL_RUN_EPISODE_COUNT} episodes)"
        ;;
    single)
        EPISODES=1
        export TOTAL_CYCLES=1
        export EPISODES_PER_CYCLE=1
        export TOTAL_EPISODES=1

        SINGLE_SOURCE_DIR="$ORIGINAL_TOPOMAP_BASE_DIR/episode_$SINGLE_EPISODE_NUMBER"
        if [ ! -d "$SINGLE_SOURCE_DIR" ]; then
            echo "❌ Error: Topomap for episode $SINGLE_EPISODE_NUMBER not found at $SINGLE_SOURCE_DIR"
            exit 1
        fi

        # Create temporary symlink for single episode
        TMP_TOPOMAP_DIR="$PROJECT_ROOT/topomaps/tmp_single_run"
        rm -rf "$TMP_TOPOMAP_DIR"
        mkdir -p "$TMP_TOPOMAP_DIR"
        ln -sfn "$SINGLE_SOURCE_DIR" "$TMP_TOPOMAP_DIR/episode_1"
        export TOPOMAP_BASE_DIR="$TMP_TOPOMAP_DIR"
        RUN_MODE_DESC="Single episode (source episode $SINGLE_EPISODE_NUMBER)"
        ;;
    *)
        echo "❌ Error: Unknown RUN_MODE '$RUN_MODE'. Use 'full' or 'single'."
        exit 1
        ;;
esac

# ViNT navigation parameters (tuned to avoid getting stuck)
export VINT_CLOSE_THRESHOLD=2    # Lower threshold, easier to reach nodes
export VINT_RADIUS=6             # Larger search radius, consider more candidates
export VINT_WAYPOINT_INDEX=1     # Use closer waypoint, more conservative movement

echo "Using Python: $(python3 --version) from $(which python3)"

# Verify configuration
echo ""
echo "=========================================="
echo "ViNT Run Configuration:"
echo "Mode: $RUN_MODE_DESC"
echo "Model Path: $MODEL_PATH"
echo "Topomap Base Dir: $TOPOMAP_BASE_DIR"
echo "Episodes: $TOTAL_EPISODES"
echo "=========================================="

# Check model file
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found at $MODEL_PATH"
    exit 1
else
    echo "✓ Model file found"
fi

# Check topomap directory
if [ ! -d "$TOPOMAP_BASE_DIR" ]; then
    echo "❌ Error: Topomap base directory not found at $TOPOMAP_BASE_DIR"
    exit 1
else
    # Verify all required topomaps exist
    echo "Checking topomaps for episodes 1-$TOTAL_EPISODES..."
    missing_episodes=()
    
    for i in $(seq 1 $TOTAL_EPISODES); do
        episode_dir="$TOPOMAP_BASE_DIR/episode_$i"
        if [ ! -d "$episode_dir" ]; then
            missing_episodes+=("episode_$i")
        fi
    done
    
    if [ ${#missing_episodes[@]} -eq 0 ]; then
        # Show info for first and last episodes
        EPISODE_1_DIR="$TOPOMAP_BASE_DIR/episode_1"
        EPISODE_LAST_DIR="$TOPOMAP_BASE_DIR/episode_$TOTAL_EPISODES"
        IMAGE_COUNT_1=$(ls "$EPISODE_1_DIR"/*.jpg 2>/dev/null | wc -l)
        IMAGE_COUNT_LAST=$(ls "$EPISODE_LAST_DIR"/*.jpg 2>/dev/null | wc -l)
        echo "✓ All $TOTAL_EPISODES topomaps found!"
        echo "  - Episode 1: $IMAGE_COUNT_1 images"
        echo "  - Episode $TOTAL_EPISODES: $IMAGE_COUNT_LAST images"
    else
        echo "❌ Error: Missing topomaps for episodes: ${missing_episodes[*]}"
        echo "Please record topomaps for all $TOTAL_EPISODES episodes first"
        exit 1
    fi
fi

echo ""

# Check if port is in use
if lsof -i tcp:$TCP_PORT >/dev/null 2>&1; then
    echo "Port $TCP_PORT is in use. Killing process..."
    sudo lsof -t -i tcp:$TCP_PORT | xargs sudo kill -9
fi

# Create log directory
mkdir -p "$PROJECT_ROOT/logs"

# Start TCP Endpoint
echo "Starting TCP Endpoint..."
ros2 run ros_tcp_endpoint default_server_endpoint > "$PROJECT_ROOT/logs/tcp_endpoint.log" 2>&1 &
TCP_PID=$!
echo "TCP Endpoint started (PID: $TCP_PID, log: logs/tcp_endpoint.log)"
sleep 1

# Wait for Unity connection
echo ""
echo "=========================================="
echo "Waiting for Unity to connect..."
echo "Please start Unity simulator now."
echo "Model: $MODEL_PATH"
echo "=========================================="
echo ""

# Detect Unity connection (by checking log for connection message)
TIMEOUT=300  # Wait up to 5 minutes
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if grep -q "Connection from" "$PROJECT_ROOT/logs/tcp_endpoint.log" 2>/dev/null; then
        echo "✓ Unity connected!"
        break
    fi
    
    # Show waiting message every 5 seconds
    if [ $((ELAPSED % 5)) -eq 0 ]; then
        echo "Waiting for Unity connection... (${ELAPSED}s elapsed)"
    fi
    
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "Warning: Timeout waiting for Unity connection. Proceeding anyway..."
fi

sleep 2

# Additional plugins (commented out)
# bash mapping.bash &

# Run ViNT baseline node
echo ""
echo "🚀 Starting ViNT baseline with $TOTAL_EPISODES episodes..."
echo ""
echo "Each episode uses its corresponding topomap:"
echo "  - Episode 1 → $TOPOMAP_BASE_DIR/episode_1"
echo "  - Episode 2 → $TOPOMAP_BASE_DIR/episode_2"
echo "  - ..."
echo "  - Episode $TOTAL_EPISODES → $TOPOMAP_BASE_DIR/episode_$TOTAL_EPISODES"
echo ""
echo "🔄 Starting navigation..."
ros2 run vln_connector $BASELINE_NAME

echo ""
echo "All episodes completed!"

# Debug utilities (commented out)
# ros2 run tf2_tools view_frames
