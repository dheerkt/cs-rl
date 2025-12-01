#!/bin/bash
# Launch multiple DeepRacer simulator containers in parallel.
# Usage: bash scripts/launch_parallel_c7.sh [NUM_ENVS]

set -euo pipefail

NUM_ENVS=${1:-16}
BASE_PORT=${BASE_PORT:-8888}
CPUS_PER_ENV=${CPUS_PER_ENV:-3}
MEM_PER_ENV=${MEM_PER_ENV:-6g}
IMAGE=${DEEPRACER_IMAGE:-deepracer}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-deepracer}
CONFIGS_DIR="$(cd "$(dirname "$0")/.." && pwd)/configs"
EVALUATION_FLAG=${EVALUATION:-}
EVAL_WORLD=${EVAL_WORLD_NAME:-}

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required to launch the simulators." >&2
    exit 1
fi

if [ ! -d "$CONFIGS_DIR" ]; then
    echo "Configs directory not found at $CONFIGS_DIR" >&2
    exit 1
fi

mkdir -p "$CONFIGS_DIR"

echo "Launching $NUM_ENVS DeepRacer containers using image '$IMAGE'..."

for ((i=0; i<NUM_ENVS; i++)); do
    HOST_PORT=$((BASE_PORT + i))
    CONTAINER_NAME="${CONTAINER_PREFIX}_${i}"

    echo "Starting $CONTAINER_NAME on host port $HOST_PORT"

    docker run --rm -d \
        --name "$CONTAINER_NAME" \
        --cpus="$CPUS_PER_ENV" \
        --memory="$MEM_PER_ENV" \
        -p "${HOST_PORT}:8888" \
        -v "$CONFIGS_DIR:/configs:ro" \
        -e EVALUATION="$EVALUATION_FLAG" \
        -e EVAL_WORLD_NAME="$EVAL_WORLD" \
        "$IMAGE" >/dev/null

    sleep 2

done

echo "All $NUM_ENVS containers launched. Use 'docker ps | grep ${CONTAINER_PREFIX}_' to verify."
