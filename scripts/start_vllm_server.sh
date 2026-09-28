#!/bin/bash
# =============================================================================
# start_vllm_server.sh — Launch a local vLLM OpenAI-compatible server on an
# NJIT HPC (Wulver) GPU node.
#
# Usage:
#   bash scripts/start_vllm_server.sh [model-key]
#
# model-key selects a preset from the MODELS table below (default: qwen32b).
# Add your own entries there for other models.
#
# The server listens on http://127.0.0.1:${PORT:-8001}/v1 — only reachable
# from the same node it's running on, which is why it's normally started in
# the same SLURM job/allocation as the client that calls it (see
# submit_qwen_job.sh in this repo for a full example).
# =============================================================================
set -euo pipefail

MODEL_KEY="${1:-qwen32b}"
PORT="${PORT:-8001}"

declare -A MODELS=(
    [qwen32b]="Qwen/Qwen2.5-Coder-32B-Instruct"
    [qwen7b]="Qwen/Qwen2.5-Coder-7B-Instruct"
)

MODEL_NAME="${MODELS[$MODEL_KEY]:-}"
if [ -z "$MODEL_NAME" ]; then
    echo "[ERROR] Unknown model key '$MODEL_KEY'. Available: ${!MODELS[*]}"
    exit 1
fi

echo "[INFO] Starting vLLM server"
echo "[INFO]   model : $MODEL_NAME"
echo "[INFO]   port  : $PORT"

# --tensor-parallel-size should match the number of GPUs requested in your
# SLURM allocation (--gres=gpu:N). Adjust --gpu-memory-utilization down if
# you hit OOM on smaller GPUs.
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$PORT" \
    --tensor-parallel-size "${VLLM_TP_SIZE:-1}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.90}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-16384}" \
    --trust-remote-code
