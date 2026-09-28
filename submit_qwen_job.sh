#!/bin/bash -l
# =============================================================================
# submit_qwen_job.sh — SLURM batch job that starts a local vLLM server on an
# NJIT HPC (Wulver) GPU node and runs one sample query against it.
#
# Fully self-contained: creates its own venv and installs vLLM on first run
# (skipped on later runs once the venv exists), so there's no manual setup
# step on a login node required.
#
# Edit the #SBATCH lines (especially --account) for your own allocation
# before submitting, then:
#
#   sbatch submit_qwen_job.sh
#   squeue -u $USER
#   tail -f qwen-job.<jobid>.out
# =============================================================================
#SBATCH --job-name=qwen-job
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=CHANGE_ME        # your PI's SLURM account, e.g. mjk76
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00

set -euo pipefail

echo "[JOB] Starting on $(hostname)"
date

# Modules — adjust versions to whatever's available on your cluster
# (check with `module avail CUDA` / `module spider <name>`).
module load easybuild
module load CUDA/12.8.0
module load Python/3.12.3

cd "$SLURM_SUBMIT_DIR"

if [ ! -d venv ]; then
    echo "[JOB] No venv found — creating one and installing vLLM (first run only)..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install vllm
else
    echo "[JOB] Reusing existing venv"
    source venv/bin/activate
fi

echo "[JOB] Launching vLLM server..."
bash scripts/start_vllm_server.sh qwen32b > vllm_server.log 2>&1 &
VLLM_PID=$!

echo "[JOB] Waiting for vLLM health..."
READY=0
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1; then
        echo "[JOB] vLLM is ready"
        READY=1
        break
    fi
    sleep 10
done

if [ "$READY" -ne 1 ]; then
    echo "[ERROR] vLLM failed to start within timeout"
    tail -n 100 vllm_server.log
    kill "$VLLM_PID" 2>/dev/null || true
    exit 1
fi

echo "[JOB] Running sample query..."
python3 scripts/query_vllm.py "In one sentence, what is a large language model?"

echo "[JOB] Stopping vLLM server"
kill "$VLLM_PID" 2>/dev/null || true

date
echo "[JOB] Finished"
