# local_llm_hpc

Minimal example of running a local LLM (Qwen2.5-Coder via [vLLM](https://github.com/vllm-project/vllm)) as a GPU job on the NJIT HPC cluster (Wulver), with a sample script to query it.

## What's here

- `scripts/start_vllm_server.sh` — starts a vLLM OpenAI-compatible API server for a chosen model.
- `scripts/query_vllm.py` — sends one prompt to the running server and prints the reply.
- `submit_qwen_job.sh` — a SLURM batch job that ties the two together: starts the server, waits for it to be healthy, runs a sample query, then shuts it down.

## Setup (one time, on a login node)

```bash
git clone https://github.com/erfan-arvan/local_llm_hpc.git
cd local_llm_hpc

module load easybuild
module load CUDA/12.8.0

python3 -m venv venv
source venv/bin/activate
pip install vllm
