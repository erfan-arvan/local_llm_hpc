# local_llm_hpc

Minimal example of running a local LLM (Qwen2.5-Coder via [vLLM](https://github.com/vllm-project/vllm)) as a GPU job on the NJIT HPC cluster (Wulver), with a sample script to query it.

## What's here

- `scripts/start_vllm_server.sh` — starts a vLLM OpenAI-compatible API server for a chosen model.
- `scripts/query_vllm.py` — sends one prompt to the running server and prints the reply.
- `submit_qwen_job.sh` — a self-contained SLURM batch job: creates its own venv and installs vLLM on first run, starts the server, waits for it to be healthy, runs a sample query, then shuts it down. No manual setup on a login node required.

## Running

1. Clone the repo on the cluster:

   ```bash
   git clone https://github.com/erfan-arvan/local_llm_hpc.git
   cd local_llm_hpc
   ```

2. Edit `--account` (and any other `#SBATCH` lines) in `submit_qwen_job.sh` to match your own SLURM allocation.

3. Submit:

   ```bash
   sbatch submit_qwen_job.sh
   squeue -u $USER
   ```

4. Watch progress:

   ```bash
   tail -f qwen-job.<jobid>.out
   ```

The **first run** installs vLLM into a local `venv/` (a few minutes) and downloads the model weights (larger, one-time). Every run after that reuses the existing `venv/` and cached weights, so it starts much faster.

You should see something like:

```
[JOB] No venv found — creating one and installing vLLM (first run only)...
...
[JOB] vLLM is ready
[JOB] Running sample query...
A large language model is a sophisticated artificial intelligence system trained on vast amounts of text data to understand and generate human-like text.
[JOB] Stopping vLLM server
[JOB] Finished
```

## Querying interactively

If you'd rather keep a server running and send it multiple prompts (instead of the one-shot batch job above), request an interactive allocation, start the server in the background, and call the query script as many times as you like:

```bash
salloc --partition=gpu --qos=standard --account=CHANGE_ME --gres=gpu:1 --time=01:00:00
module load easybuild
module load CUDA/12.8.0
module load GCCcore/13.3.0
module load Python/3.12.3

export HF_HOME=/scratch/YOUR_ACCOUNT/YOUR_USERNAME/local_llm_hpc_cache/huggingface
export PIP_CACHE_DIR=/scratch/YOUR_ACCOUNT/YOUR_USERNAME/local_llm_hpc_cache/pip
export TMPDIR=/scratch/YOUR_ACCOUNT/YOUR_USERNAME/local_llm_hpc_cache/tmp
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

source venv/bin/activate   # after at least one run of submit_qwen_job.sh has created it

bash scripts/start_vllm_server.sh qwen32b &   # leave running in the background
# wait for "Uvicorn running on http://0.0.0.0:8001" in the output, then:

python3 scripts/query_vllm.py "Explain recursion in one paragraph"
python3 scripts/query_vllm.py "Write a Python function to reverse a string"
```

## Notes

- **Don't let caches or venvs land in your home directory.** On this cluster, home (and `/tmp`) sit on a small quota-limited local disk — a 32B-parameter model's weights alone are tens of GB and will blow right through it. `submit_qwen_job.sh` redirects `HF_HOME`, `PIP_CACHE_DIR`, and `TMPDIR` to `/scratch` for exactly this reason; keep that pattern if you adapt the scripts.
- The vLLM server only listens on `127.0.0.1`, so it's only reachable from the *same node* it's running on — that's why the server and the client always run inside the same SLURM allocation/job.
- Swap in a different model by adding an entry to the `MODELS` table in `scripts/start_vllm_server.sh` and passing that key as the first argument (e.g. `bash scripts/start_vllm_server.sh qwen7b`).
- Check available modules on your cluster with `module avail CUDA` / `module spider <name>` — some modules (like `Python/3.12.3` on this cluster) require a prerequisite toolchain module (`GCCcore/13.3.0`) to be loaded first, or Lmod will refuse to load them even though `module avail` lists them.
- Check your quota with `quota -s` before a first run, especially for larger models.
