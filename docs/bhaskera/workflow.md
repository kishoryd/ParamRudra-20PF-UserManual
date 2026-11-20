---
icon: material/play-circle
---

# End-to-end Example

## Introduction

This page takes one model, **Qwen3-14B**, from raw data to a trained LoRA adapter, using the
`qwen3-14b-qlora.yaml` recipe. Every other recipe works the same way; only the model and the node
count change (see [Recipes](recipes.md)).

The run has four steps, all started from a login node:

- Download the model and prepare the data
- Tokenise the data (one CPU job, once per model and dataset)
- Run a short first job (10 steps)
- Run the full training job

## Input

### Run folder

```bash
$ module load bhaskera/1.1.1
$ cp -r $BHASKERA_KIT /scratch/<project>/<username>/bhaskera-runs
$ cd /scratch/<project>/<username>/bhaskera-runs
$ export SLURM_ACCOUNT=<your_account>
```

### Model

Compute nodes have no internet access, so download the model on a **login node**:

```bash
$ huggingface-cli download Qwen/Qwen3-14B --local-dir /scratch/<project>/<username>/models/Qwen3-14B
```

!!! note
    Llama and Mistral models are gated: accept the licence on the model's Hugging Face page, then run
    `huggingface-cli login --token <your_token>` once before downloading.

### Training data

One conversation per line in a JSONL file, in ChatML form:

```json
{"messages": [{"role": "user", "content": "What is HPC?"}, {"role": "assistant", "content": "High-performance computing is ..."}]}
```

Put a separate validation file next to it (a few hundred to a few thousand conversations):

```text
/scratch/<project>/<username>/data/train.jsonl
/scratch/<project>/<username>/data/val.jsonl
```

`alpaca` (`instruction`, `input`, `output`) and `sharegpt` (`conversations` with `from` / `value`) files
also work: set `data.format` in the tokenize recipe.

## Step 1: Tokenise

Tokenising turns text into the numbers the model reads. It is done once; every training run reuses
the result. In `recipes/tokenize-chatml.yaml`, replace `<project>/<username>` and set `model.name` to the
model you will train, then submit:

```bash
$ ./submit.sh tokenize recipes/tokenize-chatml.yaml
```

### Sample SLURM script

`submit.sh` runs `tokenize.sbatch` on one CPU node of the `debug` partition:

```bash
#!/bin/bash
#SBATCH --job-name=bhaskera-tokenize
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=180G                     # debug partition maximum per node
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
CONFIG="${1:?usage: sbatch tokenize.sbatch <tokenize-config.yaml>}"
: "${BHASKERA_DIR:?export BHASKERA_DIR=/scratch/<project>/<username>/Bhaskera before submitting}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "$BHASKERA_DIR/bhaskera-activate.sh"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"

python -m bhaskera.launcher.tokenize --config "$CONFIG" --split both
echo "=== tokenize exit code: $? ==="
```

### Expected Output

The end of `logs/bhaskera-tokenize_<jobid>.out` (3,000 conversations took under 4 minutes):

```text
║   [train] /scratch/<project>/<username>/cache/local_train_d9cc22aaa7a4fdb4
║   [val] /scratch/<project>/<username>/cache/local_val_eb6d6d3703815f3c
║
║ Paste this into your training config:
║
║ data:
║   name: "local"
║   seq_len: 4096
║   format: "chatml"
║   tokenized_path: "/scratch/<project>/<username>/cache/local_train_d9cc22aaa7a4fdb4"
║   val_tokenized_path: "/scratch/<project>/<username>/cache/local_val_eb6d6d3703815f3c"
╚══════════════════════════════════════════════════════════════════════════════╝

=== tokenize exit code: 0 ===
```

## Step 2: Edit the Recipe

Open `recipes/qwen3-14b-qlora.yaml`, replace `<project>/<username>`, and paste the two paths:

```yaml
model:
  name: /scratch/<project>/<username>/models/Qwen3-14B
data:
  tokenized_path: /scratch/<project>/<username>/cache/local_train_d9cc22aaa7a4fdb4
  val_tokenized_path: /scratch/<project>/<username>/cache/local_val_eb6d6d3703815f3c
```

## Step 3: First Run (10 steps)

The recipe runs 10 steps. Submit it on 8 nodes with a 1-hour limit:

```bash
$ ./submit.sh recipes/qwen3-14b-qlora.yaml 8 1
```

`submit.sh` first checks the recipe on the login node (model files, libraries, data, and whether it fits
in GPU memory), then picks the partition and submits:

```text
== preflight (login node, a few seconds) ==
Preflight: recipes/qwen3-14b-qlora.yaml
[PASS] model weights              8 safetensors file(s) in /scratch/<project>/<username>/models/Qwen3-14B
[PASS] tokenizer                  Qwen2TokenizerFast, vocab 151669
[PASS] flash-attn                 2.7.3 imports against this torch
[PASS] liger-kernel               imports
[PASS] tokenized_path             1 parquet file(s), seq_len/model match
[PASS] val_tokenized_path         1 parquet file(s), seq_len/model match
[INFO] weights                    9.9 GB in 4-bit NF4 (QLoRA) (40 layers, ~0.17 GB per layer unit)
[PASS] est. per-GPU @ 16 GPUs     17.6 GiB of 80.0 GiB
[PASS] min GPUs (DDP)             any count (full replica fits)
RESULT: OK
submitted job <jobid>: qwen3-14b-qlora on 8 nodes (16 GPUs), gpu-small, 1 h
follow it:  tail -f logs/qwen3-14b-qlora_<jobid>.out | grep --line-buffered -E '\[step|eval|exit code'
```

### Sample SLURM script

`train.sbatch` starts a Ray cluster across all nodes of the job over InfiniBand and one training worker
per GPU:

```bash
#!/bin/bash
#SBATCH --job-name=bhaskera
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=170G                     # host RAM per node; the GPU partitions default to only 20 GB
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
CONFIG="${1:?usage: sbatch train.sbatch <config.yaml>}"
: "${BHASKERA_DIR:?export BHASKERA_DIR=/scratch/<project>/<username>/Bhaskera before submitting}"

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MLFLOW_ALLOW_FILE_STORE=true
source "$BHASKERA_DIR/bhaskera-activate.sh"

export NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=WARN NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_DISTRIBUTED_TIMEOUT=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if grep -qE '^[[:space:]]*strategy:[[:space:]]*"?ddp' "$CONFIG"; then CDMC=1; else CDMC=32; fi
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-$CDMC}

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_ip=$(srun --nodes=1 --ntasks=1 -w "${nodes[0]}" ip -4 addr show ib0 | awk '/inet / {print $2}' | cut -d/ -f1)
if [[ -z "$head_ip" ]]; then
    echo "=== could not read the ib0 address of ${nodes[0]} (Slurm controller unreachable or ib0 down); resubmit ==="
    exit 1
fi
port=$(( 6379 + SLURM_JOB_ID % 1000 ))
export RAY_ADDRESS="${head_ip}:${port}" RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}" RAY_raylet_start_wait_time_s=300
GPUS_PER_NODE=2
WORKERS=$(( SLURM_NNODES * GPUS_PER_NODE ))

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" \
    ray symmetric-run --address "$RAY_ADDRESS" --min-nodes "$SLURM_NNODES" \
        --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$GPUS_PER_NODE" \
        -- \
        python -m bhaskera.launcher.train --config "$CONFIG" --num-workers "$WORKERS" --no-dashboard
RC=$?
echo "=== training exit code: $RC ==="
exit $RC
```

### Expected Output

The first step appears after Ray start-up and model loading (about 5 minutes for a 14 B model, 10–20
minutes for 70 B–235 B). The last lines of a Qwen3-14B QLoRA run on 8 nodes:

```text
[epoch 0][step 9] loss=1.1703 lr=1.81e-05 grad_norm=0.2473 tok/s=27711 MFU=35.8% HFU=53.2% pad=0% step_s=4.70 MFU_step=36.0% mem=16.0/16.3GiB
[epoch 0][step 10] loss=1.0350 lr=2.01e-05 grad_norm=0.2650 tok/s=27620 MFU=35.7% HFU=53.0% pad=0% step_s=4.81 MFU_step=35.2% mem=16.0/16.3GiB
[epoch 0][step 10] eval_loss=1.5705 eval_acc=0.6557
=== training exit code: 0 ===
```

What each field means is explained in [Monitoring](monitoring.md).

## Step 4: Full Run

Work out the number of steps for one pass over your data from the first run:

```text
max_steps = (tokens in your training set) / (tok/s × step_s)
```

For example, with 1.24 billion tokens and the run above: 1,240,000,000 / (27,620 × 4.81) ≈ 9,300 steps.
Set `max_steps` in the recipe, check [Choosing Nodes and Hours](recipes.md#choosing-nodes-and-hours),
and submit:

```bash
$ ./submit.sh recipes/qwen3-14b-qlora.yaml 8 20
```

A checkpoint with the trained LoRA weights and the optimizer state is written to `checkpoint.save_dir` when the
run ends. Submitting the job again with the same `save_dir` and a larger `max_steps` continues from it.
