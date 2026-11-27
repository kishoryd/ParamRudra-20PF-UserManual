---
icon: material/play-circle
---

# End-to-end Example

!!! tip "New to this?"
    [Start Here: Your First Run](start-here.md) explains the words and walks through three cases (no model and no data, your own model and data, benchmarking a model you have). This page is the longer example with the 14 B model.

## Introduction

This page takes one model, **Qwen3-14B**, from raw data to a trained LoRA adapter, using the
`qwen3-14b-qlora.yaml` recipe. Every other recipe works the same way; only the model and the node
count change (see [Recipes](recipes.md)).

The run has nine steps, all started from a login node. Steps 1 to 4 train the model; steps 5 to 9 are what you do with it afterwards.

1. Download the model and prepare the data
2. Tokenise the data (one CPU job, once per model and dataset)
3. Run a short first job (10 steps)
4. Run the full training job
5. Watch it: MLflow, and optionally the Ray Dashboard and Grafana
6. Chat with the trained model (serve)
7. Evaluate it on your own questions
8. Benchmark it against the base model
9. Save it under a name (register)

## Input

### Run folder

```bash
$ module load bhaskera/1.0.0
$ mkdir -p /scratch/<project>/<username>/bhaskera-runs && cp -r $BHASKERA_KIT/. /scratch/<project>/<username>/bhaskera-runs/
$ cd /scratch/<project>/<username>/bhaskera-runs
```

Jobs are charged to your login name, which is your Slurm account on PARAM Rudra, so there is nothing to set. If you work under another account, `export SLURM_ACCOUNT=<account>` first.
The run folder holds recipes, logs and checkpoints (large for big models), so a folder on `/scratch` is the usual choice. If you make none, the dashboard creates `~/bhaskera-runs` from the shared kit the first time it starts.

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

`alpaca` (`instruction`, `input`, `output`), `sharegpt` (`conversations` with `from` / `value`) and `prompt_completion` (`prompt`, `completion`) files
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
: "${BHASKERA_DIR:?export BHASKERA_DIR=/scratch/<project>/<username>/ExaAI before submitting}"
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
: "${BHASKERA_DIR:?export BHASKERA_DIR=/scratch/<project>/<username>/ExaAI before submitting}"

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

!!! note "Keep the adapter"
    Steps 6 to 9 use the **adapter** the run stores, which needs `logging.mlflow_log_adapter: true`. The shipped recipes and the recipes the dashboard writes set it. A full fine-tune stores no adapter, so it has no Chat, Benchmark by run or Register.

Steps 5 to 9 need the **run id** of the finished training run: the 32-character id in the **Runs** table of the dashboard (and in MLflow's run page). Every command below can also be done with a button on the [ExaAI dashboard](homepage.md); the commands are shown because they work from any terminal.

## Step 5: Watch It (MLflow, Ray Dashboard, Grafana)

Open the dashboard on a login node. Run this on your laptop and leave it open (the command comes from `bhaskera-dashboard ray --info`; add `--node login01` to land on a login node of your choice, and see [Monitoring](monitoring.md#ray-dashboard-charts-and-mlflow-in-one-place)):

```bash
$ ssh -p 4422 -L 8265:localhost:<port> <user>@paramrudra.cdacb.in "bash -lc 'module load bhaskera/1.0.0 && bhaskera-dashboard ray'"
```

Then open `http://localhost:8265`.

| Tool | Use it for | Needs |
|---|---|---|
| **MLflow** | The run's loss and evaluation curves, throughput, GPU energy and the model registry, kept after the job ends | Nothing extra (the dashboard starts it) |
| **Ray Dashboard** *(optional)* | While the job runs: every node's CPU, memory and GPU load, the workers and their logs. Open it on the job's card | A running job |
| **Grafana** *(optional)*, shown as **GPU charts** | GPU, CPU and memory over time for every node | Prometheus and Grafana installed next to ExaAI; without them the other two still work |

To look at MLflow alone, without the dashboard, run `bhaskera-dashboard start` on a login node and `bhaskera-dashboard tunnel` prints the command for your laptop. MLflow started this way stops when the login session that started it ends.

## Step 6: Chat With the Trained Model

```bash
$ ./submit.sh serve --run <run id>                 # prints the job number
$ tail -f logs/bhaskera-serve_<job>.out            # wait for: === serving on <node>:<port> ===
```

Loading takes 5 to 8 minutes for a 14 B model. On the dashboard, press **Chat** on the run instead: the window opens by itself when the model is ready, with **Tuned**, **Base** and **Compare** answers side by side. From a terminal, link to the node from the login node and ask:

```bash
$ ssh -N -L <port>:localhost:<port> <node> &
$ curl -s localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' \
       -d '{"messages": [{"role": "user", "content": "What is HPC?"}], "max_tokens": 200}'
```

Add `"use_adapter": false` to get the base model's answer from the same weights. The server stops after two hours (`SERVE_HOURS` changes it) or with `scancel <job>`, which frees the GPU node.

## Step 7: Evaluate on Your Own Questions

Standard benchmarks say little about your own use, so test the model on examples from your work. Put them in a JSONL file, one `{"question": "...", "answer": "..."}` per line, then:

```bash
$ python -m bhaskera.launcher.evaluate init-task --name my_test --data /scratch/<project>/<username>/my.jsonl
$ ./submit.sh eval --run <run id> --tasks my_test --save-samples
$ python -m bhaskera.launcher.evaluate review <folder>        # fixed by tuning, broken by tuning, still wrong
```

`review` sorts every example into **fixed by tuning**, **broken by tuning** (the base model had it right), **still wrong** and **still right**. Fix or add training examples for what is still wrong, tokenise again, train again and run the same test: the numbers show whether it helped. More in [The ExaAI Dashboard](homepage.md#test-the-model-on-your-own-questions-and-find-what-to-correct).

## Step 8: Benchmark Against the Base Model

Training loss does not say what the model can still do. Download the test data once on a login node (compute nodes have no internet), then run the standard tests on the base and the tuned model:

```bash
$ python -m bhaskera.launcher.evaluate prepare --tasks arc_easy,hellaswag
$ ./submit.sh eval --run <run id> --tasks arc_easy,hellaswag --limit 200      # --limit: a quick check; leave it out for the real score
```

The result shows the change per task and counts as *better* or *worse* only when it is larger than twice the standard error. It is stored with the run and shown in the **Runs** table; **Compare models** puts several side by side. Use the same tests and the same number of examples for every model you compare.

## Step 9: Save the Model

Give the trained model a name, a version and a note, so a colleague can find it without a run id:

```bash
$ python -m bhaskera.launcher.register add --run <run id> --name my-assistant --alias champion
$ python -m bhaskera.launcher.register list
```

Chat, benchmark and compare accept the registered name in place of a run id (`my-assistant`, `my-assistant:2`, `my-assistant@champion`). On the dashboard this is **Register**. See [Model Registry](registry.md) for versions, aliases and notes. A run that stopped early can be continued with `./submit.sh resume --run <run id>` ([Resuming a Job](resume.md)).
