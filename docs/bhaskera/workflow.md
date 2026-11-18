# Fine-tuning Workflow

This page walks through a complete fine-tuning run on PARAM Rudra, from raw data to a running
multi-node job. It assumes Bhaskera is available ([Installation](installation.md)).

## Step 1: Get the Model and Data

Compute nodes have no internet access, so download on a **login node** into `/scratch`:

```bash
module load bhaskera/2.2.0
huggingface-cli download Qwen/Qwen3-14B --local-dir /scratch/<project>/<username>/models/Qwen3-14B
```

Your training data goes in JSONL files, one example per line. The three built-in formats:

=== "chatml"

    ```json
    {"messages": [{"role": "user", "content": "What is HPC?"}, {"role": "assistant", "content": "High-performance computing is ..."}]}
    ```

=== "alpaca"

    ```json
    {"instruction": "Summarise the text.", "input": "PARAM Rudra is ...", "output": "A 20 PF system ..."}
    ```

=== "sharegpt"

    ```json
    {"conversations": [{"from": "human", "value": "What is HPC?"}, {"from": "gpt", "value": "High-performance computing is ..."}]}
    ```

Keep a separate validation file (a few hundred to a few thousand examples) for evaluation.

## Step 2: Tokenise the Data

Tokenising turns text into the token IDs the model reads. Bhaskera does it **once** and saves
the result (a Parquet cache); every training run then reads the cache.

Start from the example config (copied in the step above) and set your model, files and
sequence length in `configs/my_tokenize.yaml`:

```yaml
model:
  name: /scratch/<project>/<username>/models/Qwen3-14B
data:
  name: local
  format: chatml                 # or alpaca / sharegpt
  train_path: /scratch/<project>/<username>/data/train.jsonl
  val_path:   /scratch/<project>/<username>/data/val.jsonl
  seq_len: 4096
  cache_dir: /scratch/<project>/<username>/cache
```

Run it as a small job on `gpu-debug`:

```bash
#!/bin/bash
#SBATCH --job-name=bhaskera-tokenize
#SBATCH --account=<your_account>
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module load bhaskera/2.2.0        # Option B: source <your clone>/bhaskera-activate.sh
cd /scratch/<project>/<username>/bhaskera-runs
bhaskera-tokenize --config configs/my_tokenize.yaml --split both
```

At the end it prints the lines to paste into your training config:

```text
data:
  tokenized_path: "cache/local_train_<hash>"
  val_tokenized_path: "cache/local_val_<hash>"
```

!!! warning "Do not tokenise on a login node"
    Tokenising uses Ray and many CPU cores. On a busy shared login node it can stall; run it
    as a job as shown above. In testing, 20,000 conversations tokenised in under 2 minutes on one node.

## Step 3: Choose and Edit a Training Config

`$BHASKERA_HOME/configs/release/` holds the 15 configurations tested on PARAM Rudra. Pick the
one closest to your model and strategy, copy it and edit the paths:

```bash
cp $BHASKERA_HOME/configs/release/rel-ddp-qwen14b.yaml configs/my_run.yaml
```

The fields you usually change:

```yaml
model:
  name: /scratch/<project>/<username>/models/Qwen3-14B
  quantization: none             # "qlora" for 4-bit QLoRA
data:
  tokenized_path: /scratch/<project>/<username>/cache/local_train_<hash>
  val_tokenized_path: /scratch/<project>/<username>/cache/local_val_<hash>
  runtime_packing: true          # recommended: no compute wasted on padding
  num_workers: 16                # 2 x number of nodes
training:
  batch_size: 1                  # rows per GPU per step
  max_steps: 2000                # the release configs use 10 (they are tests)
  eval_steps: 200                # evaluate every N steps
  eval_max_rows: 1000            # cap rows per evaluation
  distributed:
    strategy: ddp                # ddp | fsdp | pipeline
checkpoint:
  enabled: true                  # the release configs switch this off
  save_dir: ./checkpoints/my_run
logging:
  tracker: mlflow
  run_name: my_run
```

| Config file | Model | Strategy |
|---|---|---|
| `rel-ddp-qwen14b.yaml` | Qwen3-14B | DDP |
| `rel-fsdp-qwen14b.yaml`, `rel-hsdp-qwen14b.yaml` | Qwen3-14B | FSDP, HSDP |
| `rel-pp-qwen14b-1f1b-packed-b2.yaml`, `rel-pp-qwen14b-gpipe-ackpt-b2.yaml` | Qwen3-14B | Pipeline (1F1B, GPipe) |
| `rel-qlora-llama70b.yaml` | Llama-3.3-70B | QLoRA + DDP |
| `rel-fsdp-llama70b.yaml`, `rel-pp-llama70b-a.yaml` | Llama-3.3-70B | FSDP, Pipeline |
| `rel-pp-mistral123b.yaml` | Mistral-Large-123B | Pipeline |
| `rel-pp-moe-qwen30b.yaml`, `rel-fsdp-moe-qwen30b.yaml` | Qwen3-30B-A3B (MoE) | Pipeline, FSDP |
| `rel-pp-moe-qwen235b.yaml` | Qwen3-235B-A22B (MoE) | Pipeline |

## Step 4: Preflight Check

Before queueing, check the run on the login node. It takes seconds and needs no GPU:

```bash
bhaskera-preflight --config configs/my_run.yaml --num-gpus 16
```

```text
[PASS] model weights              8 safetensors file(s) in .../models/Qwen3-14B
[PASS] tokenizer                  Qwen2TokenizerFast, vocab 151669
[PASS] flash-attn                 2.7.3 imports against this torch
[PASS] liger-kernel               imports
[PASS] tokenized_path             11 parquet file(s), seq_len/model match
[PASS] val_tokenized_path         1 parquet file(s), seq_len/model match
[PASS] est. per-GPU @ 16 GPUs     35.9 GiB of 80.0 GiB
[PASS] min GPUs (DDP)             any count (full replica fits)
RESULT: OK
```

Fix every `FAIL` before submitting. Leave out `--num-gpus` to see the memory estimate for every GPU count.

## Step 5: Submit the Training Job

`$BHASKERA_HOME/scripts/train.sbatch` is the launcher used for all benchmark runs. It starts a Ray cluster
across every node of the job over InfiniBand, sets the NCCL and memory settings, and starts one
training worker per GPU.

```bash
module load bhaskera/2.2.0
cd /scratch/<project>/<username>/bhaskera-runs
mkdir -p logs                                   # SLURM does not create it
sbatch -A <your_account> --nodes=8 --partition=gpu-small --time=08:00:00 \
       $BHASKERA_HOME/scripts/train.sbatch configs/my_run.yaml
```

The job inherits the loaded module, so every node uses the same installation.

| Nodes | Partition | Example |
|---|---|---|
| 1–4 | `gpu-debug` (≤ 1 h) | `sbatch -A <acct> --nodes=2 --partition=gpu-debug --time=00:45:00 $BHASKERA_HOME/scripts/train.sbatch configs/my_run.yaml` |
| 8–16 | `gpu-small` | `sbatch -A <acct> --nodes=16 --partition=gpu-small --time=12:00:00 ...` |
| 32–64 | `gpu-large` | `sbatch -A <acct> --nodes=64 --partition=gpu-large --time=12:00:00 ...` |
| 128 | `gpu-massive` (≤ 6 h) | `sbatch -A <acct> --nodes=128 --partition=gpu-massive --time=06:00:00 ...` |

The job log is `logs/bhaskera_<jobid>.out`. The first training step appears after Ray startup
and model loading:

| Model size | Time to first step |
|---|---|
| ~14 B | about 5 min |
| 70 B–123 B | about 10–15 min |
| 235 B (MoE) | about 15–20 min |

!!! warning "Check the log, not just the job state"
    A run can fail (for example on a faulty node) while SLURM still shows the job as
    `COMPLETED`. Confirm that the log contains `[step` lines and ends with
    `=== training exit code: 0 ===`.

Next: [Monitoring](monitoring.md).
