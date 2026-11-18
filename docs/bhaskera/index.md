# Bhaskera (Distributed LLM Fine-tuning)

**Bhaskera** is a Ray-native framework for fine-tuning large language models (LLMs) with
**LoRA** on PARAM Rudra's GPU nodes, from **1 node up to 128 nodes (256 A100 GPUs)**. One
YAML file describes a run: the model, the data, and how to spread the work across GPUs.
Bhaskera starts Ray on the nodes SLURM gives your job, loads and splits the model, trains,
evaluates, saves checkpoints and reports live metrics.

It has been benchmarked on PARAM Rudra with five models from 14 B to 235 B parameters,
including mixture-of-experts (MoE) models, at 8–128 nodes (see
[Benchmark results](benchmarks.md)).

!!! info "At a glance"
    | | |
    |---|---|
    | **Module** | `module load bhaskera/2.2.0` |
    | **Source** | <https://github.com/CDACIndia/Bhaskera> (version 2.2.0, Python ≥ 3.11) |
    | **Tested models** | Qwen3-14B, Llama-3.3-70B, Mistral-Large-123B (dense); Qwen3-30B-A3B, Qwen3-235B-A22B (MoE) |
    | **Parallelism** | DDP, FSDP2, HSDP, QLoRA + DDP (4-bit), Pipeline (1F1B / GPipe) |
    | **Partitions** | `gpu-debug` (tests, tokenising), `gpu-small`, `gpu-large`, `gpu-massive` |
    | **Monitoring** | Per-step log line, MLflow, Ray Dashboard |

## Quick Start

```bash
ssh -p 4422 <username>@paramrudra.cdacb.in
module load bhaskera/2.2.0

mkdir -p /scratch/<project>/<username>/bhaskera-runs && cd $_ && mkdir -p configs logs
cp $BHASKERA_HOME/configs/release/rel-ddp-qwen14b.yaml configs/my_run.yaml   # set model and data paths
bhaskera-preflight --config configs/my_run.yaml --num-gpus 16                 # all PASS?

sbatch -A <your_account> --nodes=8 --partition=gpu-small --time=08:00:00 \
       $BHASKERA_HOME/scripts/train.sbatch configs/my_run.yaml
tail -f logs/bhaskera_<jobid>.out | grep --line-buffered -E '\[step|eval'
```

## In This Section

| Page | What it covers |
|---|---|
| [Installation](installation.md) | Loading the shared module, or installing your own copy |
| [Fine-tuning Workflow](workflow.md) | Model and data, tokenising, configs, preflight check, submitting the job |
| [Monitoring](monitoring.md) | The step log, the MLflow dashboard, the Ray Dashboard and SSH tunnels |
| [Parallelism Strategies](parallelism.md) | DDP, QLoRA, Pipeline, FSDP and HSDP: which to use and how to set it |
| [Benchmarks on PARAM Rudra](benchmarks.md) | 125-run results for five models at 8–128 nodes |
| [Troubleshooting & Reference](troubleshooting.md) | Common errors, known limitations, command reference |

## Overview

Fine-tuning adapts a pre-trained model to your own data. Bhaskera uses **LoRA**: instead of
updating all of the model's weights, it trains small add-on weights (an *adapter*) and keeps
the base model frozen. This needs far less GPU memory and far less communication between
nodes, which is what lets a 70 B model train efficiently on many PARAM Rudra nodes.

| Feature | What it gives you |
|---|---|
| Five parallelism strategies | Choose per run in YAML how the model is spread across GPUs (see [Choosing a parallelism strategy](parallelism.md)) |
| QLoRA | Stores the frozen model in 4-bit so a 70 B–123 B model fits on one 80 GB A100 |
| MoE support | Frozen routers, load-balancing loss, router z-loss; fast expert computation with the Pipeline strategy |
| Tokenisation cache | `bhaskera-tokenize` tokenises your data once into Parquet files; every run reuses them |
| Sequence packing | Joins short conversations into full rows so no GPU time is spent on padding |
| Chat formats | `chatml`, `alpaca`, `sharegpt`, or your own registered format |
| Evaluation | Held-out loss and next-token accuracy during training |
| Preflight check | `bhaskera-preflight` checks model, libraries, data and memory fit on the login node, in seconds |
| Checkpoint and resume | Sharded checkpoints; a stopped job continues from the last save |
| Observability | One log line per step, MLflow charts, Ray Dashboard |

## PARAM Rudra Environment

Bhaskera runs on the GPU nodes described in [GPU Computing](../gpu.md): each node has
**2 × NVIDIA A100 80 GB**, 48 CPU cores and about 187 GB RAM.

| Item | Value on PARAM Rudra | Why it matters for Bhaskera |
|---|---|---|
| GPUs per node | 2 × A100 80 GB (`sm_80`) | Workers = 2 × nodes |
| Interconnect | 100 Gb/s InfiniBand per node (`ib0`) | Ray and NCCL use `ib0` |
| GPUDirect RDMA | Not usable (the NIC sits on the other CPU socket) | Inter-node GPU traffic is staged through host memory; this is why FSDP slows at 128 nodes |
| Internet on compute nodes | **No** | Download models and datasets on a login node first |
| Storage | `/scratch` for code, models, data, logs | Keep everything on `/scratch` |
| Shared `$HOME` | Visible on login and compute nodes | MLflow writes run history to `~/mlflow-runs` |
| CUDA | `module load cuda/12.4` (done by Bhaskera's environment script) | Matches the PyTorch wheels Bhaskera installs |

Partitions used (full policy in [Batch System (SLURM)](../batch.md#partitions-and-limits)):

| Partition | Nodes per job | Max walltime | Use with Bhaskera |
|---|---|---|---|
| `gpu-debug` | 1–4 | 01:00:00 | Tokenising, short tests, rehearsals |
| `gpu-small` | 5–25 | 24:00:00 | 8–16-node training |
| `gpu-large` | 26–100 | 12:00:00 | 32–64-node training |
| `gpu-massive` | 101–200 | 06:00:00 | 128-node training |

!!! note "Account"
    As for every job on PARAM Rudra, pass your project account with `-A <your_account>`
    (see [Your account (`-A`) is mandatory](../batch.md#your-account-a-is-mandatory)).
