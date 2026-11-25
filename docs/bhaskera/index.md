---
icon: material/brain
---

# Bhaskera

## Introduction

Bhaskera is a framework for fine-tuning large language models (LLMs), and models that also read images, video and
audio, on Slurm GPU clusters. On PARAM Rudra it runs from 1 node up to 128 nodes (256 A100 GPUs). It uses **LoRA**: the pre-trained model stays frozen and
only small add-on weights (an *adapter*) are trained, which needs far less GPU memory and far less
communication between nodes than full fine-tuning.

One YAML file (a *recipe*) describes a run: the model, the data, and how the work is spread across
GPUs. Bhaskera starts a Ray cluster on the nodes SLURM gives your job, loads and splits the model,
trains, evaluates, saves checkpoints and prints one line of metrics per step. A web page lists your jobs, starts new
ones, and lets you chat with the model you trained.

Official site: <https://github.com/kishoryd/Bhaskera>.

Bhaskera has been benchmarked on PARAM Rudra with five models from 14 B to 235 B parameters, including
mixture-of-experts (MoE) models, on 8 to 128 nodes (see [Benchmarks](benchmarks.md)).

| | |
|---|---|
| **Version** | 1.0.0 (Python 3.11, PyTorch 2.6 with CUDA 12.4, transformers 4.57) |
| **Tested models** | Qwen3-14B, Llama-3.3-70B, Mistral-Large-123B (dense); Qwen3-30B-A3B, Qwen3-235B-A22B (MoE) |
| **Parallelism** | QLoRA + DDP (4-bit), DDP, Pipeline, FSDP, HSDP |
| **Multimodal** | Images, video and audio with Qwen3-VL and Qwen2.5-Omni (DDP, FSDP, HSDP) |
| **Partitions** | `debug` (tokenising), `gpu-small`, `gpu-large`, `gpu-massive` (training) |

## Quick Start

```bash
$ module load bhaskera/1.0.0
$ cp -r $BHASKERA_KIT /scratch/<project>/<username>/bhaskera-runs     # your run folder: any folder you choose
$ cd /scratch/<project>/<username>/bhaskera-runs

$ ./submit.sh tokenize recipes/tokenize-chatml.yaml        # once: tokenise your data
$ ./submit.sh recipes/qwen3-14b-qlora.yaml 8 12             # recipe, nodes, hours
$ tail -f logs/qwen3-14b-qlora_<jobid>.out | grep --line-buffered -E '\[step|eval|exit code'
```

!!! note
    If `module avail bhaskera` shows nothing, install your own copy as described in
    [Installation](installation.md); the rest of this guide is the same.

## In This Section

| Page | What it covers |
|---|---|
| [Why Bhaskera](why.md) | Features, measured benefits, and how it is built |
| [Installation](installation.md) | Loading the module, or installing your own copy |
| [End-to-end Example](workflow.md) | Model and data, tokenising, training, expected output, resuming |
| [Recipes](recipes.md) | Ready configurations for each model, and how many nodes and hours to request |
| [Monitoring](monitoring.md) | The step log, MLflow, and the Ray Dashboard with charts for every node |
| [Home Page](homepage.md) | One page for your jobs: start a job, chat with a trained model, benchmark it |
| [Multimodal](multimodal.md) | Fine-tuning with images, video and audio |
| [Model Registry](registry.md) | Naming and versioning trained models, aliases such as champion, using them by name |
| [Resuming a Job](resume.md) | Checkpoints, continuing after a failure, going back after a loss spike |
| [Parallelism Strategies](parallelism.md) | QLoRA, DDP, Pipeline, FSDP and HSDP: which to use |
| [Benchmarks](benchmarks.md) | Measured speed, MFU, memory and accuracy for every model, strategy and node count |
| [Troubleshooting & Reference](troubleshooting.md) | Common errors, limitations, command reference |

## PARAM Rudra Environment

| Item | Value | Why it matters for Bhaskera |
|---|---|---|
| GPUs per node | 2 × NVIDIA A100 80 GB | Training workers = 2 × nodes |
| Interconnect | InfiniBand (`ib0`) | Ray and NCCL communicate over `ib0` |
| GPUDirect RDMA | Not usable | GPU-to-GPU traffic between nodes goes through host memory; this is why FSDP slows down at 128 nodes |
| GPUs on login nodes | None | Installation must name CUDA explicitly (see [Installation](installation.md)) |
| Storage | `/scratch` | Keep code, models, data, checkpoints and logs on `/scratch` |

| Partition | Nodes per job | Max walltime | Used for |
|---|---|---|---|
| `debug` | 1 (CPU node) | 01:00:00 | Tokenising data |
| `gpu-small` | 5–25 | 24:00:00 | 8–16-node training |
| `gpu-large` | 26–100 | 12:00:00 | 32–64-node training |
| `gpu-massive` | 101–200 | 06:00:00 | 128-node training |

`submit.sh` chooses the partition from the node count you ask for. Full policy:
[Resource Management](../resource-management.md).
