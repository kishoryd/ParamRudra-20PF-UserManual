---
icon: material/brain
---

# Bhaskera

## Introduction

Bhaskera teaches a pre-trained language model something new, on a cluster of GPUs, from the command line. You give it three things: a **model** to start from, **data** to learn from, and one small file (a *recipe*) that says how. It does the rest: it checks that everything is in place, asks Slurm for the GPUs, starts the training across all of them, saves progress as it goes, and keeps a record of every run. It handles text models, and models that also read images, video and audio.

This is called **fine-tuning**. A model such as Qwen3 or Llama has learned general language from a huge amount of text. Fine-tuning continues its training on your own examples (your support tickets, your documents, your question and answer pairs) so that it answers the way you need. Bhaskera is for people who want to do that on a shared Slurm cluster without writing the distributed-training code themselves.

On PARAM Rudra it runs from 1 node up to 128 nodes (256 A100 GPUs). Most runs use **LoRA**: the pre-trained model stays as it is and only a small add-on (an *adapter*) is trained, which needs far less GPU memory and far less communication between nodes. **Full fine-tuning**, which trains every weight, is available too.

Official site: <https://github.com/kishoryd/Bhaskera>. If you have never done this before, start with [Start Here: Your First Run](start-here.md).

## What Bhaskera brings to the table

Fine-tuning a large model on a cluster is not one hard step but many small ones, and each can fail in its own way. Bhaskera is the work around the training, as well as the training itself.

| You need to ... | What Bhaskera does |
|---|---|
| Describe a run | One YAML file: the model, the data, how to spread the work over the GPUs. The same file runs on 1 node or 128. |
| Train at any size | LoRA and QLoRA (4-bit) with DDP, FSDP2, HSDP and Pipeline parallelism, measured on PARAM Rudra with five models from 14 B to 235 B parameters, including mixture-of-experts models, on 8 to 128 nodes ([Benchmarks](benchmarks.md)). Full fine-tuning of every weight with DDP, FSDP2 or HSDP, tried on a 14 B model ([Recipes](recipes.md#full-fine-tuning-training-every-weight)). |
| Train models that see and hear | Images, video and audio (Qwen3-VL, Qwen2.5-Omni) with the same recipes and commands. |
| Get your data in | Tokenising from JSONL, JSON, CSV, TSV, Parquet or Arrow; models and datasets named by their Hugging Face address and downloaded for you, even from a login node that has no internet; datasets with names, versions and fingerprints, so every run records exactly what it trained on. |
| Avoid wasting GPU hours | Checks before anything is submitted: files and libraries are there, the model fits in GPU memory, the partition limits allow it, the disk has room for the checkpoints. A node that failed before is skipped. |
| Survive failures | Checkpoints, and `resume` to continue from the last one, or from an earlier one after a loss spike. A failed job ends with a plain explanation of what went wrong and on which node. |
| Know whether it worked | Benchmark the tuned model against its base on standard tests or on your own questions, see what tuning fixed and what it broke, compare models side by side, and chat with the result. |
| Keep track | Every run, its curves, its settings, its adapter and its GPU energy in MLflow, a model registry with names, versions and hand-off notes, and a web page for all of it. |
| Use it on your cluster | Everything that belongs to one cluster is a profile (partitions, GPUs, network), not code. See [Setting Up on Another Cluster](new-cluster.md). |

Two ways to use it: the **command line** (this guide's focus: `./submit.sh`, a few small commands) and a **web page** that does the same things with buttons ([Home Page](homepage.md)). Everything the page does can be done from a terminal.

## Pick your starting point

| Your situation | Go to |
|---|---|
| **I have no model and no data.** I want to see it work, or I will use a public model and dataset. | [Case 1](start-here.md#case-1-i-have-no-model-and-no-dataset) |
| **I bring my own model and my own data**, tokenised or not. | [Case 2](start-here.md#case-2-i-bring-my-own-model-and-my-own-data) |
| **I only want to measure a model I already have**, with no training. | [Case 3](start-here.md#case-3-benchmark-a-model-you-already-have) |

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
| [Start Here: Your First Run](start-here.md) | The words, the idea, and three step-by-step cases: no model and no data, your own model and data, benchmarking a model you have |
| [Why Bhaskera](why.md) | Features, measured benefits, and how it is built |
| [Installation](installation.md) | Loading the module, or installing your own copy |
| [Setting Up on Another Cluster](new-cluster.md) | For an administrator: a first install on a cluster other than PARAM Rudra |
| [End-to-end Example](workflow.md) | Model and data, tokenising, training, expected output, resuming |
| [Recipes](recipes.md) | Ready configurations for each model, full fine-tuning, and how many nodes and hours to request |
| [Downloads and Dataset Versions](data.md) | Data formats, Hugging Face names in a recipe, downloading, naming and versioning data |
| [Monitoring](monitoring.md) | The step log, MLflow, GPU energy, and the Ray Dashboard with charts for every node |
| [Multimodal](multimodal.md) | Fine-tuning with images, video and audio |
| [Model Registry](registry.md) | Naming and versioning trained models, aliases such as champion, using them by name |
| [Resuming a Job](resume.md) | Checkpoints, continuing after a failure, going back after a loss spike |
| [Parallelism Strategies](parallelism.md) | QLoRA, DDP, Pipeline, FSDP and HSDP: which to use |
| [Benchmarks](benchmarks.md) | Measured speed, MFU, memory and accuracy for every model, strategy and node count |
| [Home Page](homepage.md) | The web page: your jobs, runs, data and models, and the same actions with buttons |
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
