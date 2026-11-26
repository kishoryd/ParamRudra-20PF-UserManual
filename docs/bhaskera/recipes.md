---
icon: material/chef-hat
---

# Recipes

## Introduction

A recipe is a ready Bhaskera configuration for one model, using the fastest strategy measured on PARAM Rudra.


| Model | QLoRA + DDP | LoRA + DDP | Pipeline | HSDP | FSDP2 |
|---|---|---|---|---|---|
| `Qwen/Qwen3-14B` | `qwen3-14b-qlora` | `qwen3-14b-ddp` | `qwen3-14b-pipeline` | `qwen3-14b-hsdp` | `qwen3-14b-fsdp` |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE) | `qwen3-30b-a3b-qlora` | `qwen3-30b-a3b-ddp` | `qwen3-30b-a3b-pipeline` | `qwen3-30b-a3b-hsdp` | `qwen3-30b-a3b-fsdp` |
| `meta-llama/Llama-3.3-70B-Instruct` | `llama-3.3-70b-qlora` | — | `llama-3.3-70b-pipeline` | `llama-3.3-70b-hsdp` | `llama-3.3-70b-fsdp` |
| `mistralai/Mistral-Large-Instruct-2411` | `mistral-large-123b-qlora` | — | `mistral-large-123b-pipeline` | `mistral-large-123b-hsdp` | `mistral-large-123b-fsdp` |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` (MoE) | — | — | `qwen3-235b-a22b-pipeline` | `qwen3-235b-a22b-hsdp` | `qwen3-235b-a22b-fsdp` |

Each file is `recipes/<name>.yaml`. The fastest one for each model is in [Benchmarks](benchmarks.md); in short, QLoRA + DDP for the dense 14 B to 123 B models and Pipeline for the MoE models.
Node counts: any for DDP, QLoRA and FSDP2. Pipeline needs 2 GPUs per node to add up to a whole number of stages (4 stages: a multiple of 2 nodes, 8 stages: a multiple of 4, 16 stages: a multiple of 8). HSDP needs a whole number of shard groups (a group of 4 GPUs: a multiple of 2 nodes, a group of 16: a multiple of 8).

Other recipes:

| Recipe | Model (Hugging Face name) | Strategy | Node count |
|---|---|---|---|
| `param-1-2.9b-ddp.yaml` | `bharatgenai/Param-1-2.9B-Instruct` | DDP, eager attention | any |
| `param-1-7b-moe-fsdp.yaml` | `bharatgenai/Param-1-7B` | FSDP | any |
| `param2-17b-moe-fsdp.yaml` | `bharatgenai/Param2-17B-A2.4B-Thinking` | FSDP | any |
| `tokenize-chatml.yaml` | the model you train | tokenising | 1 CPU node |

!!! note "How the recipes relate to the benchmark"
    The speed and memory in each header are the measured benchmark numbers for that model and method. The recipes were written
    to the same settings (LoRA rank 16, sequence length 4096, the shard group sizes above) and every one loads cleanly in
    Bhaskera's configuration check, but a recipe is a starting point: run `DRY_RUN=1 ./submit.sh recipes/<name>.yaml <nodes> 1`
    first, which checks the model, data, GPU memory estimate and node count without submitting.

A pipeline recipe splits the model into *stages* on different GPUs, so the number of GPUs (2 per node) must be
a multiple of the number of stages. `submit.sh` checks this before it submits.


## Full Fine-tuning (Training Every Weight)

Every recipe above trains a **LoRA** add-on: the model stays as it is and a small adapter is trained, which is what makes 14 B to 235 B models practical. To train **every weight** instead, set LoRA off and use a method that spreads the model over the GPUs:

```yaml
lora:
  enabled: false
training:
  lr: 1.0e-5                      # much lower than the 1e-4 of LoRA
  distributed:
    strategy: fsdp                # fsdp (FSDP2), or ddp for a model that fits on one GPU; hsdp also works
checkpoint:
  enabled: true                   # the result lives in the checkpoints, so keep this on
logging:
  mlflow_log_adapter: false       # there is no adapter to keep
```

## Choosing Nodes and Hours

**Checkpoints.** A checkpoint holds the trained LoRA weights and the optimizer state. With **QLoRA + DDP**,
**DDP** and **FSDP** recipes it is written when the run ends, either at `max_steps` or at the end of an epoch
(`checkpoint.save_interval` counts epochs). With **Pipeline** recipes it is written every `save_interval`
steps (500 in the recipes). Resubmitting a job with the same `checkpoint.save_dir` continues from the latest
checkpoint, and the learning-rate schedule continues with it.

So a QLoRA, DDP or FSDP run that is longer than the partition's time limit has to be split by hand: set
`max_steps` so that each job ends before the limit, and resubmit with a larger `max_steps` to continue.

Time for **one epoch of a 1.24-billion-token dataset** (scale it by your own token count):

| Recipe | 8 nodes | 16 nodes | 32 nodes | 64 nodes | 128 nodes |
|---|---|---|---|---|---|
| Qwen3-14B QLoRA | 12.5 h | 6.3 h | 3.2 h | 1.8 h | 49 min |
| Qwen3-30B-A3B Pipeline | 12.3 h | 6.1 h | 3.1 h | 1.6 h | 53 min |
| Llama-3.3-70B QLoRA | 2.2 days | 26.5 h | 13.3 h | 6.6 h | 3.3 h |
| Llama-3.3-70B Pipeline | 2.8 days | 33.7 h | 16.9 h | 8.4 h | 4.6 h |
| Mistral-Large-123B QLoRA | 4.1 days | 2.1 days | 24.9 h | 12.4 h | 6.4 h |
| Mistral-Large-123B Pipeline | 5.2 days | 2.6 days | 31.3 h | 15.5 h | 7.8 h |
| Qwen3-235B-A22B Pipeline | 2.6 days | 31.0 h | 17.8 h | 7.8 h | 4.3 h |

## Starter Configs for Other Models

The recipes above are tested for the models in the table. For another model, copy a starter from `configs/` in the
Bhaskera repository, replace the `<project>/<username>` paths, and run it like a recipe. There is one starter per parallelism
for text and for multimodal models, and a tokenizer config for each model type:

| | Text | Multimodal |
|---|---|---|
| DDP | `configs/text/ddp.yaml` | `configs/multimodal/ddp.yaml` |
| FSDP2 | `configs/text/fsdp.yaml` | `configs/multimodal/fsdp.yaml` |
| HSDP | `configs/text/hsdp.yaml` | `configs/multimodal/hsdp.yaml` |
| Pipeline | `configs/text/pipeline.yaml` | not available |
| Tokenise first | `configs/text/tokenize.yaml` | `tokenize_vision.yaml` (image, video), `tokenize_audio.yaml` |

`configs/config.yaml` lists every option with its default. A key that does nothing, such as a typo, is reported with the
closest real name when the config is loaded.
