---
icon: material/chef-hat
---

# Recipes

## Introduction

A recipe is a ready Bhaskera configuration for one model, using the fastest strategy measured on
PARAM Rudra. The recipes are in `recipes/` of your run folder (copied from `$BHASKERA_KIT`). Each file
lists its measured speed and memory in its header.

| Recipe | Model (Hugging Face name) | Strategy | Node count |
|---|---|---|---|
| `qwen3-14b-qlora.yaml` | `Qwen/Qwen3-14B` | QLoRA + DDP | any |
| `qwen3-14b-ddp.yaml` | `Qwen/Qwen3-14B` | DDP | any |
| `qwen3-30b-a3b-pipeline.yaml` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Pipeline, 4 stages | multiple of 2 |
| `llama-3.3-70b-qlora.yaml` | `meta-llama/Llama-3.3-70B-Instruct` | QLoRA + DDP | any |
| `llama-3.3-70b-pipeline.yaml` | `meta-llama/Llama-3.3-70B-Instruct` | Pipeline, 4 stages | multiple of 2 |
| `mistral-large-123b-qlora.yaml` | `mistralai/Mistral-Large-Instruct-2411` | QLoRA + DDP (batch 1) | any |
| `mistral-large-123b-pipeline.yaml` | `mistralai/Mistral-Large-Instruct-2411` | Pipeline, 8 stages | multiple of 4 |
| `qwen3-235b-a22b-pipeline.yaml` | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Pipeline, 16 stages | multiple of 8 |
| `tokenize-chatml.yaml` | (the model you train) | tokenising | 1 CPU node |

A pipeline recipe splits the model into *stages* on different GPUs, so the number of GPUs (2 per node)
must be a multiple of the number of stages. `submit.sh` checks this before submitting.

## Editing a Recipe

Each recipe needs only your paths. Replace every `<project>/<username>` and set the two cache paths
printed by the tokenize job:

```yaml
model:
  name: /scratch/<project>/<username>/models/Qwen3-14B
data:
  tokenized_path: /scratch/<project>/<username>/cache/local_train_<hash>
  val_tokenized_path: /scratch/<project>/<username>/cache/local_val_<hash>
training:
  max_steps: 10            # smoke test; see "Full runs" below
  eval_steps: 200          # evaluate every 200 steps
checkpoint:
  save_dir: /scratch/<project>/<username>/bhaskera-runs/checkpoints/qwen3-14b-qlora
```

Do not change the `training.distributed` block, `batch_size` or `grad_accum` unless you know why: these
are the settings that were measured. In particular, `mistral-large-123b-qlora.yaml` uses
`batch_size: 1` because batch 2 does not fit in 80 GB.

## Full Runs

The recipes run 10 steps so that a first job finishes in minutes and shows the speed. For a full run,
read `tok/s` and `step_s` from a `[step ...]` line of that smoke test and set:

```text
max_steps = (tokens in your training set) / (tok/s × step_s)
```

The tokens in your training set are printed by the tokenize job.

## Choosing Nodes and Hours

**Checkpoints differ by strategy.** With **QLoRA + DDP** and **DDP**, a checkpoint is written only when
the run ends (`checkpoint.save_interval` counts epochs), so the whole run must fit in one job. With
**Pipeline**, a checkpoint is written every `save_interval` steps (500 in the recipes), and a resubmitted
job continues from it, so a long run can span several jobs.

Measured time for **one epoch of a 1.24-billion-token dataset** (scale it by your own token count), and
whether it fits in a single job within the partition's walltime:

| Recipe | 8 nodes | 16 nodes | 32 nodes | 64 nodes | 128 nodes |
|---|---|---|---|---|---|
| Qwen3-14B QLoRA | 12.5 h ✅ | 6.3 h ✅ | 3.2 h ✅ | 1.8 h ✅ | 49 min ✅ |
| Qwen3-30B-A3B Pipeline | 12.3 h ✅ | 6.1 h ✅ | 3.1 h ✅ | 1.6 h ✅ | 53 min ✅ |
| Llama-3.3-70B QLoRA | 2.2 days ❌ | 26.5 h ❌ | 13.3 h ❌ | 6.6 h ✅ | 3.3 h ✅ |
| Llama-3.3-70B Pipeline | 2.8 days ↻ | 33.7 h ↻ | 16.9 h ↻ | 8.4 h ✅ | 4.6 h ✅ |
| Mistral-Large-123B QLoRA | 4.1 days ❌ | 2.1 days ❌ | 24.9 h ❌ | 12.4 h ❌ | 6.4 h ❌ |
| Mistral-Large-123B Pipeline | 5.2 days ↻ | 2.6 days ↻ | 31.3 h ↻ | 15.5 h ↻ | 7.8 h ↻ |
| Qwen3-235B-A22B Pipeline | 2.6 days ↻ | 31.0 h ↻ | 17.8 h ↻ | 7.8 h ✅ | 4.3 h ✅ |

✅ fits in one job · ↻ longer than one job, continues from its checkpoint when resubmitted ·
❌ longer than one job and cannot resume mid-run: use more nodes, the Pipeline recipe, or a smaller
`max_steps`.

Walltime limits: `gpu-small` (5–25 nodes) 24 h, `gpu-large` (26–100 nodes) 12 h, `gpu-massive`
(101–200 nodes) 6 h.
