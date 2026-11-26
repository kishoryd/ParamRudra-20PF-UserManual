---
icon: material/chef-hat
---

# Recipes

## Introduction

A recipe is a ready Bhaskera configuration for one model, using the fastest strategy measured on PARAM Rudra.
The recipes are in the `recipes/` folder of your run folder (copied from `$BHASKERA_KIT`). Each file lists its
speed and memory use in the header.

To make your own, press **New run** on the home page and choose **Create recipe**: it asks for the model, the data and how long, supports text and multimodal
models with every strategy, and writes the file for you (see [The Home Page](homepage.md#create-a-recipe)).

| Recipe | Model (Hugging Face name) | Strategy | Node count |
|---|---|---|---|
| `qwen3-14b-qlora.yaml` | `Qwen/Qwen3-14B` | QLoRA + DDP | any |
| `qwen3-14b-ddp.yaml` | `Qwen/Qwen3-14B` | DDP | any |
| `qwen3-30b-a3b-pipeline.yaml` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Pipeline, 4 stages | multiple of 2 |
| `llama-3.3-70b-qlora.yaml` | `meta-llama/Llama-3.3-70B-Instruct` | QLoRA + DDP | any |
| `llama-3.3-70b-pipeline.yaml` | `meta-llama/Llama-3.3-70B-Instruct` | Pipeline, 4 stages | multiple of 2 |
| `mistral-large-123b-qlora.yaml` | `mistralai/Mistral-Large-Instruct-2411` | QLoRA + DDP, batch size 1 | any |
| `mistral-large-123b-pipeline.yaml` | `mistralai/Mistral-Large-Instruct-2411` | Pipeline, 8 stages | multiple of 4 |
| `qwen3-235b-a22b-pipeline.yaml` | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Pipeline, 16 stages | multiple of 8 |
| `param-1-2.9b-ddp.yaml` | `bharatgenai/Param-1-2.9B-Instruct` | DDP, eager attention | any |
| `param-1-7b-moe-fsdp.yaml` | `bharatgenai/Param-1-7B` | FSDP | any |
| `param2-17b-moe-fsdp.yaml` | `bharatgenai/Param2-17B-A2.4B-Thinking` | FSDP | any |
| `tokenize-chatml.yaml` | the model you train | tokenising | 1 CPU node |

A pipeline recipe splits the model into *stages* on different GPUs, so the number of GPUs (2 per node) must be
a multiple of the number of stages. `submit.sh` checks this before it submits.

## BharatGen Models

The three BharatGen recipes use the models' own code (`trust_remote_code: true`), so the model folder must
contain the Python files that come with the download.

| Model | Type | Speed on 5 nodes (10 GPUs) | Memory per GPU |
|---|---|---|---|
| Param-1-2.9B-Instruct | dense, 2.9 B | 1.6 s per step, 11% MFU | 24 GiB |
| Param-1-7B | MoE, 6.7 B (64 experts, top 8) | 10.1 s per step, 0.9% MFU | 12 GiB |
| Param2-17B-A2.4B-Thinking | MoE, 17 B (64 experts, top 6) | 9.4 s per step, 1.6% MFU | 17 GiB |

Param-1-2.9B supports neither flash-attention nor PyTorch attention, so its recipe uses `attn_impl: eager`
and no sequence packing. The two MoE models run under FSDP, where experts are computed one at a time and MFU
stays low; Pipeline does not support their custom code.

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

In the web page this is **What to train: the whole model**, which sets all of this for you. What it costs and what to expect:

- **GPU memory.** Every weight, its gradient and the optimizer state live on the GPUs, several times more than LoRA. Use FSDP2 or HSDP over several nodes; plain DDP only for a model that fits on one GPU.
- **Not with QLoRA or Pipeline.** QLoRA trains an add-on on a 4-bit model, and Pipeline trains a LoRA add-on only; both are refused for a full fine-tune.
- **Disk.** A checkpoint holds every weight and the optimizer state: about five times the size of the model (an estimate; around 140 GB for a 14 B model), against about 1 GB for a LoRA run (measured: 449 MB of model and 493 MB of optimizer state for Qwen3-14B). Use `keep_last_n` to keep few, and a large disk.
- **No adapter.** The result is the checkpoint, not an adapter: Chat, Benchmark and Register on the home page, which need an adapter, are not offered for it yet.
- Measured on PARAM Rudra, Qwen3-14B on 10 GPUs, 60 steps, batch 2 × 4096 per GPU: FSDP2 took 24 s per step and HSDP 15 s per step (LoRA with DDP: 5 s per step). One run each, so treat the numbers as a rough guide.

## Editing a Recipe

A recipe needs only your paths. Replace every `<project>/<username>` and set the two cache paths printed by
the tokenize job:

```yaml
model:
  name: /scratch/<project>/<username>/models/Qwen3-14B
data:
  tokenized_path: /scratch/<project>/<username>/cache/local_train_<hash>
  val_tokenized_path: /scratch/<project>/<username>/cache/local_val_<hash>
training:
  max_steps: 10            # a short first run; see "Full Runs" below
  eval_steps: 200          # evaluate every 200 steps
checkpoint:
  save_dir: /scratch/<project>/<username>/bhaskera-runs/checkpoints/qwen3-14b-qlora
```

Leave the `training.distributed` block, `batch_size` and `grad_accum` as they are unless you know why you
are changing them. In particular, `mistral-large-123b-qlora.yaml` uses `batch_size: 1` because batch size 2
does not fit in 80 GB.

## Full Runs

The recipes run 10 steps so that a first job finishes in minutes and shows the speed. For a full run, read
`tok/s` and `step_s` from a `[step ...]` line of that first job and set:

```text
max_steps = (tokens in your training set) / (tok/s × step_s)
```

The tokenize job prints the number of tokens in your training set.

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

The partition time limits are 24 h on `gpu-small` (5–25 nodes), 12 h on `gpu-large` (26–100 nodes) and 6 h on
`gpu-massive` (101–200 nodes). An epoch fits in one job for Qwen3-14B QLoRA and Qwen3-30B-A3B Pipeline at
every node count, for Llama-3.3-70B (QLoRA and Pipeline) and Qwen3-235B-A22B (Pipeline) from 64 nodes, and for
no node count with Mistral-Large-123B. Everything else takes several jobs.

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
