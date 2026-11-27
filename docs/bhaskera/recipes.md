---
icon: material/chef-hat
---

# Recipes

## Introduction

A recipe is a ready ExaAI configuration for one model, using the fastest strategy measured on PARAM Rudra.


| Model | QLoRA + DDP | LoRA + DDP | Pipeline | HSDP | FSDP2 |
|---|---|---|---|---|---|
| `Qwen/Qwen3-14B` | `qwen3-14b-qlora` | `qwen3-14b-ddp` | `qwen3-14b-pipeline` | `qwen3-14b-hsdp` | `qwen3-14b-fsdp` |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE) | `qwen3-30b-a3b-qlora` | `qwen3-30b-a3b-ddp` | `qwen3-30b-a3b-pipeline` | `qwen3-30b-a3b-hsdp` | `qwen3-30b-a3b-fsdp` |
| `meta-llama/Llama-3.3-70B-Instruct` | `llama-3.3-70b-qlora` | — | `llama-3.3-70b-pipeline` | `llama-3.3-70b-hsdp` | `llama-3.3-70b-fsdp` |
| `mistralai/Mistral-Large-Instruct-2411` | `mistral-large-123b-qlora` | — | `mistral-large-123b-pipeline` | `mistral-large-123b-hsdp` | `mistral-large-123b-fsdp` |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` (MoE) | — | — | `qwen3-235b-a22b-pipeline` | `qwen3-235b-a22b-hsdp` | `qwen3-235b-a22b-fsdp` |

Each file is `recipes/<name>.yaml`, shown in full under [Full Recipes](#full-recipes). The fastest one for each model is in [Benchmarks](benchmarks.md); in short, QLoRA + DDP for the dense 14 B to 123 B models and Pipeline for the MoE models.
Node counts: any for DDP, QLoRA and FSDP2. Pipeline needs 2 GPUs per node to add up to a whole number of stages (4 stages: a multiple of 2 nodes, 8 stages: a multiple of 4, 16 stages: a multiple of 8). HSDP needs a whole number of shard groups (a group of 4 GPUs: a multiple of 2 nodes, a group of 16: a multiple of 8).

Other recipes:

| Recipe | Model (Hugging Face name) | Strategy | Node count |
|---|---|---|---|
| `param-1-2.9b-ddp.yaml` | `bharatgenai/Param-1-2.9B-Instruct` | DDP, eager attention | any |
| `param-1-7b-moe-fsdp.yaml` | `bharatgenai/Param-1-7B` | FSDP | any |
| `param2-17b-moe-fsdp.yaml` | `bharatgenai/Param2-17B-A2.4B-Thinking` | FSDP | any |

Tokenise recipes (run once per model and dataset, on one CPU node; use the same model and `seq_len` as the training recipe):

| Recipe | For data that is | Command |
|---|---|---|
| `tokenize-chatml.yaml` | Chat: a `messages` column (also ShareGPT, Dolly, Q&A: change `data.format`) | `./submit.sh tokenize recipes/tokenize-chatml.yaml` |
| `tokenize-alpaca.yaml` | Alpaca: `instruction`, optional `input`, `output` | `./submit.sh tokenize recipes/tokenize-alpaca.yaml` |
| `tokenize-prompt-completion.yaml` | A prompt column and a completion column (any names) | `./submit.sh tokenize recipes/tokenize-prompt-completion.yaml` |
| `tokenize-plaintext.yaml` | Plain text files: books, code, articles | `./submit.sh tokenize recipes/tokenize-plaintext.yaml` |
| `tokenize-hf-name.yaml` | A Hugging Face model and dataset, named in the recipe and downloaded first | `./submit.sh tokenize recipes/tokenize-hf-name.yaml` |

Copy one, replace `<project>/<username>` and the model and data paths, and run it. The layouts are in [Data Format](data.md).

!!! note "How the recipes relate to the benchmark"
    The speed and memory in each header are the measured benchmark numbers for that model and method. The recipes were written
    to the same settings (LoRA rank 16, sequence length 4096, the shard group sizes above) and every one loads cleanly in
    ExaAI's configuration check, but a recipe is a starting point: run `DRY_RUN=1 ./submit.sh recipes/<name>.yaml <nodes> 1`
    first, which checks the model, data, GPU memory estimate and node count without submitting.

A pipeline recipe splits the model into *stages* on different GPUs, so the number of GPUs (2 per node) must be
a multiple of the number of stages. `submit.sh` checks this before it submits.


## Full Recipes

Every recipe in full, for reference. Open one, copy it to `recipes/<name>.yaml` in your run folder (they are already there when you copy the kit), replace `<project>/<username>` and the two tokenised data paths, and run it. The comment lines at the top give the measured speed.

!!! tip "Reading a recipe"
    - `model`: the model folder, precision, attention and kernel choices, and `quantization: qlora` for 4-bit.
    - `data`: the tokenised caches (`tokenized_path`, `val_tokenized_path`), `seq_len` and `runtime_packing`.
    - `lora`: the adapter (`r`, `alpha`, which modules). `moe`: router settings for mixture-of-experts models.
    - `training`: batch size, learning rate, `max_steps`, and `distributed.strategy` (`ddp`, `fsdp` or `pipeline`, with `hsdp_shard_size` for HSDP and `stages` / `microbatches` for pipeline).
    - `checkpoint`, `logging`, `monitoring`: where checkpoints go and how often, the MLflow run name, and the metrics recorded.

### Qwen3-14B

??? example "qwen3-14b-qlora  (QLoRA + DDP)"
    ```yaml title="recipes/qwen3-14b-qlora.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-14b-qlora.yaml"
    ```

??? example "qwen3-14b-ddp  (LoRA + DDP)"
    ```yaml title="recipes/qwen3-14b-ddp.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-14b-ddp.yaml"
    ```

??? example "qwen3-14b-pipeline  (Pipeline)"
    ```yaml title="recipes/qwen3-14b-pipeline.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-14b-pipeline.yaml"
    ```

??? example "qwen3-14b-hsdp  (HSDP)"
    ```yaml title="recipes/qwen3-14b-hsdp.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-14b-hsdp.yaml"
    ```

??? example "qwen3-14b-fsdp  (FSDP2)"
    ```yaml title="recipes/qwen3-14b-fsdp.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-14b-fsdp.yaml"
    ```

### Qwen3-30B-A3B (MoE)

??? example "qwen3-30b-a3b-qlora  (QLoRA + DDP)"
    ```yaml title="recipes/qwen3-30b-a3b-qlora.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-30b-a3b-qlora.yaml"
    ```

??? example "qwen3-30b-a3b-ddp  (LoRA + DDP)"
    ```yaml title="recipes/qwen3-30b-a3b-ddp.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-30b-a3b-ddp.yaml"
    ```

??? example "qwen3-30b-a3b-pipeline  (Pipeline)"
    ```yaml title="recipes/qwen3-30b-a3b-pipeline.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-30b-a3b-pipeline.yaml"
    ```

??? example "qwen3-30b-a3b-hsdp  (HSDP)"
    ```yaml title="recipes/qwen3-30b-a3b-hsdp.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-30b-a3b-hsdp.yaml"
    ```

??? example "qwen3-30b-a3b-fsdp  (FSDP2)"
    ```yaml title="recipes/qwen3-30b-a3b-fsdp.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-30b-a3b-fsdp.yaml"
    ```

### Llama-3.3-70B

??? example "llama-3.3-70b-qlora  (QLoRA + DDP)"
    ```yaml title="recipes/llama-3.3-70b-qlora.yaml"
    --8<-- "docs/bhaskera/files/recipes/llama-3.3-70b-qlora.yaml"
    ```

??? example "llama-3.3-70b-pipeline  (Pipeline)"
    ```yaml title="recipes/llama-3.3-70b-pipeline.yaml"
    --8<-- "docs/bhaskera/files/recipes/llama-3.3-70b-pipeline.yaml"
    ```

??? example "llama-3.3-70b-hsdp  (HSDP)"
    ```yaml title="recipes/llama-3.3-70b-hsdp.yaml"
    --8<-- "docs/bhaskera/files/recipes/llama-3.3-70b-hsdp.yaml"
    ```

??? example "llama-3.3-70b-fsdp  (FSDP2)"
    ```yaml title="recipes/llama-3.3-70b-fsdp.yaml"
    --8<-- "docs/bhaskera/files/recipes/llama-3.3-70b-fsdp.yaml"
    ```

### Mistral-Large-123B

??? example "mistral-large-123b-qlora  (QLoRA + DDP)"
    ```yaml title="recipes/mistral-large-123b-qlora.yaml"
    --8<-- "docs/bhaskera/files/recipes/mistral-large-123b-qlora.yaml"
    ```

??? example "mistral-large-123b-pipeline  (Pipeline)"
    ```yaml title="recipes/mistral-large-123b-pipeline.yaml"
    --8<-- "docs/bhaskera/files/recipes/mistral-large-123b-pipeline.yaml"
    ```

??? example "mistral-large-123b-hsdp  (HSDP)"
    ```yaml title="recipes/mistral-large-123b-hsdp.yaml"
    --8<-- "docs/bhaskera/files/recipes/mistral-large-123b-hsdp.yaml"
    ```

??? example "mistral-large-123b-fsdp  (FSDP2)"
    ```yaml title="recipes/mistral-large-123b-fsdp.yaml"
    --8<-- "docs/bhaskera/files/recipes/mistral-large-123b-fsdp.yaml"
    ```

### Qwen3-235B-A22B (MoE)

??? example "qwen3-235b-a22b-pipeline  (Pipeline)"
    ```yaml title="recipes/qwen3-235b-a22b-pipeline.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-235b-a22b-pipeline.yaml"
    ```

??? example "qwen3-235b-a22b-hsdp  (HSDP)"
    ```yaml title="recipes/qwen3-235b-a22b-hsdp.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-235b-a22b-hsdp.yaml"
    ```

??? example "qwen3-235b-a22b-fsdp  (FSDP2)"
    ```yaml title="recipes/qwen3-235b-a22b-fsdp.yaml"
    --8<-- "docs/bhaskera/files/recipes/qwen3-235b-a22b-fsdp.yaml"
    ```

### Tokenise recipes

??? example "tokenize-chatml  (Chat data)"
    ```yaml title="recipes/tokenize-chatml.yaml"
    --8<-- "docs/bhaskera/files/recipes/tokenize-chatml.yaml"
    ```

??? example "tokenize-alpaca  (Alpaca data)"
    ```yaml title="recipes/tokenize-alpaca.yaml"
    --8<-- "docs/bhaskera/files/recipes/tokenize-alpaca.yaml"
    ```

??? example "tokenize-prompt-completion  (Prompt and completion)"
    ```yaml title="recipes/tokenize-prompt-completion.yaml"
    --8<-- "docs/bhaskera/files/recipes/tokenize-prompt-completion.yaml"
    ```

??? example "tokenize-plaintext  (Plain text)"
    ```yaml title="recipes/tokenize-plaintext.yaml"
    --8<-- "docs/bhaskera/files/recipes/tokenize-plaintext.yaml"
    ```

??? example "tokenize-hf-name  (Hugging Face names)"
    ```yaml title="recipes/tokenize-hf-name.yaml"
    --8<-- "docs/bhaskera/files/recipes/tokenize-hf-name.yaml"
    ```

### Multimodal models

The image, video and audio starters are in `configs/multimodal/` of your run folder (copied from the kit with `recipes/`). A run folder made earlier lacks them: `cp -r $BHASKERA_KIT/configs .`. Tokenise first with a `tokenize_*` config, which prints the two cache paths to paste in, then run the training config as its first comment lines show. See [Multimodal](multimodal.md) for the data and chat.

??? example "multimodal/tokenize_vision  (Tokenise images and video)"
    ```yaml title="configs/multimodal/tokenize_vision.yaml"
    --8<-- "docs/bhaskera/files/configs/multimodal/tokenize_vision.yaml"
    ```

??? example "multimodal/tokenize_audio  (Tokenise audio)"
    ```yaml title="configs/multimodal/tokenize_audio.yaml"
    --8<-- "docs/bhaskera/files/configs/multimodal/tokenize_audio.yaml"
    ```

??? example "multimodal/ddp  (Images, DDP)"
    ```yaml title="configs/multimodal/ddp.yaml"
    --8<-- "docs/bhaskera/files/configs/multimodal/ddp.yaml"
    ```

??? example "multimodal/fsdp  (Images, FSDP2)"
    ```yaml title="configs/multimodal/fsdp.yaml"
    --8<-- "docs/bhaskera/files/configs/multimodal/fsdp.yaml"
    ```

??? example "multimodal/hsdp  (Images, HSDP)"
    ```yaml title="configs/multimodal/hsdp.yaml"
    --8<-- "docs/bhaskera/files/configs/multimodal/hsdp.yaml"
    ```

??? example "multimodal/video  (Video)"
    ```yaml title="configs/multimodal/video.yaml"
    --8<-- "docs/bhaskera/files/configs/multimodal/video.yaml"
    ```

??? example "multimodal/audio  (Audio)"
    ```yaml title="configs/multimodal/audio.yaml"
    --8<-- "docs/bhaskera/files/configs/multimodal/audio.yaml"
    ```

### Other models

??? example "param-1-2.9b-ddp  (Param-1-2.9B, DDP)"
    ```yaml title="recipes/param-1-2.9b-ddp.yaml"
    --8<-- "docs/bhaskera/files/recipes/param-1-2.9b-ddp.yaml"
    ```

??? example "param-1-7b-moe-fsdp  (Param-1-7B, FSDP)"
    ```yaml title="recipes/param-1-7b-moe-fsdp.yaml"
    --8<-- "docs/bhaskera/files/recipes/param-1-7b-moe-fsdp.yaml"
    ```

??? example "param2-17b-moe-fsdp  (Param2-17B, FSDP)"
    ```yaml title="recipes/param2-17b-moe-fsdp.yaml"
    --8<-- "docs/bhaskera/files/recipes/param2-17b-moe-fsdp.yaml"
    ```

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
ExaAI repository, replace the `<project>/<username>` paths, and run it like a recipe. There is one starter per parallelism
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
