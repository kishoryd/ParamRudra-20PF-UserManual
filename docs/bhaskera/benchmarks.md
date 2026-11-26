---
icon: material/speedometer
---

# Benchmarks on PARAM Rudra

## Introduction


All combinations of **5 models × 5 strategies × 5 node counts (8, 16, 32, 64, 128)** were run on
PARAM Rudra in November 2026:

!!! info "How these numbers were measured"
    - Each run trains for 10 steps with LoRA, then evaluates once on about 256 held-out
      conversations. Sequence length 4096, 2 × A100 80 GB per node.
    - **Time to train** is projected for **one pass over a 1.24-billion-token training set**
      from the measured speed over steps 3–10; startup (5–20 min) is excluded.
    - **MFU** is the share of the GPUs' peak bf16 maths speed spent on model work. For MoE
      models it counts only the active parameters.
    - Runs hit by a faulty GPU, node or network were rerun automatically; the tables show the
      successful attempt.
    - \* At least one step of this run took more than 1.5× its median step time (a slow node or
      network hiccup). Slow steps are included in the numbers.
    - The training data was a 528,450-conversation ChatML instruction dataset (1.24 B real tokens);
      evaluation used a separate 3,000-conversation set.

## Fastest setup per model (128 nodes, 256 GPUs)

| Model | Strategy | Tokens / s | MFU | Time to train | Eval accuracy |
|---|---|---|---|---|---|
| Qwen3-14B | QLoRA + DDP | 424,899 | 34.4% | 49 min | 65.5% |
| Qwen3-30B-A3B (MoE) | Pipeline | 386,473 | 7.7% | 53 min | 67.1% |
| Llama-3.3-70B | QLoRA + DDP | 103,481 | 38.8% | 3.3 h | 62.8% |
| Mistral-Large-123B | QLoRA + DDP | 57,457 | 37.3% | 6.4 h | 71.9% |
| Qwen3-235B-A22B (MoE) | Pipeline | 80,110\* | 9.9% | 4.3 h | 68.0% |

## Scaling with node count

Each cell is **time to train one epoch (MFU)**. "Out of memory" means the model does not fit
that strategy on 80 GB GPUs.

### Qwen3-14B

| Strategy | 8 nodes | 16 nodes | 32 nodes | 64 nodes | 128 nodes | Peak GPU memory |
|---|---|---|---|---|---|---|
| DDP | 14.1 h (31.6%) | 7.2 h (31.1%) | 3.6 h (30.9%) | 1.8 h (30.4%) | 54 min (31.1%) | 32 / 80 GB |
| QLoRA + DDP | 12.5 h (35.5%) | 6.3 h (35.4%) | 3.2 h (35.2%) | 1.8 h (30.9%)\* | 49 min (34.4%) | 16 / 80 GB |
| Pipeline | 19.1 h (23.2%) | 9.6 h (23.2%) | 4.8 h (23.3%) | 2.4 h (23.1%) | 1.2 h (22.9%) | 16 / 80 GB |
| HSDP | 34.8 h (12.8%) | 18.1 h (12.3%) | 18.6 h (6.0%)\* | 4.5 h (12.2%) | 3.7 h (7.6%) | 24 / 80 GB |
| FSDP | 37.7 h (11.8%) | 20.3 h (11.0%) | 14.3 h (7.8%) | 8.8 h (6.8%) | 5.9 h (4.7%) | 10 / 80 GB |

### Qwen3-30B-A3B (MoE)

| Strategy | 8 nodes | 16 nodes | 32 nodes | 64 nodes | 128 nodes | Peak GPU memory |
|---|---|---|---|---|---|---|
| DDP | 3.1 days (1.5%) | 37.8 h (1.4%) | 20.4 h (1.4%) | 10.2 h (1.3%) | 5.2 h (1.3%) | 61 / 80 GB |
| QLoRA + DDP | 2.7 days (1.7%) | 33.0 h (1.7%) | 17.0 h (1.6%) | 10.9 h (1.3%)\* | 4.3 h (1.6%) | 23 / 80 GB |
| Pipeline | 12.3 h (8.9%) | 6.1 h (9.0%) | 3.1 h (8.8%) | 1.6 h (8.7%) | 53 min (7.7%) | 25 / 80 GB |
| HSDP | 6.7 days (0.7%) | 3.7 days (0.6%) | 2.1 days (0.5%) | 29.4 h (0.4%) | 18.6 h (0.4%) | 38 / 80 GB |
| FSDP | 7.7 days (0.6%) | 6.3 days (0.4%)\* | 3.3 days (0.4%) | 2.0 days (0.3%) | 33.4 h (0.2%) | 10 / 80 GB |

### Llama-3.3-70B

| Strategy | 8 nodes | 16 nodes | 32 nodes | 64 nodes | 128 nodes | Peak GPU memory |
|---|---|---|---|---|---|---|
| DDP | out of memory | out of memory | out of memory | out of memory | out of memory | — |
| QLoRA + DDP | 2.2 days (39.1%) | 26.5 h (39.0%) | 13.3 h (38.9%) | 6.6 h (38.9%) | 3.3 h (38.8%) | 55 / 80 GB |
| Pipeline | 2.8 days (31.0%) | 33.7 h (30.6%) | 16.9 h (30.5%) | 8.4 h (30.5%) | 4.6 h (28.1%)\* | 44 / 80 GB |
| HSDP | 7.1 days (12.1%) | 3.6 days (11.9%) | 43.6 h (11.9%) | 22.0 h (11.8%) | 18.3 h (7.0%) | 51 / 80 GB |
| FSDP | 7.7 days (11.5%) | 3.8 days (11.2%) | 46.6 h (11.0%) | 25.7 h (10.7%) | 20.8 h (6.2%) | 19 / 80 GB |

### Mistral-Large-123B

| Strategy | 8 nodes | 16 nodes | 32 nodes | 64 nodes | 128 nodes | Peak GPU memory |
|---|---|---|---|---|---|---|
| DDP | out of memory | out of memory | out of memory | out of memory | out of memory | — |
| QLoRA + DDP | 4.1 days (38.8%) | 2.1 days (38.6%) | 24.9 h (38.6%) | 12.4 h (38.5%) | 6.4 h (37.3%) | 78 / 80 GB |
| Pipeline | 5.2 days (30.8%) | 2.6 days (30.9%) | 31.3 h (30.5%) | 15.5 h (30.8%) | 7.8 h (30.7%) | 50 / 80 GB |
| HSDP | 13.4 days (11.8%) | 6.8 days (11.8%) | 3.4 days (11.8%) | 41.0 h (11.6%) | 35.2 h (6.8%) | 50 / 80 GB |
| FSDP | 14.2 days (11.3%) | 7.0 days (11.5%) | 3.5 days (11.2%) | 44.4 h (10.8%) | 37.9 h (6.3%) | 23 / 80 GB |

### Qwen3-235B-A22B (MoE)

| Strategy | 8 nodes | 16 nodes | 32 nodes | 64 nodes | 128 nodes | Peak GPU memory |
|---|---|---|---|---|---|---|
| DDP | out of memory | out of memory | out of memory | out of memory | out of memory | — |
| QLoRA + DDP | out of memory | out of memory | out of memory | out of memory | out of memory | — |
| Pipeline | 2.6 days (10.9%) | 31.0 h (11.0%) | 17.8 h (9.5%) | 7.8 h (10.8%) | 4.3 h (9.9%)\* | 45 / 80 GB |
| HSDP | 30.4 days (0.9%) | 16.0 days (0.9%) | 9.7 days (0.7%)\* | 6.6 days (0.6%) | 3.6 days (0.5%) | 54 / 80 GB |
| FSDP | 30.7 days (0.9%) | 17.1 days (0.8%) | 12.4 days (0.6%) | 10.0 days (0.4%) | 5.3 days (0.3%) | 29 / 80 GB |

## What the results show

- **QLoRA, DDP and Pipeline scale almost linearly**: doubling the nodes nearly halves the time,
  and MFU stays flat from 8 to 128 nodes. Qwen3-14B with QLoRA goes from 12.5 h on 8 nodes to
  49 min on 128 nodes.
- **For dense 70 B–123 B models, QLoRA + DDP is the fastest** (37–39% MFU), followed by
  Pipeline (28–31%).
- **FSDP and HSDP are 3–4× slower and lose efficiency at 128 nodes** (MFU 5–7%), because they
  move model weights over the network every step.
- **MoE models need the Pipeline strategy.** With other strategies they run at 0.2–1.7% MFU.
- **Accuracy agrees across strategies.** Within each model, eval accuracy differs by about one
  point or less between strategies and node counts, so the choice of strategy is about speed
  and memory, not quality.

