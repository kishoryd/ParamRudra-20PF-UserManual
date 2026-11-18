# Parallelism Strategies

Set `training.distributed.strategy` in the config. HSDP is `fsdp` with
`fsdp.sharding_strategy: HYBRID_SHARD`; QLoRA is `ddp` with `model.quantization: qlora`.

| Strategy | In plain words | Use it for | Avoid when |
|---|---|---|---|
| **DDP** | Every GPU holds the whole model; only the small LoRA updates cross the network | Models that fit one GPU in bf16 (≈ 14 B and below) | The model is larger than 80 GB |
| **QLoRA + DDP** | The frozen model is stored in 4-bit so it fits one GPU | 70 B–123 B dense models: fastest at scale | MoE models; Qwen3-235B does not fit |
| **Pipeline** | Model layers split into stages on different GPUs, like an assembly line | Very large models and **all MoE models** | Very small models |
| **FSDP** | The model is cut into pieces over all GPUs and fetched every step | Only when nothing else fits | Large node counts on PARAM Rudra |
| **HSDP** | FSDP inside small groups of GPUs | As FSDP; keep a group within one node | Groups spanning nodes (no gain here) |

=== "DDP"

    ```yaml
    training:
      distributed:
        strategy: ddp
    ```

=== "QLoRA + DDP"

    ```yaml
    model:
      quantization: qlora
    training:
      distributed:
        strategy: ddp
    ```

=== "Pipeline"

    ```yaml
    training:
      distributed:
        strategy: pipeline
        pipeline:
          stages: 4              # GPUs per model copy; must divide the GPU count
          microbatches: 16       # more micro-batches = less idle time
          schedule: 1f1b         # or gpipe
          activation_checkpointing: true
    data:
      runtime_packing: true
    ```

=== "FSDP / HSDP"

    ```yaml
    training:
      distributed:
        strategy: fsdp
        fsdp:
          sharding_strategy: FULL_SHARD     # HYBRID_SHARD for HSDP
          # hsdp_shard_size: 2              # HSDP: GPUs per group (2 = one node)
    ```

!!! info "Why DDP, QLoRA and Pipeline scale best on PARAM Rudra"
    With LoRA the base model is frozen, so DDP, QLoRA and Pipeline only send small adapter
    updates between nodes. FSDP and HSDP instead fetch model weights over the network every
    step. Without GPUDirect RDMA those transfers go through host memory, so FSDP/HSDP efficiency
    drops at 128 nodes (MFU 5–7%).
