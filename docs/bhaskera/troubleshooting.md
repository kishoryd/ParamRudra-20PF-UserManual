# Troubleshooting and Reference

## Troubleshooting

| Symptom (in the job log) | Cause | What to do |
|---|---|---|
| `CUDA out of memory` / `GPU OOM` | Model or batch too large for the strategy | Use QLoRA or Pipeline; lower `batch_size` and raise `grad_accum` |
| `node running low on memory` (Ray kills workers) | Host RAM exhausted while loading a very large model | Use more Pipeline stages or a strategy that loads once per node (FSDP) |
| `invalid device ordinal` | A GPU on one node failed to start | Resubmit with `--exclude=<node>` |
| `No space left on device` / `Local disk is full` | That node's local `/tmp` is full | Resubmit with `--exclude=<node>` |
| `DistBackendError` / NCCL timeout | Network fault or a hung rank | Resubmit; if it repeats on one node, exclude it |
| Job `COMPLETED` but no `[step` lines | Ray or training failed at startup | Read the end of the log and the `.err` output for the first error |
| `Timed out waiting for other nodes` | Ray could not start on every node | Resubmit; check that `ib0` is up on the nodes |
| MLflow page shows `404 Not Found` | Another service holds that port on the login node | Start the dashboard on a different `--port` |
| Steps much slower than usual; `slowest=rN` | A straggler node | Check that node; exclude it next time |

Find the node behind a failing rank in the log (`ip=172.41.x.y`), then map it to a hostname:

```bash
grep -w 172.41.x.y /etc/hosts     # -> 172.41.x.y  cbgpuXXXX-ib0 ...
```

## Known Limitations

- **MoE models are slow outside Pipeline** (0.2–1.7% MFU), because the standard expert code
  processes experts one at a time. Use the Pipeline strategy for MoE models.
- **Checkpoint frequency depends on the strategy.** With Pipeline, a checkpoint is saved every
  `checkpoint.save_interval` steps. With FSDP, HSDP, DDP and QLoRA, it is saved only when an
  epoch ends, so a long single-epoch run has no mid-run checkpoint. Split long runs, or use
  Pipeline.
- **DDP needs the whole model on each GPU**: 70 B and larger do not fit. Use QLoRA or Pipeline.
- **Qwen3-235B does not fit with QLoRA** (about 118 GB of 4-bit weights per GPU).
  Mistral-Large-123B with QLoRA fits only at `batch_size: 1` (78 of 80 GB).
- **gpt-oss-120b is not supported** (its MXFP4 experts cannot run on A100).
- **Inference serving is not part of this release.** `bhaskera-infer` exists but was not
  re-validated here.

## Quick Reference

| Task | Command |
|---|---|
| Load | `module load bhaskera/2.2.0` (own install: `source bhaskera-activate.sh`) |
| Get the tested configs | `cp $BHASKERA_HOME/configs/release/<file>.yaml configs/` |
| Tokenise | `bhaskera-tokenize --config configs/my_tokenize.yaml --split both` (as a `gpu-debug` job) |
| Check before submitting | `bhaskera-preflight --config configs/my_run.yaml --num-gpus 16` |
| Train | `sbatch -A <acct> --nodes=8 --partition=gpu-small --time=08:00:00 $BHASKERA_HOME/scripts/train.sbatch configs/my_run.yaml` |
| Follow the log | `tail -f logs/bhaskera_<jobid>.out \| grep -E '\[step\|eval'` |
| MLflow UI | `bhaskera-dashboard start --port 5055` / `stop` / `status` |
| Ray Dashboard | `ssh -N -L 8265:localhost:8265 <first-node-of-job> &` (from the login node) |
| NCCL / network check | `bhaskera-diag` |

## References

- Bhaskera source and documentation: <https://github.com/CDACIndia/Bhaskera>
  (`README.md`, `ARCHITECTURE.md`, `configs/README.md`, `docs/scale-matrix-results.md`)
- [GPU Computing](../gpu.md), [Machine Learning / DL](../machine-learning.md),
  [Batch System (SLURM)](../batch.md) on this site
- PyTorch FSDP2: <https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html>
- LoRA: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- QLoRA: Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)
- Ray Train: <https://docs.ray.io/en/latest/train/train.html>
