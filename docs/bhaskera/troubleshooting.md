---
icon: material/lifebuoy
---

# Troubleshooting & Reference

## Troubleshooting

| Symptom (in the job log) | Cause | What to do |
|---|---|---|
| `InvalidAuthToken: Authentication token is missing or incorrect`, job ends after ~15 s | Ray newer than 2.56 in your own installation | Install the pinned versions from [Installation](installation.md#option-b-your-own-installation) |
| `Timed out while waiting for GCS to become available` | Ray could not start on the job's first node | Resubmit; if it repeats on the same node, exclude it (below) |
| `torch.version.cuda` is `None`, or `CUDA built: None` after `setup.sh` | `setup.sh` ran on a login node without `BHASKERA_CUDA=12.4` | Delete `.venv` and rerun `BHASKERA_CUDA=12.4 bash setup.sh` |
| `preflight reported FAIL` from `submit.sh` | The run does not fit in GPU memory as configured | Use the QLoRA or Pipeline recipe, more stages, or `batch_size: 1` |
| `CUDA out of memory` | Model or batch too large for the strategy | As above; lower `batch_size` and raise `grad_accum` |
| `node running low on memory` (Ray kills workers) | Host RAM (about 187 GB per node) exhausted while loading a very large model | Use the Pipeline recipe for that model |
| `invalid device ordinal` | A GPU on one node failed to start | Resubmit with that node excluded |
| `DistBackendError` or an NCCL timeout | Network fault or a hung process | Resubmit; if it repeats on one node, exclude it |
| `oom_kill` events in a tokenize job | The tokenize job ran with a small default memory | Use the provided `tokenize.sbatch` (`--exclusive --mem=0`) |

### Excluding a faulty node

Find the node behind a failing worker from the address in the log (`ip=172.41.x.y`):

```bash
$ getent hosts 172.41.x.y
172.41.x.y   cbgpuXXXX-ib0 cbgpuXXXX-ib0.cdacb.in
```

Then keep your jobs off it. `submit.sh` passes this to `sbatch`:

```bash
$ export SBATCH_EXCLUDE=cbgpuXXXX            # comma-separated for several nodes
$ ./submit.sh recipes/<recipe>.yaml 8 12
```

## Known Limitations

- **Checkpoints with QLoRA and DDP are written only when the run ends**, so the run must fit in one job
  (see [Choosing Nodes and Hours](recipes.md#choosing-nodes-and-hours)). Pipeline recipes checkpoint
  every 500 steps and resume when resubmitted.
- **MoE models are slow outside the Pipeline strategy** (0.2–1.7% MFU), because the standard expert code
  processes experts one at a time. Use the Pipeline recipes for MoE models.
- **DDP needs the whole model on each GPU**: 70 B and larger need QLoRA or Pipeline.
- **Qwen3-235B does not fit with QLoRA** (about 118 GB of 4-bit weights per GPU).
- **gpt-oss-120b** is stored in MXFP4, which A100 GPUs cannot compute in; it has not been validated
  for training.

## Quick Reference

| Task | Command |
|---|---|
| Load Bhaskera | `module load bhaskera/2.2.0` (own installation: `source $BHASKERA_DIR/.venv/bin/activate`) |
| Get the job scripts and recipes | `cp -r $BHASKERA_KIT /scratch/<project>/<username>/bhaskera-runs` |
| Set your account | `export SLURM_ACCOUNT=<your_account>` |
| Download a model (login node) | `huggingface-cli download <org>/<model> --local-dir /scratch/<project>/<username>/models/<model>` |
| Tokenise | `./submit.sh tokenize recipes/tokenize-chatml.yaml` |
| Train (recipe, nodes, hours) | `./submit.sh recipes/<recipe>.yaml 8 12` |
| Follow the log | `tail -f logs/<recipe>_<jobid>.out \| grep -E '\[step\|eval\|exit code'` |
| Check a recipe without submitting | `python -m bhaskera.launcher.preflight --config recipes/<recipe>.yaml --num-gpus 16` |
| MLflow dashboard | `bhaskera-dashboard start --port 5055` / `status` / `tunnel` / `stop` |

## References

- Bhaskera source: <https://github.com/kishoryd/Bhaskera>
- [GPU Computing](../gpu.md), [Machine Learning / DL](../machine-learning.md),
  [Resource Management](../resource-management.md) on this site
- LoRA: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- QLoRA: Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)
- Ray Train: <https://docs.ray.io/en/latest/train/train.html>
