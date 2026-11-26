---
icon: material/lifebuoy
---

# Troubleshooting

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
| `oom_kill` events in a tokenize job | The tokenize job ran with a small default memory | Use the provided `tokenize.sbatch` (`--exclusive --mem=180G`) |

### Excluding a faulty node

Find the node behind a failing worker from the address in the log (`ip=172.41.x.y`):

```bash
$ grep -w 172.41.x.y /etc/hosts
172.41.x.y cbgpuXXXX-ib0 cbgpuXXXX-ib0.cdacb.in
```

Then keep your jobs off it. `submit.sh` passes `BHASKERA_EXCLUDE` to `sbatch --exclude`:

```bash
$ export BHASKERA_EXCLUDE=cbgpuXXXX          # comma-separated for several nodes
$ ./submit.sh recipes/<recipe>.yaml 8 12
```

!!! note
    The `SBATCH_EXCLUDE` environment variable is not applied on PARAM Rudra; use `BHASKERA_EXCLUDE`
    (or `--exclude=` when calling `sbatch` yourself).

## Known Limitations

- **QLoRA, DDP and FSDP recipes write a checkpoint only when the run ends**, so a run longer than the
  time limit has to be split with `max_steps` (see
  [Choosing Nodes and Hours](recipes.md#choosing-nodes-and-hours)). Pipeline recipes checkpoint every 500
  steps. All of them continue from the latest checkpoint when resubmitted, but the data starts again from
  the beginning of the training set.
- **MoE models are slow outside the Pipeline strategy** (0.2–1.7% MFU), because the standard expert code
  processes experts one at a time. Use the Pipeline recipes for MoE models.
-

## Quick Reference

| Task | Command |
|---|---|
| Load Bhaskera | `module load bhaskera/1.0.0` (own installation: `source $BHASKERA_DIR/.venv/bin/activate`) |
| Get the job scripts and recipes | `cp -r $BHASKERA_KIT /scratch/<project>/<username>/bhaskera-runs` |
| Download a model (login node) | `huggingface-cli download <org>/<model> --local-dir /scratch/<project>/<username>/models/<model>` |
| Tokenise | `./submit.sh tokenize recipes/tokenize-chatml.yaml` |
| Train (recipe, nodes, hours) | `./submit.sh recipes/<recipe>.yaml 8 12` |
| Follow the log | `tail -f logs/<recipe>_<jobid>.out \| grep -E '\[step\|eval\|exit code'` |
| Check a recipe without submitting | `python -m bhaskera.launcher.preflight --config recipes/<recipe>.yaml --num-gpus 16` |
| MLflow dashboard | `bhaskera-dashboard start --port 5055` / `status` / `tunnel` / `stop` |
| Bhaskera Dashboard: jobs, start, chat | `bhaskera-dashboard ray` (see [Bhaskera Dashboard](homepage.md)) |
| Chat with a trained run | `./submit.sh serve --run <run id>` |
| Benchmark a trained run | `./submit.sh eval --run <run id> --tasks arc_easy,hellaswag --limit 200` |
| Continue a run that stopped | `./submit.sh resume --run <run id>` (add `--step N` to go back) |
| Save a trained model in the registry | `python -m bhaskera.launcher.register add --run <run id> --name <model> --alias champion` (see [Model Registry](registry.md)) |
| Hyperparameter sweep | `python -m bhaskera.launcher.sweep plan\|submit\|report sweep.yaml` |
| Check and clean a data file | `python -m bhaskera.launcher.data_tools inspect\|dedupe\|filter\|split` |

## The MLflow link says the site cannot be reached

The page's links to MLflow and to each job's Ray Dashboard use addresses like `mlflow.localhost:8265`, which go through the same SSH tunnel as the page itself.

- If the page opens but `mlflow.localhost` is refused, the browser may be trying the IPv6 address (`::1`) while your tunnel listens on IPv4 only. Add a second forward to the
  SSH command, `-L [::1]:8265:localhost:<port>`, or use Chrome or Firefox, which resolve `*.localhost` themselves.
- If MLflow itself is not running on the login node, start it with `bhaskera-dashboard start`.
- The Bhaskera dashboard's **Models** and **Runs** sections do not need MLflow's page: they read the same files directly.
On PARAM Rudra the module is found on every login node except login04, whose `MODULEPATH` lacks `/home/apps/AIStack` (a setup difference on that node, not a Bhaskera fault). Use another login node, or add the path yourself: `module use /home/apps/AIStack`, then load it. If it is unknown elsewhere, the module cache may be out of date: `module --ignore_cache load bhaskera/1.0.0`.
