---
icon: material/view-dashboard
---

# The Home Page: Jobs, Start, Chat

`bhaskera-dashboard ray` serves one web page for everything you run. It lists your jobs and your recent runs, starts new
jobs, and lets you talk to a model you have trained. It listens on the login node, is private to your user, and is opened
from your laptop with the one SSH command described under [Monitoring](monitoring.md#ray-dashboard-charts-and-mlflow-in-one-place).

## What it shows

- **Jobs**: state, partition, nodes and GPUs, progress and time left, and the live numbers from the job log (loss,
  evaluation loss and accuracy, MFU, tokens per second, step time). The GPU load of each node is one click away. Buttons
  open the Ray Dashboard, the GPU charts, the log, and the run in MLflow.
- **Runs**: your recent MLflow runs, including finished and failed ones. A run that stopped writing is marked *stalled*.
  Runs that were benchmarked show the result.
- **Cluster limits and node health** (at the bottom): your job slots in each partition, and any node a job found faulty.
  New jobs keep away from those nodes for three days.

## Start a job without the terminal

Press **Start a job**. Pick what to start:

| | You choose | What runs |
|---|---|---|
| Train a model | a recipe, the number of nodes, optionally the hours | `./submit.sh <recipe> <nodes> <hours>` |
| Chat with a trained model | a trained run | `./submit.sh serve --run <run id>` |
| Benchmark a trained model | a trained run, the benchmarks | `./submit.sh eval --run <run id> --tasks ...` |

**Check first** runs the partition limits and the recipe check and submits nothing. **Start** submits. The page runs the
same `submit.sh` you would run by hand, so every check still applies, and none can be skipped from the page. A recipe that
still has `<project>/<username>` placeholders is listed but cannot be started.

The page needs to know your account and where `submit.sh` is. Start it from your runs folder, with the account set:

```bash
$ ssh -p 4422 -L 8265:localhost:<port> <user>@paramrudra.cdacb.in \
    "bash -lc 'module load bhaskera/1.0.0 && cd /scratch/<project>/<username>/bhaskera-runs && SLURM_ACCOUNT=<account> bhaskera-dashboard ray'"
```

(`bhaskera-dashboard ray` prints the first part of this command with your port.) Without the account and folder the page
still works; the Start button then says what is missing.

## Chat with a trained model

Train with `logging.mlflow_log_adapter: true` so the adapter is stored with the run. Then press **Chat** on a finished run.
A GPU node loads the base model and the adapter, which takes a few minutes. When it is ready, a chat window opens on the page:

- **Tuned** answers with the fine-tuned model, **Base** with the original model, **Compare** shows both side by side. The base
  answer comes from the same weights with the adapter switched off, so nothing is loaded twice.
- *Options* sets the creativity and the longest answer.

The server runs on the node, stops after two hours (`SERVE_HOURS` changes it) or when you cancel the job, and answers only you.

## Benchmark a trained model

Training loss says little about what the model can still do. **Benchmark** runs standard tests
([lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): ARC, HellaSwag, MMLU, GSM8K and more) on the
base model and on the tuned model, and shows the change per task. A change counts as *better* or *worse* only when it is
larger than twice the standard error; otherwise it reads "no clear change". The result is stored with the training run and
shown in the **Runs** table.

Compute nodes have no internet, so download the test data once on a login node:

```bash
$ python -m bhaskera.launcher.evaluate prepare --tasks arc_easy,hellaswag
```

From the terminal: `./submit.sh eval --run <run id> --tasks arc_easy,hellaswag --limit 200` (`--limit` uses 200 examples per
task for a quick check; leave it out for the real score).

!!! note "Shared accounts"
    The page lists the jobs of the Unix account you are logged in with. On an account that several people share, you see
    everyone's jobs, and anyone on it can use the Start button.
