---
icon: material/monitor-dashboard
---

# Monitoring

## Introduction

A Bhaskera job can be followed in three places:

| Where | What it shows | Best for |
|---|---|---|
| The job log | One line per step, and a one-line summary of all nodes every 10 steps | A quick look while the job runs |
| MLflow | Training curves, and GPU and host curves for the whole job and for each node | Comparing runs, and looking back at a finished run |
| Ray Dashboard | Every node of the running job with its CPU, memory and GPU load, and the logs of each worker | Looking at the nodes while the job runs, at any number of nodes |

## The Step Log

Every training step prints one line to the job log. Follow it while the job runs:

```bash
$ squeue -u $USER
$ tail -f logs/<recipe>_<jobid>.out | grep --line-buffered -E '\[step|eval|exit code'
```

Two steps and the evaluation from a Qwen3-14B QLoRA run on 8 nodes:

```text
[epoch 0][step 9] loss=1.1703 lr=1.81e-05 grad_norm=0.2473 tok/s=27711 MFU=35.8% HFU=53.2% pad=0% step_s=4.70 MFU_step=36.0% mem=16.0/16.3GiB
[epoch 0][step 10] loss=1.0350 lr=2.01e-05 grad_norm=0.2650 tok/s=27620 MFU=35.7% HFU=53.0% pad=0% step_s=4.81 MFU_step=35.2% mem=16.0/16.3GiB
[epoch 0][step 10] eval_loss=1.5705 eval_acc=0.6557
```

| Field | Meaning |
|---|---|
| `loss` | Training loss (lower is better) |
| `eval_loss`, `eval_acc` | Loss and next-token accuracy on your validation data |
| `lr`, `grad_norm` | Learning rate; size of the update (a sudden jump is a warning sign) |
| `tok/s` | Real (non-padding) tokens processed per second by all GPUs together |
| `MFU` | Share of the GPUs' peak maths speed doing useful model work |
| `pad` | Share of each batch that is padding (about 0% with sequence packing) |
| `step_s` | Seconds for this step |
| `mem` | Peak GPU memory on the busiest GPU of the job, allocated / reserved |
| `slowest=rN` | Shown when one GPU lags: the slowest rank, and its step time against the median |

Every 10 steps a second line summarises all nodes of the job:

```text
[step 30] nodes=5 gpus=10 gpu_util=94/98/99% (min/mean/max) gpu_mem_max=17.7GiB temp_max=58C
```

A low minimum next to a high mean points at one GPU that is waiting for the others. The interval is
`monitoring.metrics.node_stats_every_n_steps` in the recipe (0 turns the summary and the node curves off).

!!! note "Check the log, not only the job state"
    A job can end as `COMPLETED` in SLURM even when training failed. A successful run ends with
    `=== training exit code: 0 ===`.

## MLflow Dashboard

Every run is recorded with MLflow under `~/mlflow-runs`, so runs can be charted and compared in a web page.
Start the dashboard on a **login node**:

```bash
$ module load bhaskera/1.1.0
$ bhaskera-dashboard start --port 5055      # the port is remembered for next time
$ bhaskera-dashboard status
$ bhaskera-dashboard tunnel                 # prints the SSH command for your laptop
```

`bhaskera-dashboard tunnel` prints the command for the login node you are on, for example:

```text
  From your laptop:
    ssh -L 5055:localhost:5055 login02
    open http://localhost:5055
```

On your laptop, run that command through your usual connection to PARAM Rudra (see
[Getting Access](../access.md)) so that it ends on the same login node, and open <http://localhost:5055>, then
the experiment **bhaskera**. A multi-node job is one run. It has these curves:

| Metric | What it is |
|---|---|
| `loss`, `lr`, `grad_norm`, `eval_loss`, `eval_accuracy` | The model |
| `throughput_tokens_per_sec`, `throughput_mfu_pct` | Speed of the whole job |
| `time_step_max_s`, `time_step_median_s`, `time_slowest_rank` | Spread of the step time over all ranks: the straggler signal |
| `cluster_gpu_util_pct_min`, `_mean`, `_max` | GPU utilisation over every GPU of every node |
| `cluster_gpu_mem_used_gib_max`, `cluster_gpu_temp_c_max`, `cluster_gpu_power_w_sum` | Memory, temperature and power over every GPU |
| `node_<host>_gpu<i>_util_pct`, `_mem_used_gib`, `_temp_c`, `_power_w` | One set of curves per GPU of each node, for example `node_cbgpu0241_gpu0_util_pct` |
| `node_<host>_cpu_pct`, `_mem_used_pct`, `_loadavg_1m` | CPU and memory of each node |

MLflow shows the `/` of a metric name as `_`. Type `cluster` or a node name in the metric filter to find
these curves. The older `gpu_0_...`, `gpu_1_...` and `sys_...` curves describe only the node of rank 0.

The `node_<host>_...` curves are written while the job has at most 16 nodes
(`monitoring.metrics.node_stats_per_node_max`). A larger job would write more than a thousand curves every
time, so it keeps the `cluster_...` curves, and the Ray Dashboard shows the individual nodes.

Stop the MLflow dashboard when you are done:

```bash
$ bhaskera-dashboard stop
```

!!! note
    The MLflow dashboard runs only on the login node where you started it, so the tunnel must end on that
    login node. If the port is in use, choose another with `--port`.

## Ray Dashboard

Ray starts a dashboard on the first node of every job. It lists all nodes of the job with their CPU, memory
and GPU load while the job runs, with charts over time, and it works for any number of nodes. It listens on
that node only, so it is reached through an SSH tunnel. You do not need to know which node that is, or wait
for the job to start: `bhaskera-dashboard ray` finds your job, waits while it is still queued, and links the
first node of the job to a port of your own on the login node.

```bash
$ module load bhaskera/1.1.0
$ bhaskera-dashboard ray                  # or: bhaskera-dashboard ray --job <jobid>

From your laptop, this one command logs in and keeps the dashboard link open
(the command is always the same; it finds your job and its nodes by itself):

    ssh -p 4422 -L 8265:localhost:20029 -L 3000:localhost:30029 <user>@paramrudra.cdacb.in "bash -lc 'module load bhaskera/1.1.0 && bhaskera-dashboard ray'"
```

Run that command on your laptop, leave it open, and open <http://localhost:8265> in a browser. Run it
before or after you submit the job; it waits for the job to get its nodes. The numbers after `-L` are
ports reserved for your user name on the login node, so they differ from user to user and stay the same
for you.

To avoid typing the command, run `bhaskera-dashboard ssh-config` on the login node and paste the block it
prints into `~/.ssh/config` on your laptop (once). From then on:

```bash
$ ssh bhaskera-ray
```

!!! note "Laptops"
    The `ssh-config` entry needs OpenSSH 7.6 or newer, which Mac, Linux and Windows 10 or later have. With
    PuTTY, or if port 8265 or 3000 is already in use on your laptop, forward the ports from VS Code (below).

In the dashboard:

- **Cluster** lists every node with its CPU, memory and, per GPU, utilisation and memory. A node that is idle
  or full stands out here.
- **Cluster utilization** (on the Overview page) and **Metrics** draw GPU, CPU and memory over time for
  every node. They are Grafana charts, also open at <http://localhost:3000>. They are there when
  Prometheus and Grafana are installed next to Bhaskera (the module sets `BHASKERA_OBSERVABILITY`); the
  job log says `charts: Prometheus and Grafana started` when they run. Without them the node list still
  works, only the charts are missing.
- **Jobs** shows the training job and its workers; **Logs** has the output of each worker, which helps when
  one node fails.
- The dashboard exists only while the job runs. After it ends, use the MLflow run.

!!! note "Two jobs in the list"
    Ray numbers its jobs itself (`01000000`, `02000000`), so these numbers differ from the SLURM job ID. One
    SLURM job shows two entries. The first is `ray symmetric-run`, which starts the cluster on every node
    and has no tasks. The second is the training itself.

If you run several jobs at once, the command follows the newest one; use `--job <jobid>` for another.

!!! tip "VS Code"
    If you work through VS Code Remote-SSH, run `bhaskera-dashboard ray` in its terminal, then forward port
    8265 (and 3000 for the charts, 5055 for MLflow) from the **Ports** tab instead of typing the tunnel
    commands.
