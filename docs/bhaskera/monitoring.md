---
icon: material/monitor-dashboard
---

# Monitoring

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
| `mem` | Peak GPU memory on the busiest GPU, allocated / reserved |

!!! note "Check the log, not only the job state"
    A job can end as `COMPLETED` in SLURM even when training failed. A successful run ends with
    `=== training exit code: 0 ===`.

## MLflow Dashboard

Every run is also recorded with MLflow under `~/mlflow-runs`, so runs can be charted and compared in a
web page. Start the dashboard on a **login node**:

```bash
$ module load bhaskera/1.0.0
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
[Getting Access](../access.md)) so that it ends on the same login node, and open
<http://localhost:5055>, then the experiment **bhaskera**. Useful charts: `loss`, `eval_loss`,
`eval_accuracy`, `throughput/mfu_pct`.

Stop the dashboard when you are done:

```bash
$ bhaskera-dashboard stop
```

!!! note
    The dashboard runs only on the login node where you started it, so the tunnel must end on that
    login node. If the port is in use, choose another with `--port`.
