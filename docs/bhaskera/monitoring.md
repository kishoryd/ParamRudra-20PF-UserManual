# Monitoring a Run

## The step log

```bash
squeue -u $USER
tail -f logs/bhaskera_<jobid>.out | grep --line-buffered -E '\[step|eval'
```

A step line from a test run on PARAM Rudra:

```text
[step 15] loss=1.5586 lr=3.01e-05 grad_norm=0.3502 tok/s=6940 MFU=36.0% HFU=53.4% pad=0% step_s=4.68 MFU_step=36.2% mem=16.0/16.3GiB | step=4.59s data=0.00s fwd+bwd=4.58s opt=0.00s
[step 15] eval_loss=1.5579 eval_acc=0.6567
```

| Field | Meaning |
|---|---|
| `loss` | Training loss (lower is better) |
| `eval_loss`, `eval_acc` | Loss and next-token accuracy on your validation data |
| `lr`, `grad_norm` | Learning rate; size of the update (a sudden jump is a warning sign) |
| `tok/s` | Real (non-padding) tokens processed per second by all GPUs |
| `MFU` | Model FLOPs utilisation: share of the GPUs' peak maths speed doing useful model work |
| `HFU`, `MFU_step` | Like MFU, also counting activation-checkpoint recompute; MFU of this step alone |
| `pad` | Share of the batch that was padding (≈ 0% with packing) |
| `mem` | Peak GPU memory on the busiest GPU (allocated / reserved) |
| `step`, `data`, `fwd+bwd`, `opt` | Where the step's time went |
| `slowest=rN` | Shown when one GPU lags: the slowest rank (a straggler) |

## MLflow dashboard

Every run is logged to MLflow under `~/mlflow-runs`, so all your runs can be charted and compared.

On the **login node** (note which one you are on):

```bash
bhaskera-dashboard start --port 5055    # the port is remembered next time
hostname -i                             # internal address of this login node, e.g. 172.40.0.22
```

On your **laptop**, open a tunnel through the PARAM Rudra gateway to that login node:

```bash
ssh -J <username>@paramrudra.cdacb.in:4422 -L 5055:localhost:5055 <username>@<login-node-address>
```

Then open <http://localhost:5055> → experiment **bhaskera** → your run. Useful charts: `loss`,
`eval_loss`, `eval_accuracy`, `throughput_tokens_per_sec_per_gpu`, `throughput_mfu_pct`.
Stop the UI with `bhaskera-dashboard stop`.

!!! note "Why the tunnel goes through SSH to one login node"
    `paramrudra.cdacb.in` can place you on any of several login nodes, and the login nodes
    block each other's ports except SSH. The dashboard runs only on the login node where you
    started it, so the tunnel must end on that node. If port 5055 is taken, pick another free port.

## Ray Dashboard

The Ray Dashboard shows the job's nodes, one worker per GPU, GPU use and per-worker logs. It
runs on the job's first node, and only while the job is running. On the login node:

```bash
HEAD=$(scontrol show hostnames $(squeue -j <jobid> -h -o %N) | head -1)
ssh -N -L 8265:localhost:8265 $HEAD &
```

On your laptop, add `-L 8265:localhost:8265` to the tunnel command above and open
<http://localhost:8265>.

!!! tip "VS Code"
    If you work through VS Code Remote-SSH, forward ports 5055 and 8265 from the **Ports** tab
    instead of typing tunnel commands.
