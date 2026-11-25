---
icon: material/download
---

# Installation

There are two ways to use Bhaskera on PARAM Rudra. Most users should load the module.

## Option A: Module (recommended)

```bash
$ module avail bhaskera
$ module load bhaskera/1.0.0
```

The module provides:

| Item | Where |
|---|---|
| Commands `bhaskera-tokenize`, `bhaskera-train`, `bhaskera-dashboard` and the Python environment | on `PATH` |
| Job scripts (`submit.sh`, `train.sbatch`, `tokenize.sbatch`, `eval.sbatch`, `serve.sbatch`), tested recipes and the PARAM Rudra cluster profile | `$BHASKERA_KIT` |
| Bhaskera source | `$BHASKERA_DIR` |

Copy the job scripts and recipes into your own folder on `/scratch`:

```bash
$ cp -r $BHASKERA_KIT /scratch/<project>/<username>/bhaskera-runs
$ cd /scratch/<project>/<username>/bhaskera-runs
$ ls
clusters  eval.sbatch  recipes  serve.sbatch  submit.sh  tokenize.sbatch  train.sbatch
```

Continue with the [End-to-end Example](workflow.md).

## Option B: Your Own Installation

Install once from a login node into your `/scratch` area (5 to 15 minutes, depending on file-system load). Use this if the module is
not available, or if you want to change Bhaskera's code.

```bash
$ cd /scratch/<project>/<username>
$ git clone https://github.com/kishoryd/Bhaskera.git
$ cd Bhaskera
$ BHASKERA_CUDA=12.4 bash setup.sh
$ source .venv/bin/activate
$ uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
$ uv pip install -r slurm/requirements-paramrudra.lock
```

!!! warning "Why each step is needed"
    - **`BHASKERA_CUDA=12.4`**: login nodes have no GPU. Without it, `setup.sh` finds no CUDA and
      installs a CPU-only PyTorch that cannot train.
    - **Prebuilt flash-attention**: `setup.sh` tries to compile flash-attention, which needs the CUDA
      compiler (`nvcc`); there is no CUDA 12 compiler module on PARAM Rudra, so that step fails. The
      prebuilt package needs no compiler. All recipes need flash-attention (sequence packing uses it).
    - **`requirements-paramrudra.lock`**: the exact package versions used for every benchmark run.
      With newer Ray and grpcio releases, multi-node jobs fail at start-up (`InvalidAuthToken`, or the
      worker nodes time out waiting for the first node).

Then tell the job scripts where Bhaskera is, and copy the job scripts and recipes:

```bash
$ export BHASKERA_DIR=/scratch/<project>/<username>/Bhaskera      # add this line to ~/.bashrc
$ cp -r $BHASKERA_DIR/slurm /scratch/<project>/<username>/bhaskera-runs
```

## Verify

```bash
$ python -c "import torch, flash_attn, bitsandbytes, bhaskera; print('OK', torch.__version__, torch.version.cuda)"
OK 2.6.0+cu124 12.4
```

## Where your files go

Everything you create stays in your **run folder**, the one that holds `submit.sh`: the MLflow store (`mlflow-runs/`, with the model registry), your small settings
(`.bhaskera/`), the benchmark data cache (`.cache/huggingface/`), recipes, checkpoints, logs and caches. Nothing is written to the home folder any more. If you already have
a store in your home folder it is still used until a copy or a link exists in the run folder, so no earlier run disappears. To keep an old store, link it:
`ln -s ~/mlflow-runs mlflow-runs`.

## Shared models and data

If the administrators keep models and tokenised data in `/home/apps/bhaskera/models` and `/home/apps/bhaskera/data`, the page offers them under *Shared on this cluster*
in the model and data lists. You can always type the path of your own folder instead.

## For Administrators: Central Installation

On a cluster other than PARAM Rudra, follow [Setting Up on Another Cluster](new-cluster.md) instead: it covers the cluster profile, the modulefile and a first check.

`install-bhaskera.sh` builds Bhaskera for all users and `bhaskera/1.0.0.lua` is the matching Lmod
modulefile. Both are in the Bhaskera repository under `slurm/`.

```bash
$ bash slurm/install-bhaskera.sh                       # installs to /home/apps/bhaskera/1.0.0
$ mkdir -p /home/apps/AIStack/bhaskera
$ cp slurm/modulefiles/bhaskera/1.0.0.lua /home/apps/AIStack/bhaskera/1.0.0.lua
```

The installer must be run with the final path (`PREFIX=...` to change it), because a Python
environment cannot be moved after it is built. It also installs Prometheus and Grafana under
`/home/apps/bhaskera/1.0.0/observability` (for the charts in the Ray Dashboard); this step needs internet
access, and `WITH_CHARTS=0` skips it. It also installs the lm-evaluation-harness for benchmarks (`WITH_EVAL=0` skips it).
The modulefile sets `BHASKERA_CLUSTER_PROFILE` to the PARAM Rudra profile, which holds the partitions, GPUs per node and network
interface; another site writes its own profile (`clusters/example.yaml`). It ends by printing the installed versions:

```text
OK  torch 2.6.0+cu124 (CUDA 12.4)  flash-attn 2.7.3  bitsandbytes 0.50.2  transformers 4.57.6  ray 2.56.1  peft 0.20.0
```
