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
$ mkdir -p /scratch/<project>/<username>/bhaskera-runs && cp -r $BHASKERA_KIT/. /scratch/<project>/<username>/bhaskera-runs/
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
