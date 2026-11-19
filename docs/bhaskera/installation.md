# Installation

There are two ways to use Bhaskera on PARAM Rudra. Most users should load the module.

## Option A: Module (recommended)

```bash
$ module avail bhaskera
$ module load bhaskera/2.2.0
```

The module provides:

| Item | Where |
|---|---|
| Commands `bhaskera-tokenize`, `bhaskera-train`, `bhaskera-dashboard` and the Python environment | on `PATH` |
| Job scripts (`train.sbatch`, `tokenize.sbatch`, `submit.sh`) and tested recipes | `$BHASKERA_KIT` |
| Bhaskera source | `$BHASKERA_DIR` |

Copy the job scripts and recipes into your own folder on `/scratch`:

```bash
$ cp -r $BHASKERA_KIT /scratch/<project>/<username>/bhaskera-runs
$ cd /scratch/<project>/<username>/bhaskera-runs
$ ls
recipes  submit.sh  tokenize.sbatch  train.sbatch
```

Continue with the [End-to-end Example](workflow.md).

## Option B: Your Own Installation

Install once from a login node into your `/scratch` area (about 5 minutes). Use this if the module is
not available, or if you want to change Bhaskera's code.

```bash
$ cd /scratch/<project>/<username>
$ git clone https://github.com/kishoryd/Bhaskera.git
$ cd Bhaskera
$ BHASKERA_CUDA=12.4 bash setup.sh
$ source .venv/bin/activate
$ uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
$ uv pip install "ray[train,data,default]==2.56.1" peft==0.20.0 accelerate==1.14.0 liger-kernel==0.8.1 mlflow==3.15.0 pyarrow==25.0.0 bitsandbytes==0.50.2
```

!!! warning "Why each step is needed"
    - **`BHASKERA_CUDA=12.4`**: login nodes have no GPU. Without it, `setup.sh` finds no CUDA and
      installs a CPU-only PyTorch that cannot train.
    - **Prebuilt flash-attention**: `setup.sh` tries to compile flash-attention, which needs the CUDA
      compiler (`nvcc`); there is no CUDA 12 compiler module on PARAM Rudra, so that step fails. The
      prebuilt package needs no compiler. All recipes need flash-attention (sequence packing uses it).
    - **Pinned versions**: these are the versions used for every benchmark run. Newer Ray versions
      require an authentication token between nodes and multi-node jobs fail at start-up with
      `InvalidAuthToken`.

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

## For Administrators: Central Installation

`install-bhaskera.sh` builds Bhaskera for all users and `bhaskera/2.2.0.lua` is the matching Lmod
modulefile. Both are in the Bhaskera repository under `slurm/`.

```bash
$ bash slurm/install-bhaskera.sh                       # installs to /home/apps/bhaskera/2.2.0
$ cp slurm/modulefiles/bhaskera/2.2.0.lua /home/apps/modulefiles/bhaskera/2.2.0.lua
```

The installer must be run with the final path (`PREFIX=...` to change it), because a Python
environment cannot be moved after it is built. It ends by printing the installed versions:

```text
OK  torch 2.6.0+cu124 (CUDA 12.4)  flash-attn 2.7.3  bitsandbytes 0.50.2  transformers 4.57.6  ray 2.56.1  peft 0.20.0
```
