---
icon: material/server-network
---

# Setting Up Bhaskera on Another Cluster

This page is for the person who installs Bhaskera on a cluster for the first time: an administrator, or an engineer with a few Slurm GPU nodes. It takes about an hour, most of it waiting for downloads.
The other pages of this guide describe PARAM Rudra, where Bhaskera has been built and tested. Everything that is specific to a cluster lives in two places you will edit (a cluster profile and a modulefile), so no code needs to change.

!!! warning "What has been tested"
    Bhaskera has been run end to end only on PARAM Rudra: A100 80 GB GPUs, 2 GPUs per node, InfiniBand, Slurm, Lmod, and compute nodes without internet. On any other cluster, treat your first job as a test and follow the check at the end of this page before you tell users it works.
    GPUs other than the A100 80 GB, CPUs other than x86-64, and CUDA versions other than 12.x have not been tried.

## What your cluster needs

| Need | Why | Check |
|---|---|---|
| Slurm with GPUs requested as `--gres=gpu:N` | Every job goes through `sbatch` | `sinfo -o '%P %G %l'` lists a GPU partition |
| NVIDIA GPUs and a driver that supports CUDA 12.4 | The installed PyTorch is built for CUDA 12.4 | `nvidia-smi` on a GPU node |
| Linux on x86-64 | The prebuilt flash-attention package is for it | `uname -m` prints `x86_64` |
| A shared file system that the login nodes and all compute nodes see at the same path | The install, recipes, data and checkpoints are read by every node | The same `ls` on a login node and in a job |
| A login node with internet (compute nodes do not need it) | The installer downloads Python packages and flash-attention; models and datasets are downloaded there too | `curl -sI https://pypi.org` |
| `git` and about 15 GB free for the install | The install is about 8 GB (the Python environment with PyTorch is most of it), plus room to update | `df -h <install folder>` |
| Permission to give users read access to the install folder | One central install serves everyone | You can write the folder and `chmod a+rX` it |

Helpful but optional: Lmod or Environment Modules (users then run `module load bhaskera`), `sacctmgr` (so the page can show each user's job limits), and a place to keep shared models and data.
Compute nodes without internet are fine, and are the usual case: Bhaskera switches Hugging Face to offline mode inside jobs.

## Decide these first

You will write these down once, in the cluster profile.

| Decision | How to find it |
|---|---|
| Where to install, for example `/apps/bhaskera/1.0.0` | A path that users and compute nodes can read. The environment is built there and cannot be moved afterwards |
| GPUs per node | `scontrol show node <gpu node> | grep Gres` |
| Memory of one GPU, in GB | `nvidia-smi --query-gpu=memory.total --format=csv` |
| CPUs and memory to ask for on each node | `scontrol show node <gpu node>`. Many sites give GPU jobs a small memory default, so name a number |
| The fast network interface | `ip -4 addr` on a compute node: the InfiniBand or high-speed Ethernet name (`ib0`, `eth1`, ...). Leave it empty to skip the setting |
| Partitions, from the smallest jobs to the largest, with their time limits | `sinfo -o '%P %D %l'` and your site's policy on how many nodes each partition takes |

## Step 1: Install

Run this once, on a login node, as the person who owns the install folder. Use a checkout of the version you want:

```bash
$ git clone https://github.com/kishoryd/Bhaskera.git && cd Bhaskera
$ REF=main PREFIX=/apps/bhaskera/1.0.0 bash slurm/install-bhaskera.sh
```

`REF` is the version to install (a tag, branch or commit). The script's default is the tag `v1.0.0`, which was made before the newest features; use `REF=main`, or a newer tag, to get the page described in this guide.
`PREFIX` is the install folder. `WITH_CHARTS=0` skips Prometheus and Grafana (the time-series charts of the Ray Dashboard; the dashboard still lists nodes and GPUs without them), and `WITH_EVAL=0` skips the benchmark package.

The script makes, under `PREFIX`:

| Folder | What it holds |
|---|---|
| `src/` | The Bhaskera source, at `REF` |
| `python/`, `venv/` | A Python 3.11 interpreter and an environment with PyTorch 2.6 (CUDA 12.4), flash-attention 2.7.3, bitsandbytes, Ray, PEFT and the pinned versions below |
| `observability/` | Prometheus and Grafana, unless skipped |
| `share/kit/` | The job scripts (`submit.sh`, `*.sbatch`), ready recipes, the text and multimodal starter configs (`configs/`) and cluster profiles, which users copy |

!!! note "Why the versions are pinned"
    `slurm/requirements-paramrudra.lock` is the exact set of package versions used for every benchmark. With newer Ray and grpcio releases, multi-node jobs fail at start-up (`InvalidAuthToken`, or workers timing out while they wait for the first node). The file's name says PARAM Rudra, but it is simply the tested set. Keep it.

A login node has no GPU, so the installer names CUDA 12.4 itself. At the end it prints one line with the installed versions; check that torch reports a CUDA build, not `cpu`:

```text
OK  torch 2.6.0+cu124 (CUDA 12.4)  flash-attn 2.7.3  bitsandbytes 0.50.2  transformers 4.57.6  ray 2.56.1  peft 0.20.0
```

If the login node cannot reach the internet, the installer cannot run as it is: it needs `pypi.org`, `download.pytorch.org` and `github.com`. Use a machine that can, or an internal mirror; offline installs have not been tried.

## Step 2: Write the cluster profile

The profile tells Bhaskera what your nodes and partitions are. Copy the example and edit it:

```bash
$ cp $PREFIX/share/kit/clusters/example.yaml $PREFIX/share/kit/clusters/mycluster.yaml
```

```yaml
name: My Cluster          # shown on the dashboard
gpus_per_node: 8
gpu_mem_gb: 80            # memory of one GPU; the "will it fit" check before a job uses it
cpus_per_node: 64         # CPUs asked for on each node
mem_per_node: 400G        # host memory asked for on each node
net_interface: ib0        # for NCCL and Ray; "" = not set
docs_url: https://example.org/bhaskera
partitions:               # smallest jobs first; max_hours is the partition's time limit
  - {max_nodes: 2,  name: debug, max_hours: 1}
  - {max_nodes: 32, name: gpu,   max_hours: 48}
```

Every line is optional. A job of up to `max_nodes` nodes goes to the first partition that fits it, with that partition's time limit. Delete `param-rudra.yaml` from the copy users get if you do not want it there.
Check what Bhaskera reads from it:

```bash
$ BHASKERA_CLUSTER_PROFILE=$PREFIX/share/kit/clusters/mycluster.yaml \
    $PREFIX/venv/bin/python -m bhaskera.launcher.cluster_profile plan 2
debug 1 8 64 ib0 400G       # partition, hours, GPUs per node, CPUs, interface, memory: here for a job of 2 nodes
```

## Step 3: Make Bhaskera easy to start

**With Lmod.** Copy the modulefile and edit its first lines:

```bash
$ mkdir -p <a MODULEPATH folder>/bhaskera
$ cp slurm/modulefiles/bhaskera/1.0.0.lua <a MODULEPATH folder>/bhaskera/1.0.0.lua
```

Change `root` to your `PREFIX`, `BHASKERA_CLUSTER_PROFILE` to your profile, `BHASKERA_LOGIN_HOST` and `BHASKERA_LOGIN_PORT` to the address people use to log in over SSH (the dashboard prints the tunnel command for their laptop from them), and the manual link in `help`. If your cluster keeps software in a hierarchy, put the file where your other applications are.

**Without modules.** Users, or a small script you give them, set:

```bash
export BHASKERA_DIR=$PREFIX/src
export BHASKERA_KIT=$PREFIX/share/kit
export BHASKERA_CLUSTER_PROFILE=$PREFIX/share/kit/clusters/mycluster.yaml
export MLFLOW_ALLOW_FILE_STORE=true
export PATH=$PREFIX/venv/bin:$PATH
```

Then look at `$PREFIX/src/bhaskera-activate.sh`, which every job sources. The installer wrote it, and it loads a `cuda/12.4` module when one exists. If your CUDA module has another name, edit that line. A name that cannot be loaded does not stop the job (the load is allowed to fail), so a wrong one shows only as a missing CUDA toolkit; read the first lines of a job log once.

## Step 4: Shared models and data (optional, recommended)

Compute nodes have no internet, so models and data must be on disk first. A shared place saves every user a download:

```bash
$ mkdir -p $(dirname $PREFIX)/models $(dirname $PREFIX)/data
```

The Bhaskera dashboard lists anything in `models/` and `data/` next to the install folder, or in the folders named by `BHASKERA_MODELS` and `BHASKERA_DATA`, under *Shared on this cluster*. Put Hugging Face model folders in `models/` (users can also fetch their own from the **Downloads** menu of the page). Put tokenised data in `data/` in folders named `<dataset>_<model>_seq<length>_train` and `…_val`: a tokenised folder only fits the model whose tokenizer made it. `chmod -R a+rX` both folders.

## Step 5: Tune what depends on your GPUs

| Setting | Where | Why |
|---|---|---|
| `gpu_mem_gb` | the cluster profile | The check before a job estimates whether it fits in GPU memory. Without it, 80 GB is assumed |
| `peak_tflops_per_gpu` | a recipe, under `monitoring: metrics:` | The MFU figure is measured against this peak. The default 312 is the A100's bf16 figure; use your GPU's, from the vendor's datasheet, or MFU will read wrongly |
| `net_interface` | the cluster profile | Multi-node jobs talk over this interface. A wrong name makes NCCL slow or stuck |

## Step 6: Check it as an ordinary user

Do this from a normal user account, not the one that installed. Each step names what should happen.

```bash
$ module load bhaskera/1.0.0                         # or the exports above
$ python -c "import torch, flash_attn, bitsandbytes, bhaskera; print('OK', torch.__version__, torch.version.cuda)"
OK 2.6.0+cu124 12.4
$ cp -r $BHASKERA_KIT ~/bhaskera-runs && cd ~/bhaskera-runs   # or any folder you choose
$ DRY_RUN=1 ./submit.sh recipes/qwen3-14b-qlora.yaml 1 1      # every check, nothing submitted
```

The dry run should end with `RESULT: OK`, or fail only on the placeholders in the sample recipe (`<project>/<username>`), which means the install is reading its own files. Then run something small for real:

1. Download or copy the smallest model you have into a folder you own, and tokenise a few hundred examples (`./submit.sh tokenize …`, or **Prepare data** on the page).
2. Make a recipe on the page (**New run, Create recipe**) with 10 steps, on 1 node, and **Start** it.
3. In the job log you should see `[step]` lines with a loss, and the run should finish with `training exit code: 0`.
4. Open the page from your laptop with the SSH command the module prints (`bhaskera-dashboard ray --info`), and check that the run, its loss curve and its GPU energy are listed.

Then try 2 nodes. A single node working does not prove the network settings: most multi-node problems show up only here.

Users need to set nothing: jobs are charged to their login name, and the page makes `~/bhaskera-runs` for them on first start. If your Slurm accounts are not the login names, tell users to `export SLURM_ACCOUNT=<account>`. Bhaskera checks the free space of a disk before it writes anything large (checkpoints, prepared data, downloads) and refuses a job on a disk with under 1 GB free, and the page warns on Home below 20 GB; a per-user quota is not visible to it. If home folders have small quotas, tell users to keep their run folder on the large file system (`cp -r $BHASKERA_KIT <big folder>/bhaskera-runs`, then start the page from it): checkpoints are large.

## Where it breaks on a new cluster

| What you see | Likely cause | What to do |
|---|---|---|
| No CUDA toolkit in a job, with nothing said about it | The CUDA module has another name, and the failed load is ignored | Edit the module line in `$PREFIX/src/bhaskera-activate.sh` |
| Multi-node job hangs or is slow at the first step | Wrong or missing `net_interface`, or a firewall between compute nodes | Check `ip -4 addr` on two nodes; set `net_interface`; test with `NCCL_DEBUG=INFO` |
| `InvalidAuthToken`, or workers waiting for the first node | Ray or grpcio newer than the pinned set | Reinstall with the lock file (Step 1) |
| Job cannot be submitted: "invalid account" | The Slurm account differs from the login name | `export SLURM_ACCOUNT=<account>` |
| Job is rejected for memory | The site's default memory per node is small | Set `mem_per_node` in the profile |
| The site does not allow `--exclusive` | `train.sbatch` asks for whole nodes | Remove the `--exclusive` line from `share/kit/train.sbatch` (and its copies in users' folders) |
| The page shows no job slots, or no limits | `sacctmgr` is missing or accounting is off | The limit checks are skipped; jobs still start |
| "will it fit" is wrong | GPUs are not 80 GB | Set `gpu_mem_gb` in the profile |
| Downloads, or the benchmark package, fail | The login node has no internet | Fetch models and data on a machine that has it, and copy them in |
| A node is skipped for three days | The node check found a fault on it (for example uncorrected GPU memory errors) | It is working as intended; ask your hardware team about the node |

## Keeping it up to date

A new version needs three things, in this order:

1. Update the source: `git -C $PREFIX/src fetch && git -C $PREFIX/src checkout <ref>`.
2. **Copy the job scripts again.** The install copies `submit.sh`, the `*.sbatch` files, recipes and profiles into `share/kit/` as separate files, so updating the source does not update them. Rerun `install-bhaskera.sh`, or copy the files from `$PREFIX/src/slurm/` into `$PREFIX/share/kit/` yourself. Users who already copied the kit keep their own copy and have to copy the new scripts too.
3. `chmod -R a+rX $PREFIX`.

Restart the dashboard after an update: a page that is already running keeps the old code.
