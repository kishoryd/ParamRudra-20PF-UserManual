# Installing and Loading Bhaskera

There are two ways to use Bhaskera. Most users should load the shared module.

=== "Option A: shared module (recommended)"

    Bhaskera is installed centrally in the AIStack. Load it like any other module:

    ```bash
    module avail bhaskera            # list installed versions
    module load bhaskera/2.2.0
    echo $BHASKERA_HOME              # tested configs, the SLURM launcher and the docs
    which bhaskera-train
    ```

    | Provided by the module | Location |
    |---|---|
    | Commands `bhaskera-tokenize`, `bhaskera-preflight`, `bhaskera-train`, `bhaskera-dashboard`, `bhaskera-infer`, `bhaskera-diag` | on `PATH` |
    | Tested configurations | `$BHASKERA_HOME/configs/release/` |
    | SLURM launcher | `$BHASKERA_HOME/scripts/train.sbatch` |
    | Documentation | `$BHASKERA_HOME/README.md`, `$BHASKERA_HOME/docs/` |
    | Python environment | PyTorch 2.6.0 + CUDA 12.4, Transformers 4.57, Ray 2.56, PEFT, bitsandbytes, flash-attention 2.7.3, Liger, MLflow |

    Work in your own folder on `/scratch` and copy the configs you need:

    ```bash
    mkdir -p /scratch/<project>/<username>/bhaskera-runs && cd $_
    mkdir -p configs logs
    cp $BHASKERA_HOME/configs/release/rel-ddp-qwen14b.yaml configs/my_run.yaml
    cp $BHASKERA_HOME/configs/local_data_tokenize_chatml.yaml configs/my_tokenize.yaml
    ```

    !!! note "Availability"
        If `module avail bhaskera` shows nothing, the shared install is not on the system yet:
        use Option B, or contact the C-DAC support desk (rudrasupport@cdac.in).

=== "Option B: your own installation"

    Install once, in your `/scratch` area, from a login node (needed if you want to change
    Bhaskera's code):

    ```bash
    ssh -p 4422 <username>@paramrudra.cdacb.in
    cd /scratch/<project>/<username>
    git clone https://github.com/CDACIndia/Bhaskera.git
    cd Bhaskera
    bash setup.sh
    source bhaskera-activate.sh      # in every new shell
    ```

    `setup.sh` detects CUDA (`module load cuda/12.4`), creates a Python 3.11 environment
    with `uv`, installs PyTorch and Bhaskera with the `wandb`, `mlflow` and `qlora` extras,
    builds flash-attention (the slowest step; a failed build is not fatal) and writes
    `bhaskera-activate.sh`, which loads CUDA and activates the environment.
    `BHASKERA_CUDA=12.4 bash setup.sh` forces a CUDA version; `BHASKERA_VENV=/path bash setup.sh`
    puts the environment elsewhere.

    In this case, run the commands below from the `Bhaskera` folder and use
    `scripts/train.sbatch` instead of `$BHASKERA_HOME/scripts/train.sbatch`.

Verify either way:

```bash
python -c "import bhaskera, torch; print('bhaskera OK, torch', torch.__version__)"
bhaskera-preflight --help
```
