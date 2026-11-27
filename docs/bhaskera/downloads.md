---
icon: material/cloud-download
---

# Downloads and Dataset Versions

This page covers getting data from Hugging Face (named in a recipe, from the terminal, or from the dashboard) and naming and versioning your data so you can say **which data made which model**. For what the data should look like, see [Data Format](data.md).

## Name Hugging Face repositories in a recipe

```yaml
model:
  name: Qwen/Qwen3-1.7B                  # a Hugging Face model
data:
  name: local
  format: alpaca
  train_path: hf://tatsu-lab/alpaca      # hf://owner/name@revision pins a version
```

`./submit.sh` downloads them first (on a machine with internet) into `models/` and `data/hf/` of your run folder, then runs a **resolved copy** of the recipe (`.bhaskera/resolved/<recipe>.yaml`). Your recipe is not changed and a finished download is not repeated. Train and validation (or test) parts are sorted into `_splits/train` and `_splits/val`, and `val_path` is filled in if you gave none. `DRY_RUN=1 ./submit.sh ...` only reports what would be downloaded.

## From the terminal

```bash
$ python -m bhaskera.downloads preview --kind model --repo Qwen/Qwen3-14B       # size, files, licence; downloads nothing
$ python -m bhaskera.downloads get --kind model --repo Qwen/Qwen3-14B --target /scratch/<project>/<username>/models/Qwen3-14B
$ python -m bhaskera.downloads get --kind dataset --repo tatsu-lab/alpaca --target /scratch/<project>/<username>/data/alpaca
```

Run `get` again to continue an interrupted download. A model that asks you to accept a licence (Llama, for example) needs a Hugging Face key: `huggingface-cli login`, or **Sign in** on the dashboard.

!!! info "A login node with no internet"
    On PARAM Rudra only login01, login02, login03 and login05 reach the internet, and you cannot choose which node you land on. If yours cannot, `get`, `preview`, the dashboard and `./submit.sh` run the work on one that can, over SSH, and say so (`downloading on login01 ...`). The nodes come from `download_hosts` in the cluster profile (or `BHASKERA_DOWNLOAD_HOSTS`, comma-separated). The target folder must be on the shared file system. A key set only as `HF_TOKEN` in your shell is not passed on: sign in once so the key is kept in a file every node can read. To work on login01 yourself, see [Monitoring](monitoring.md#land-on-a-login-node-you-choose-login01-has-internet).

## From the dashboard

Open **Downloads** in the header, choose *Download a model* or *Download a dataset*, then:

1. Type the repository as on huggingface.co (for example `Qwen/Qwen3-14B`); a revision is optional.
2. Type a folder you can write to; it is created if missing.
3. **Check size** shows the size, number of files, licence and whether access is gated. Nothing is downloaded yet.
4. **Download** starts it, with a progress bar and **Cancel**. Starting it again continues where it stopped.

A finished **model** appears under *Yours* in the recipe form. A finished **dataset** is registered as a dataset version with its repository and revision as the source. Only Hugging Face is contacted, system folders are refused, a download above 50 GB asks you to confirm, free space is checked first, and one download runs at a time.

**Signing in.** Most models need no key. Private repositories and licence-gated models (Llama) do: create a *read* key at huggingface.co/settings/tokens, accept the licence on the model page, then paste the key under *Hugging Face access* in **Downloads** and press **Sign in**. It is kept in a file only you can read (`<run folder>/.cache/huggingface/token`), never shown again or written to a log or command line. **Sign out** deletes the file (an `HF_TOKEN` in the environment is not touched).

## Dataset versions

A dataset version is a **name, a number and a fingerprint of the content**, so you can say exactly which data made a model, even after the files were edited.

In **Data** on the Train page press **Register data**, give a name, and choose a tokenised cache or your raw data. ExaAI reads the files once (in the background) and records the fingerprint, row count, size, source, an optional note and, for a tokenised cache, the sequence length and tokenizer.

- Same content keeps its version; changed content becomes the next number (**New version** prefills the dialog). Old versions stay.
- A version shows **unchanged**, **changed** or **folder missing**, checked without reading the files; **Verify** reads them and compares the fingerprint.
- A run records its data version (`dataset.name`, `dataset.version` in MLflow), shown beside the run name.
- **Training on data that changed after registering is refused**, naming the version, until you register it as a new version or restore the files. Data never registered is not affected.
- **Forget** removes a version from the list; files are never touched. Runs made before this show no version.

```bash
$ python -m bhaskera.dataversions add --name support --path /scratch/<project>/<username>/data/train.jsonl --note "first cut"
$ python -m bhaskera.dataversions list
$ python -m bhaskera.dataversions verify support          # or support:2
```

The list is one small file, `.bhaskera/datasets.json` in your run folder; no server is needed.
