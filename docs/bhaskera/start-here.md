---
icon: material/rocket-launch
---

# Start Here: Your First Run

This page is for someone who has never fine-tuned a model, or has never run one on a cluster. It explains the few words you need, then walks through three situations step by step, all from the command line:

1. **[I have no model and no dataset](#case-1-i-have-no-model-and-no-dataset)**: you want to see it work, or you will use a public model and dataset.
2. **[I bring my own model and my own data](#case-2-i-bring-my-own-model-and-my-own-data)**, tokenised or not.
3. **[I only want to benchmark a model I already have](#case-3-benchmark-a-model-you-already-have)**: no training.

What was run on PARAM Rudra for this page: option B of Case 1 (tokenise and train, end to end, with the small examples shown), the tokenising of CSV, TSV and JSON data, the checks of option A, and the benchmark of Case 3. The 14 B training of option A and the "already tokenised" route in Case 2 follow the same code but were not repeated here. A web page does the same things with buttons ([Home Page](homepage.md)); this page uses the terminal because it shows exactly what happens.

## The words you will meet

| Word | What it means here |
|---|---|
| **Model** | A neural network that already knows a language (Qwen3, Llama, Param). It is a folder of files: settings, a tokenizer and the weights. |
| **Fine-tuning** | Continuing a model's training on your own examples, so that it answers the way you need. |
| **Weights** | The numbers that make up a model. Training changes them. |
| **LoRA / adapter** | A way to fine-tune by training only a small add-on (the *adapter*) on top of the unchanged model. It needs far less memory and disk, and is the usual choice. **QLoRA** does the same on a model squeezed to 4 bits. |
| **Full fine-tuning** | Training every weight. It can learn more and needs far more memory ([Recipes](recipes.md#full-fine-tuning-training-every-weight)). |
| **Dataset** | Your examples: conversations, instructions and answers. |
| **Tokenising** | Turning text into the numbers a model reads. You do it once per dataset and model; every training run reuses the result (a *cache*). |
| **Recipe** | One small YAML file that says what to train, on what, and how. |
| **Step / epoch** | A *step* teaches the model one batch of examples. An *epoch* is one pass over all your data. |
| **Checkpoint** | The saved state of a run, so that one that stops can continue (`resume`). |
| **Benchmark** | A set of questions with known answers, to measure a model. |
| **Slurm job** | How you ask the cluster for GPUs: `./submit.sh` does it for you. A *node* is one machine (2 GPUs on PARAM Rudra). |
| **Login node** | The machine you log in to. It has internet (some of them). **Compute nodes**, where jobs run, have none, so models and data must be on disk first. |

## How a run fits together

```text
 model  ───┐
           ├──►  tokenise  ──►  recipe  ──►  train  ──►  check  ──►  use
 data   ───┘     (once)         (a file)    (GPUs)     (benchmark)   (chat, save)
```

You supply a **model** and **data**. Tokenising turns the data into numbers (a few minutes, on CPUs). A **recipe** names the model, the tokenised data and the settings. `./submit.sh` checks the recipe, asks Slurm for GPUs and starts training. When it ends you have an adapter (or checkpoints), and you can benchmark it, chat with it and save it under a name.

## Before you start

Log in, load Bhaskera and make your **run folder**: the one place where your recipes, logs and checkpoints go.

```bash
$ ssh -p 4422 <user>@paramrudra.cdacb.in
$ module load bhaskera/1.0.0
$ cp -r $BHASKERA_KIT /scratch/<project>/<username>/bhaskera-runs
$ cd /scratch/<project>/<username>/bhaskera-runs
```

Everything below is run from this folder. Jobs are charged to your login name, so there is nothing to set. Use a folder on `/scratch`: checkpoints can be large, and home folders fill up quickly.

!!! tip "Already on PARAM Rudra: models and data are waiting"
    The administrators keep ready-to-use models in `/home/apps/bhaskera/models` (Qwen3-14B, Param-1-2.9B-Instruct, Param-1-7B, Param2-17B, and the multimodal Qwen3-VL and Qwen2.5-Omni) and tokenised data in `/home/apps/bhaskera/data` (`nemotron-chat_<model>_seq4096_train` and `_val` for three of the models, with a `README.txt`). You can point a recipe straight at them.

## Case 1: I have no model and no dataset

Two ways. Option A needs no download at all; option B shows how to use any public model and dataset by name.

### Option A: use what is already on the cluster

Copy a ready recipe and point it at the shared model and data:

```bash
$ sed -e 's#/scratch/<project>/<username>/models/Qwen3-14B#/home/apps/bhaskera/models/Qwen3-14B#' \
      -e 's#/scratch/<project>/<username>/cache/REPLACE_local_train_<hash>#/home/apps/bhaskera/data/nemotron-chat_qwen3-14b_seq4096_train#' \
      -e 's#/scratch/<project>/<username>/cache/REPLACE_local_val_<hash>#/home/apps/bhaskera/data/nemotron-chat_qwen3-14b_seq4096_val#' \
      -e "s#/scratch/<project>/<username>/bhaskera-runs#$PWD#" \
      recipes/qwen3-14b-qlora.yaml > recipes/my-first-run.yaml
```

Check it first. This runs every check and submits nothing:

```bash
$ DRY_RUN=1 ./submit.sh recipes/my-first-run.yaml 1 1
== preflight (login node, a few seconds) ==
[PASS] model weights              8 safetensors file(s) in /home/apps/bhaskera/models/Qwen3-14B
[PASS] tokenizer                  Qwen2TokenizerFast, vocab 151669
...
[PASS] est. per-GPU @ 2 GPUs      17.6 GiB of 80.0 GiB
RESULT: OK
dry run: every check passed; nothing was submitted
```

Then run it. The arguments are the recipe, the **number of nodes** and the **hours**. The recipe trains for only 10 steps, so a first run is over in minutes:

```bash
$ ./submit.sh recipes/my-first-run.yaml 1 1
submitted job <id>: my-first-run on 1 nodes (2 GPUs), gpu-debug, 1 h
$ tail -f logs/my-first-run_<id>.out | grep --line-buffered -E '\[step|eval|exit code'
```

You will see one line per step with the loss (how wrong the model still is: it should fall), and the run ends with `=== training exit code: 0 ===`. See [Monitoring](monitoring.md) for what each number means.

### Option B: a public model and dataset, by name

Here the model and the dataset live on Hugging Face. You do not download them yourself: name them, and Bhaskera fetches them first. This example uses a small model (Qwen3-0.6B) and the public Alpaca instruction dataset, so it is quick.

**1. Tokenise.** Save this as `recipes/alpaca-tokenize.yaml`:

```yaml
model:
  name: Qwen/Qwen3-0.6B                     # a Hugging Face model, by name
data:
  name: local
  format: alpaca                            # the dataset has instruction / input / output columns
  train_path: hf://tatsu-lab/alpaca         # a Hugging Face dataset, by name
  seq_len: 512                              # longest example, in tokens
  cache_dir: /scratch/<project>/<username>/cache
  num_workers: 8
  tokenize_batch_size: 64
```

```bash
$ ./submit.sh tokenize recipes/alpaca-tokenize.yaml
downloading model Qwen/Qwen3-0.6B (main): 1.4 GB in 10 files -> .../models/Qwen__Qwen3-0.6B
downloading dataset tatsu-lab/alpaca (main): 0.0 GB in 3 files -> .../data/hf/tatsu-lab__alpaca
resolved: model.name -> .../models/Qwen__Qwen3-0.6B; data.train_path -> .../_splits/train
Submitted batch job <id>
```

What happened: `./submit.sh` saw two Hugging Face names, downloaded them into `models/` and `data/hf/` in your run folder (on a login node that has internet; some do not, and this is handled for you, see the note below), and ran a *resolved copy* of your recipe with the real folders in it. Your recipe is unchanged, and a download that finished is not repeated.

The tokenise job takes a few minutes on CPUs (52,000 examples). When it ends, its log prints the cache path:

```bash
$ grep -a "tokenized_path" logs/bhaskera-tokenize_<id>.out
║   tokenized_path: ".../cache/local_train_<hash>"
```

Alpaca has a training part only, so this tokenises training data (52,002 examples) and says that no validation file was set.

**2. Make the training recipe** from a ready one, with the model name, the cache path and a new name:

```bash
$ CACHE=/scratch/<project>/<username>/cache/local_train_<hash>        # the path printed above
$ sed -e 's#/scratch/<project>/<username>/models/Qwen3-14B#Qwen/Qwen3-0.6B#' \
      -e 's#seq_len: 4096#seq_len: 512#' \
      -e "s#/scratch/<project>/<username>/cache/REPLACE_local_train_<hash>#$CACHE#" \
      -e '/val_tokenized_path/d' \
      -e 's#eval_steps: 200#eval_steps: 0#' \
      -e "s#/scratch/<project>/<username>/bhaskera-runs#$PWD#" \
      -e 's#qwen3-14b-qlora#alpaca-small#g' \
      recipes/qwen3-14b-qlora.yaml > recipes/alpaca-small.yaml
```

(The model name in the training recipe is the Hugging Face name again: it was downloaded in step 1 and is found, not downloaded twice. No validation data means `eval_steps: 0`.)

**3. Check, then run:**

```bash
$ DRY_RUN=1 ./submit.sh recipes/alpaca-small.yaml 1 1        # every check; nothing submitted
$ ./submit.sh recipes/alpaca-small.yaml 1 1
submitted job <id>: alpaca-small on 1 nodes (2 GPUs), gpu-debug, 1 h
```

The log shows lines like `[epoch 0][step 8] loss=2.54 ...`, and ends with `=== training exit code: 0 ===`. In this example the ten steps took about seven seconds of training (a 0.6 B model on 2 GPUs); loading the model takes most of the time.

!!! note "No internet on the login node you landed on?"
    On PARAM Rudra only some login nodes (login01, login02, login03, login05) can reach the internet. If you are on another one, downloads run on one that can, over SSH, and say so (`downloading on login01 ...`). If it cannot find one, it tells you what to do. See [Downloads and Dataset Versions](data.md#when-this-machine-has-no-internet).

You can also download by hand: `python -m bhaskera.downloads get --kind model --repo Qwen/Qwen3-0.6B --target <folder>` (`--kind dataset` for data), and look before downloading with `python -m bhaskera.downloads preview ...`.

## Case 2: I bring my own model and my own data

### Your model

A model is a **Hugging Face model folder**: it has a `config.json`, the weight files (`.safetensors`), and the tokenizer files, all in one folder. Put it on `/scratch`, for example `/scratch/<project>/<username>/models/MyModel`, and name that folder in the recipe: `model.name: /scratch/<project>/<username>/models/MyModel`.

- A model that comes with its own Python code (the BharatGen Param models, for example) needs `trust_remote_code: true`.
- Compute nodes have no internet, so the whole folder must be there, tokenizer included.
- Which models work: the ones listed under [Recipes](recipes.md) have been tried. Others may work if they use a standard architecture; the check before a run (`DRY_RUN=1`) tells you if the files, the tokenizer and the memory fit are right.

### Your data, not tokenised yet

Raw data can be a file, a folder or a pattern, in **JSONL, JSON, CSV, TSV, Parquet or Arrow** (optionally `.gz`): see [what formats work](data.md#what-data-formats-work). What matters is the *columns*, set with `data.format`:

| `data.format` | Each row needs | Example |
|---|---|---|
| `chatml` | `messages`: a list of `{role, content}` | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` |
| `alpaca` | `instruction`, `output`, optionally `input` | `{"instruction": "...", "input": "", "output": "..."}` |
| `sharegpt` | `conversations`: a list of `{from, value}` | `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}` |

Put a few hundred examples aside as a validation file; the run uses it to show whether the model is really improving. Then copy the tokenise recipe, fill in your paths, and run it:

```bash
$ cp recipes/tokenize-chatml.yaml recipes/my-tokenize.yaml
$ nano recipes/my-tokenize.yaml          # model.name, data.format, train_path, val_path, seq_len, cache_dir
$ ./submit.sh tokenize recipes/my-tokenize.yaml
```

Before tokenising, `python -m bhaskera.launcher.data_tools inspect <file.jsonl> --format chatml --seq-len 4096` finds empty answers, duplicates and rows that are too long (these helper tools read JSONL; other formats are read directly by the tokenise step). If you have a validation file (`val_path`), you get a second cache; if not, only the training data is tokenised and the log says so.

!!! warning "Tokenised data fits one model"
    The numbers depend on the model's tokenizer and on `seq_len`. A cache made for Qwen3-14B is right for Qwen3 models, not for Llama. Tokenise again for another family.

### Your data, already tokenised

If your data is already tokenised, skip the step above. Bhaskera trains from a **folder of Parquet files** with these columns:

| Column | What |
|---|---|
| `input_ids` | the token numbers |
| `labels` | the same length: the numbers to learn (use -100 for a position that should not count) |
| `attention_mask` | 1 for a real token, 0 for padding. If you have none, set `data.pack_sequences: true` and an all-ones mask is made. |

If your columns have other names, rename them in the recipe with `data.column_map: {your_name: input_ids}`. Point the recipe at the folder:

```yaml
data:
  name: local
  seq_len: 4096                       # the length of your rows: it must match, rows are never re-padded
  tokenized_path: /scratch/<project>/<username>/mydata/train_tokenised
  val_tokenized_path: /scratch/<project>/<username>/mydata/val_tokenised     # optional
```

Write a small `metadata.json` next to the Parquet files if you can (`{"model_name": "<model folder>", "seq_len": 4096}`): without it Bhaskera cannot check the length and tokenizer against your model, and warns instead. Data you tokenised yourself works from the command line; the web page lists only caches that have a `metadata.json`.

### Train

Copy a ready recipe for your model size (see [Recipes](recipes.md)), and set the model, the data and the folder for checkpoints:

```bash
$ cp recipes/qwen3-14b-qlora.yaml recipes/my-run.yaml
$ nano recipes/my-run.yaml        # model.name, tokenized_path, val_tokenized_path, checkpoint.save_dir, logging.run_name, max_steps
$ DRY_RUN=1 ./submit.sh recipes/my-run.yaml 8 12       # every check, nothing submitted: recipe, nodes, hours
$ ./submit.sh recipes/my-run.yaml 8 12
```

`max_steps` is 10 in the ready recipes so that a first job finishes in minutes. For a real run set it from your data size (the comment at the top of each recipe says how). `DRY_RUN` checks the files, the tokenizer, whether the model fits in GPU memory, the partition limits, the disk space for checkpoints and whether your data changed since it was registered.

To train **every weight** instead of a LoRA add-on, see [Full fine-tuning](recipes.md#full-fine-tuning-training-every-weight).

If a job stops half way: `./submit.sh resume --run <run id>` continues from the last checkpoint ([Resuming a Job](resume.md)).

## Case 3: Benchmark a model you already have

You do not need to train anything to measure a model. You need the model folder and the benchmark data.

**1. Prepare the benchmark data once.** It is downloaded into your run folder. Compute nodes have no internet, so do this on a login node that has it (login01, login02, login03 or login05 on PARAM Rudra):

```bash
$ python -m bhaskera.launcher.evaluate prepare --tasks arc_easy,hellaswag,winogrande
downloading arc_easy ...
done: the datasets are in the Hugging Face cache
```

**2. Run the benchmark** on a GPU node:

```bash
$ ./submit.sh eval --model /home/apps/bhaskera/models/Qwen3-14B --tasks arc_easy,hellaswag --limit 200 --out qwen3-14b.json
submitted job <id>: evaluation, gpu-debug, 1 h
$ tail -f logs/bhaskera-eval_<id>.out
```

`--limit 200` uses 200 questions per test for a quick look; leave it out for the full test. `--model` is any model folder (yours, a shared one, or one you downloaded). With no adapter given, only that base model is measured. In a check we ran this way on Qwen3-0.6B with 50 questions of `arc_easy`:

```text
no adapter given: only the base model is evaluated
arc_easy: acc_norm = 0.6000
saved .../case3_arc.json
=== evaluation exit code: 0 ===
```

**3. Compare models.** Run it again for another model with another `--out`, then read both:

```bash
$ ./submit.sh eval --model /home/apps/bhaskera/models/Param-1-7B --tasks arc_easy,hellaswag --limit 200 --out param-1-7b.json
$ python -m bhaskera.launcher.evaluate show qwen3-14b.json
$ python -m bhaskera.launcher.evaluate show param-1-7b.json
```

**4. Test on your own questions.** The standard tests say little about your own use. Put question and answer pairs in a JSONL file (`{"question": "...", "answer": "..."}` per line), turn it into a test, and run it the same way:

```bash
$ python -m bhaskera.launcher.evaluate init-task --name my_test --data /scratch/<project>/<username>/my.jsonl
$ ./submit.sh eval --model <model folder> --tasks my_test --save-samples
$ python -m bhaskera.launcher.evaluate review <samples folder>      # what is right, what is wrong, with the answers
```

The result is saved in the `--out` file. Add `--mlflow` to keep it in MLflow too. Once you have trained a model, `--run <run id>` (or a registered name) benchmarks it *against its base model* in one go ([Home Page](homepage.md#test-the-model-on-your-own-questions-and-find-what-to-correct) and [Model Registry](registry.md)).

## When something goes wrong

| What you see | What to do |
|---|---|
| `RESULT: FAIL` in the dry run | Read the `[FAIL]` line: it names the file, library or setting. Fix it, run the dry run again. |
| The job stays *queued* | The partition is busy or you hit its limit. `squeue -u $USER` shows it, and `./submit.sh` told you the limits before submitting. |
| `not enough space for ...` | A disk is full or nearly full. Choose a folder on a bigger disk, keep fewer checkpoints. |
| "cannot reach the internet" on a download | You are on a login node without internet and none was found to use. See [Troubleshooting](troubleshooting.md). |
| The run stopped half way | `./submit.sh resume --run <run id>`. |
| Anything else | [Troubleshooting & Reference](troubleshooting.md) lists the common errors, and a failed job ends with a plain explanation. |

## Where next

- [Recipes](recipes.md): ready recipes for each model, and how many nodes and hours to ask for.
- [Downloads and Dataset Versions](data.md): data formats, downloading, and naming and versioning your data.
- [Monitoring](monitoring.md): reading the step log, MLflow, and the GPU energy a run used.
- [Home Page](homepage.md): the same things with buttons, plus chat with the model you trained.
- [Model Registry](registry.md): keep a trained model under a name, with a note.
