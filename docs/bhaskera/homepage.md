---
icon: material/view-dashboard
---

# The Home Page: Train, Serve, Evaluate, Monitor

`bhaskera-dashboard ray` serves one web page for everything you run. It lists your jobs and your recent runs, starts new
jobs, and lets you talk to a model you have trained. It listens on the login node, is private to your user, and is opened
from your laptop with the one SSH command described under [Monitoring](monitoring.md#ray-dashboard-charts-and-mlflow-in-one-place).

## Starting the page

From your laptop, in a terminal you leave open (`bhaskera-dashboard ray --info` prints the port; it is 20000 plus your user id modulo 10000):

```bash
$ ssh -p 4422 -L 8265:localhost:<port> <user>@paramrudra.cdacb.in \
    "bash -lc 'module load bhaskera/1.0.0 && cd /scratch/<project>/<username>/bhaskera-runs && bhaskera-dashboard ray'"
```

Then open `http://localhost:8265`. The page finds `submit.sh` in the folder you start it from (your run folder). If you start it from another folder, it uses `~/bhaskera-runs`, which it makes from the shared kit the first time.
Jobs are charged to your login name. `module load` is needed because a login shell started by `ssh host "command"` does not load it for you.

## The pages

The page has five pages, each with its own set of cards. The names across the top are links; point at a name to see a menu of what it holds, or click it to open the page.
Clicking the Bhaskera logo returns to Home.

| Page | What is on it |
|---|---|
| **Home** | Four cards (a multimodal model, Chat, Benchmark, Save a model), your running jobs, and five numbers: running now, training runs, saved models, benchmarked, and the GPU energy used |
| **Train** | Create a recipe (text or multimodal model, prepare your own data, or a custom recipe), train from a recipe, resume a stopped run, your recent runs, and your registered data with its versions |
| **Serve** | Chat with a trained model, running chat servers, and every saved model |
| **Evaluate** | Benchmark one or several models, test your own questions, and compare models side by side |
| **Monitor** | MLflow, the Ray Dashboard while a job runs, cluster limits and node health |

The **New run** button at the top right opens the same dialogs as the cards. **Downloads** in the header fetches a model or a dataset from Hugging Face into a folder you choose, and holds the Hugging Face sign-in (see [Downloads and Dataset Versions](data.md)). **Docs** lists the guides, including the **Guide** that explains the words used on the page.
Your name at the top right opens the account menu, with the Guide and **Log out**. Log out asks first, then stops the page; jobs you already started keep running.

## What it shows

- **Jobs**: state, partition, nodes and GPUs, progress and time left, and the live numbers from the job log (loss,
  evaluation loss and accuracy, MFU, tokens per second, step time, and the GPU energy used so far). The GPU load of each node is one click away. Buttons
  open the Ray Dashboard, the GPU charts, the log, and the run in MLflow.
- **Runs**: your recent MLflow runs that Bhaskera made, including finished and failed ones. Runs that other tools wrote into the same MLflow store are not listed. A run that stopped writing is marked *stalled*.
  Runs that were benchmarked show the result, and runs that recorded GPU power show the GPU energy they used (see [Monitoring](monitoring.md#gpu-energy-used)).
- **Cluster limits and node health** (on Monitor): your job slots in each partition, and any node a job found faulty.
  New jobs keep away from those nodes for three days.

## Starting things

Every card and menu entry opens one of these dialogs:

| | You choose | What runs |
|---|---|---|
| Create a recipe | the model, the data, a method, how long | writes `recipes/<name>.yaml` (nothing runs) |
| Train | a recipe, the number of nodes, optionally the hours | `./submit.sh <recipe> <nodes> <hours>` |
| Resume | a run that failed or was killed, and the step to continue from | `./submit.sh resume --run <run id>` |
| Chat | a trained run, or a registered model | `./submit.sh serve --run <run id>` |
| Benchmark | one or more trained runs, the tests | `./submit.sh eval --run <run id> --tasks ...` |
| Register | a trained run, a name, an optional alias and hand-off note | `python -m bhaskera.launcher.register add ...` (no Slurm job: it takes a second) |

**Check first** runs the partition limits and the recipe check and submits nothing. **Start** in the dialog submits (training and resuming ask once more before they use GPU hours). The page runs the
same `submit.sh` you would run by hand, so every check still applies, and none can be skipped from the page. A recipe that
still has `<project>/<username>` placeholders is listed but cannot be started. A recipe that uses HSDP or Pipeline is refused, with the reason, when its
shard size or stages do not divide the GPUs of the nodes you chose.

## Create a recipe

A recipe is the saved list of choices for one training run: which model to teach, which examples to teach it with, and for how long. The page asks
for what you know and fills in the rest.

1. **The model.** A name for the recipe, text or multimodal (then also what the data holds: image, video, audio), and the method:
   QLoRA + DDP (the usual start for text models), DDP, FSDP2, HSDP, Pipeline, or **Custom**. QLoRA and Pipeline are text only, and the page does not offer them for
   multimodal models. The model is a Hugging Face folder that is already on disk.
2. **The data.** Tokenised data already made is found by itself (in the folders your recipes point to and in `cache` next to `submit.sh`) and listed with its model,
   sequence length and number of rows. If you only have JSONL files, open **My data is not tokenised yet**, give the files, and press **Prepare data**: a short CPU job
   tokenises them once, and the result shows up in the list.
3. **The training.** Steps, learning rate and LoRA rank are one click each; *More options* has batch size, gradient accumulation, how often to save a checkpoint, how often to
   evaluate and, for HSDP and Pipeline, the shard group size or the stages.

The recipe is checked with the training job's own loader before it is written, so it loads without warnings, and **Check first** shows the YAML without saving it. A recipe never overwrites
an existing file. Training and resuming then work from the **Train** tab.

**All settings** lists every option a recipe has (about 160), grouped as in the config file, searchable, each with what it does. Anything you type there wins over the choices above. A name that is not a
recipe option is refused with the closest real name.

**Custom** is for people who already know the settings: paste or write a whole recipe as YAML, or load one of the starters shipped with Bhaskera (the files in `configs/`) or one of your recipes and edit it.
A copy is saved under your name with its own run name and checkpoint folder, so it never writes over the original run's checkpoints. The Ray Dashboard host stays on the local machine whatever the YAML says.

## Chat with a trained model

Train with `logging.mlflow_log_adapter: true` so the adapter is stored with the run. Then press **Chat** on a finished run.
A GPU node loads the base model and the adapter, which takes a few minutes. When it is ready, a chat window opens on the page:

- **Tuned** answers with the fine-tuned model, **Base** with the original model, **Compare** shows both side by side. The base
  answer comes from the same weights with the adapter switched off, so nothing is loaded twice.
- *Options* sets the creativity and the longest answer.

If the model reads images, video or audio (a multimodal run), the chat window has an **Attach** button: pick files, check the previews, and ask.
The files stay in the conversation, so you can ask follow-up questions about them.

The **Close server** button in the chat window (or on the job's card) stops the server and frees its GPU node. It asks for a second click, so a stray
click does nothing.

The server runs on the node, stops after two hours (`SERVE_HOURS` changes it) or when you cancel the job, and answers only you.

## Benchmark a trained model

Training loss says little about what the model can still do. **Benchmark** runs standard tests
([lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): ARC, HellaSwag, MMLU, GSM8K and more) on the
base model and on the tuned model, and shows the change per task. A change counts as *better* or *worse* only when it is
larger than twice the standard error; otherwise it reads "no clear change". The result is stored with the training run and
shown in the **Runs** table.

Compute nodes have no internet, so download the test data once on a login node:

```bash
$ python -m bhaskera.launcher.evaluate prepare --tasks arc_easy,hellaswag
```

From the terminal: `./submit.sh eval --run <run id> --tasks arc_easy,hellaswag --limit 200` (`--limit` uses 200 examples per
task for a quick check; leave it out for the real score).

On the page you can tick **several models** (your runs, or registered names such as `my-assistant:2`): each gets its own job. *More options* sets how many examples are
shown in each question (few-shot) and can skip the base model when you only need the tuned score. Your own tests (below) are ticked next to the standard ones.

### Compare models

Once runs have been benchmarked, **Compare models** appears under the Runs table. Tick the models to put side by side: for each test, every model's score, with the score of
the base model it started from in brackets, and the best in each row in bold. Use the same tests and the same number of examples for every model, and note that models with different
base models are comparable only on the same test.

## The same things from the terminal

Everything the page does is also a plain command, so you can use either, or both, on the same jobs.

| What | On the page | In the terminal |
|---|---|---|
| Train a model | **New run**, Train | `./submit.sh recipes/<recipe>.yaml <nodes> <hours>` |
| Check without submitting | **Check first** | `DRY_RUN=1 ./submit.sh recipes/<recipe>.yaml <nodes> <hours>` |
| Chat with a trained model | **Chat** on a run | `./submit.sh serve --run <run id>`, then the steps below |
| Benchmark a trained model | **Benchmark** on a run | `./submit.sh eval --run <run id> --tasks arc_easy,hellaswag --limit 200 --save-samples` |
| Watch a job | the job card and **Logs** | `squeue -u $USER` and `tail -f logs/<name>_<job>.out` |
| Make a recipe | **New run**, Create recipe | write `recipes/<name>.yaml` (see [Recipes](recipes.md)) |
| Save a trained model under a name | **Register** | `python -m bhaskera.launcher.register add --run <run id> --name <model> --alias champion` (see [Model Registry](registry.md)) |
| Continue a run that stopped | **Resume** on the run | `./submit.sh resume --run <run id>` (see [Resuming a Job](resume.md)) |
| Stop a job | **Cancel job** on its card (two clicks); **Close server** for a chat server | `scancel <job>` |
| Look at runs and curves | the **Runs** table, **MLflow** | `bhaskera-dashboard start` |

### Chat from the terminal

```bash
./submit.sh serve --run <run id>                 # prints the job number
tail -f logs/bhaskera-serve_<job>.out            # wait for: === serving on <node>:<port> ===
ssh -N -L <port>:localhost:<port> <node> &       # on the login node: a link to the model
curl -s localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"messages": [{"role": "user", "content": "What is HPC?"}], "max_tokens": 200}'
```

The answer is in `choices[0].message.content` of the reply. Add `"use_adapter": false` to get the base model's answer from the same
weights. The server answers only you and stops after the hours you gave (`SERVE_HOURS`, default 2) or with `scancel <job>`. On the page
you skip all of this: the chat window opens by itself when the model is loaded.

## Test the model on your own questions, and find what to correct

The standard benchmarks say little about your own use. Test the model on examples from your own work:

1. Put them in a JSONL file on `/scratch`, one per line: `{"question": "...", "answer": "..."}`.
2. Turn the file into a test (it goes into a `tasks` folder next to `submit.sh`, which Bhaskera picks up by itself):
   `python -m bhaskera.launcher.evaluate init-task --name my_test --data /scratch/<project>/<username>/my.jsonl`
3. Run it. On the page, open **Benchmark**, then *Add a test of your own* (the file, the names of the question and answer fields): it runs the command above for you and ticks the test. In the terminal:
   `./submit.sh eval --run <run id> --tasks my_test --save-samples`
4. Review the mistakes: `python -m bhaskera.launcher.evaluate review <folder>`. It sorts every example into **fixed by tuning**, **broken by
   tuning** (the base model had it right), **still wrong** and **still right**, prints a few of each, and writes `review_my_test.jsonl` with
   the prompt, the expected answer and both models' responses. The same files are stored with the training run in MLflow, under
   `lm_eval/samples`.
5. Correct the data. *Still wrong* usually means examples are missing or wrong in your training data; *broken* means the tuning was too narrow
   or ran too long. Fix or add examples, tokenise again, train again, and run the same test: the numbers show whether it helped.

The model is asked `Question: ... Answer:`, and the test passes an example when the first line of its reply contains the expected answer, ignoring case and punctuation. For answers
that can be worded many ways, change `metric_list` in the generated file (the
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) documents the options), or write the test as multiple choice.
During training, `logging.mlflow_eval_samples: <n>` also stores example predictions on your validation split at every evaluation.

## Models

The **Models** section (on the Serve page) lists every registered model version: the model, its version, the run it came from, its aliases and its evaluation loss, with **Chat** and **Benchmark**
buttons. Only versions that Bhaskera registered are listed. It is read from the registry directly, so it works with no job running and does not need MLflow's own page. A pencil next to the name means a hand-off note is kept with the
version; hover over the row to read it. See [Saving and Finding Your Models](registry.md).

## What "job slots" means

The page shows, for example, *0 of 4 job slots used in gpu-debug*. This is a cluster rule, not a Bhaskera setting: the batch system allows
each user a fixed number of jobs submitted at once in a partition (on PARAM Rudra, `gpu-debug` allows 4 submitted and 2 running, at most
1 hour each). Bhaskera reads the limit from Slurm, shows how much of it you use, and checks it before submitting, so a job that would be
refused is stopped with a clear message. On an account that several people share, everyone's jobs count together.

!!! note "Shared accounts"
    The page lists the jobs of the Unix account you are logged in with. On an account that several people share, you see
    everyone's jobs, and anyone on it can start jobs from the page.
