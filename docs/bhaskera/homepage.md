---
icon: material/view-dashboard
---

# The Home Page: Jobs, Start, Chat

`bhaskera-dashboard ray` serves one web page for everything you run. It lists your jobs and your recent runs, starts new
jobs, and lets you talk to a model you have trained. It listens on the login node, is private to your user, and is opened
from your laptop with the one SSH command described under [Monitoring](monitoring.md#ray-dashboard-charts-and-mlflow-in-one-place).

## What it shows

- **Jobs**: state, partition, nodes and GPUs, progress and time left, and the live numbers from the job log (loss,
  evaluation loss and accuracy, MFU, tokens per second, step time). The GPU load of each node is one click away. Buttons
  open the Ray Dashboard, the GPU charts, the log, and the run in MLflow.
- **Runs**: your recent MLflow runs, including finished and failed ones. A run that stopped writing is marked *stalled*.
  Runs that were benchmarked show the result.
- **Cluster limits and node health** (at the bottom): your job slots in each partition, and any node a job found faulty.
  New jobs keep away from those nodes for three days.

## Start a job without the terminal

Press **Start a job**. Pick what to start:

| | You choose | What runs |
|---|---|---|
| Train a model | a recipe, the number of nodes, optionally the hours | `./submit.sh <recipe> <nodes> <hours>` |
| Chat with a trained model | a trained run | `./submit.sh serve --run <run id>` |
| Benchmark a trained model | a trained run, the benchmarks | `./submit.sh eval --run <run id> --tasks ...` |
| Resume a stopped run | a run that failed or was killed, and the step to continue from | `./submit.sh resume --run <run id>` |
| Register a model | a trained run, a name, an optional alias | `python -m bhaskera.launcher.register add ...` (no Slurm job: it takes a second) |

**Check first** runs the partition limits and the recipe check and submits nothing. **Start** submits (training and resuming ask once more before they use GPU hours). The page runs the
same `submit.sh` you would run by hand, so every check still applies, and none can be skipped from the page. A recipe that
still has `<project>/<username>` placeholders is listed but cannot be started.

The page needs to know your account and where `submit.sh` is. Start it from your runs folder, with the account set:

```bash
$ ssh -p 4422 -L 8265:localhost:<port> <user>@paramrudra.cdacb.in \
    "bash -lc 'module load bhaskera/1.0.0 && cd /scratch/<project>/<username>/bhaskera-runs && SLURM_ACCOUNT=<account> bhaskera-dashboard ray'"
```

(`bhaskera-dashboard ray` prints the first part of this command with your port.) Without the account and folder the page
still works; the Start button then says what is missing.

## Chat with a trained model

Train with `logging.mlflow_log_adapter: true` so the adapter is stored with the run. Then press **Chat** on a finished run.
A GPU node loads the base model and the adapter, which takes a few minutes. When it is ready, a chat window opens on the page:

- **Tuned** answers with the fine-tuned model, **Base** with the original model, **Compare** shows both side by side. The base
  answer comes from the same weights with the adapter switched off, so nothing is loaded twice.
- *Options* sets the creativity and the longest answer.

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

## The same things from the terminal

Everything the page does is also a plain command, so you can use either, or both, on the same jobs.

| What | On the page | In the terminal |
|---|---|---|
| Train a model | **Start a job**, Train a model | `./submit.sh recipes/<recipe>.yaml <nodes> <hours>` |
| Check without submitting | **Check first** | `DRY_RUN=1 ./submit.sh recipes/<recipe>.yaml <nodes> <hours>` |
| Chat with a trained model | **Chat** on a run | `./submit.sh serve --run <run id>`, then the steps below |
| Benchmark a trained model | **Benchmark** on a run | `./submit.sh eval --run <run id> --tasks arc_easy,hellaswag --limit 200 --save-samples` |
| Watch a job | the job card and **Logs** | `squeue -u $USER` and `tail -f logs/<name>_<job>.out` |
| Save a trained model under a name | **Register a model** | `python -m bhaskera.launcher.register add --run <run id> --name <model> --alias champion` (see [Model Registry](registry.md)) |
| Continue a run that stopped | **Resume** on the run | `./submit.sh resume --run <run id>` (see [Resuming a Job](resume.md)) |
| Stop a job | not on the page yet | `scancel <job>` |
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
3. Run it. On the page, press **Benchmark** and type `my_test` under *other test names*. In the terminal:
   `./submit.sh eval --run <run id> --tasks my_test --save-samples`
4. Review the mistakes: `python -m bhaskera.launcher.evaluate review <folder>`. It sorts every example into **fixed by tuning**, **broken by
   tuning** (the base model had it right), **still wrong** and **still right**, prints a few of each, and writes `review_my_test.jsonl` with
   the prompt, the expected answer and both models' responses. The same files are stored with the training run in MLflow, under
   `lm_eval/samples`.
5. Correct the data. *Still wrong* usually means examples are missing or wrong in your training data; *broken* means the tuning was too narrow
   or ran too long. Fix or add examples, tokenise again, train again, and run the same test: the numbers show whether it helped.

The test above passes an example when the first line of the answer matches the expected one, ignoring case and punctuation. For answers
that can be worded many ways, change `metric_list` in the generated file (the
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) documents the options), or write the test as multiple choice.
During training, `logging.mlflow_eval_samples: <n>` also stores example predictions on your validation split at every evaluation.

## What "job slots" means

The page shows, for example, *0 of 4 job slots used in gpu-debug*. This is a cluster rule, not a Bhaskera setting: the batch system allows
each user a fixed number of jobs submitted at once in a partition (on PARAM Rudra, `gpu-debug` allows 4 submitted and 2 running, at most
1 hour each). Bhaskera reads the limit from Slurm, shows how much of it you use, and checks it before submitting, so a job that would be
refused is stopped with a clear message. On an account that several people share, everyone's jobs count together.

!!! note "Shared accounts"
    The page lists the jobs of the Unix account you are logged in with. On an account that several people share, you see
    everyone's jobs, and anyone on it can use the Start button.
