---
icon: material/package-variant
---

# Saving and Finding Your Models

A trained model is an adapter: a folder of a few hundred megabytes that sits on top of the base model. The **model registry** gives each
adapter you want to keep a name and a version, so that "the one we use" is a name and not a run id you have to remember. It is part of
MLflow, which Bhaskera already writes to, so there is nothing to set up. The Bhaskera dashboard's **Models** section lists every version with its alias and note.

## What gets saved, and when

During training, set `logging.mlflow_log_adapter: true` in the recipe. When the run ends, the adapter is exported in the standard PEFT
format and stored with the MLflow run. Without it there is nothing to register, and neither Chat nor Benchmark can use the run.

To register at the end of training, add `logging.mlflow_register_model: <name>` too. Most people prefer to look at the results first:

## Register after you have looked at the results

On the dashboard press **New run** and choose **Register**: pick the run, give it a name (or pick a name that exists to add a
new version) and, if you like, an alias of your choice and a hand-off note. In the terminal:

```bash
python -m bhaskera.launcher.register add --run <run id> --name my-assistant --alias champion \
    --note "For support questions. Weak on legal text."
python -m bhaskera.launcher.register list
python -m bhaskera.launcher.register show my-assistant:1      # the hand-off note
```

Each version records the run it came from, its base model, the final evaluation and any benchmark results as tags, so the registry shows
what each version is without opening the run. An **alias** is a label you choose (`champion` is only an example; use your own words, such as the stage it is in) that points at one version. When a better one comes along,
register it with `--alias champion` again and the alias moves; the old version stays.

## The hand-off note

Each version also keeps a note for whoever picks it up next: the text you wrote (what it is for, which data, what it gets wrong), followed by what Bhaskera fills in from the run:
the base model, the run, the LoRA settings, steps, learning rate, batch and sequence length, the training data, the results (evaluation loss and accuracy, benchmark scores), and how to use
the adapter. The adapter needs the base model at the same path, so the note says where it was. `register show <name>:<version>` prints it, and the dashboard shows it when you hover over the row.

## Using a registered model

Everywhere a run id is accepted, a registry name works too:

```bash
./submit.sh serve --run my-assistant@champion      # chat with the champion
./submit.sh eval  --run my-assistant:3 --tasks my_test --save-samples
python -m bhaskera.launcher.register resolve my-assistant@champion     # prints the run id behind it
```

`name` means the newest version, `name:3` version 3, and `name@alias` the version the alias points at. To load an adapter in your own code:

```python
from mlflow.tracking import MlflowClient
from peft import PeftModel
client = MlflowClient(tracking_uri="file:///home/<you>/mlflow-runs")
folder = client.download_artifacts(run_id, "adapter")        # the run id from `register resolve`
model = PeftModel.from_pretrained(base_model, folder)
```

## A workflow that works

Train with `mlflow_log_adapter: true`, benchmark it and test it on your own questions (see [Bhaskera Dashboard](homepage.md)), correct the data
where it is wrong, train again, and register the version that is better as `champion`. Anyone on the project then chats with, or builds on,
`my-assistant@champion` and never needs to know which run it was.

The registry is stored in your MLflow folder (`~/mlflow-runs`), so on an account that several people share, everyone sees the same models.
