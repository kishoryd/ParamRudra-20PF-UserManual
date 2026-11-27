---
icon: material/restart
---

# When a Job Stops: Resuming

Long jobs stop: a node fails, a job runs out of its time limit, memory runs out, or a loss spike tells you to go back. What you lose
depends on the checkpoints. This page is about losing as little as possible and continuing without retyping anything.

## Checkpoints: the part you set up

A checkpoint is the trained adapter weights, the optimizer state and the position in the learning-rate schedule. Two settings in the
recipe decide what a failure costs:

```yaml
checkpoint:
  save_every_steps: 500      # a checkpoint every 500 steps (the default; 0 saves only at the end of an epoch)
  keep_last_n: 2             # how many to keep; raise it if you want to be able to go further back
```

Choose `save_every_steps` by time, not by steps. With a step of 4.7 s, 200 steps is about 16 minutes of work at risk. The checkpoint of a
LoRA run is under a gigabyte, so saving often is cheap. A full fine-tune checkpoint is tens of gigabytes: save less often there. A
checkpoint is also written when the run ends. A checkpoint that was still being written when the job died is ignored.

## Resuming

From the dashboard, a run that failed, was killed, ran out of time or stopped writing has a **Resume** button. In the terminal:

```bash
./submit.sh resume --run <run id>                  # the same recipe, nodes and time, from the latest checkpoint
./submit.sh resume --run <run id> --step 1000      # from an earlier checkpoint
./submit.sh resume --run <run id> --nodes 8 --hours 6
python -m bhaskera.launcher.resume show --run <run id>      # what the run left behind, without submitting
```

The run id is in MLflow and on the dashboard. ExaAI takes the recipe, the number of nodes and the time limit from the job that stopped,
looks for checkpoints in the folder the run saved to, and submits the job again. The job loads the checkpoint, rebuilds the learning-rate
schedule at that step, and skips the training data that the earlier steps already used.

**Restored:** the adapter weights, the optimizer state, the position in the schedule, and the position in the training data (exactly in the
first epoch; in a later epoch the data restarts at the beginning of that epoch).
**Not restored:** the random state, so dropout differs from what an uninterrupted run would have drawn. The resumed job is a new MLflow run,
tagged `resumed_from` with the run it continues, and its curves continue the step numbers.

## Changing something before you resume

The recipe is read again when the job starts, so edit it first to change what the next steps do: a lower learning rate after a loss
spike, a smaller batch size after running out of memory, a longer time limit. The checkpoint stays what it was. If you change the number of
nodes, the global batch size changes with it (`batch_size × grad_accum × GPUs`); adjust `grad_accum` to keep it.

## Going back: a loss spike

If the loss jumped at step 1800 and never recovered, resume from a checkpoint before it:

```bash
python -m bhaskera.launcher.resume show --run <run id>              # lists the checkpoint steps, e.g. 1000, 1500, 2000
./submit.sh resume --run <run id> --step 1500
```

The checkpoints after step 1500 are renamed (`rolledback-0002000`), not deleted, so nothing is lost and they cannot be mistaken for the
newest. Raise `keep_last_n` beforehand if you want more than the latest two to choose from.

## When a run cannot be resumed

The page and the command say why: the job is still running, no checkpoint was saved (the recipe had `save_every_steps: 0` and the job
stopped before the first epoch ended), the recipe file is gone, or Slurm no longer remembers the job (its accounting keeps jobs for a
limited time). If the recipe's `checkpoint.save_dir` is a relative path, submit from the folder the first job was submitted from, because
a relative folder means a different place from anywhere else. An absolute path avoids the problem.

## What usually goes wrong, and what to do

| What you see | Likely cause | What to do |
|---|---|---|
| The log stops with a node named in the failure summary | A faulty node | Resume: the node is remembered and skipped |
| The state is `TIMEOUT` (MLflow shows *killed*) | The time limit was too short | Resume with `--hours` larger |
| *Out of memory* in the failure summary | Batch too large for the GPU | Lower `batch_size` in the recipe, then resume |
| The loss jumps and stays high | Learning rate too high, or bad data | Resume from before the jump with a lower `lr` |
| Nothing was saved | `save_every_steps: 0` and the job ended mid-epoch | Set it, submit again |
