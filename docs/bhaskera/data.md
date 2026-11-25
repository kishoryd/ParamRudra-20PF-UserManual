---
icon: material/database-arrow-down
---

# Downloads and Dataset Versions

Two things happen before a model can be trained: the model and the data have to be on disk (the machines that run training have no internet), and you should be able to say later which data a model was trained on.
The home page covers both, under **Downloads** in the header and the **Data** section of the Train page.

## Download a model or a dataset from Hugging Face

Open **Downloads** in the header and choose *Download a model* or *Download a dataset*.

1. Type the repository as it appears on huggingface.co, for example `Qwen/Qwen3-14B`. A revision (a branch, tag or commit) is optional.
2. Type the folder to save into: a full path you can write to, for example `/scratch/<project>/<username>/models/Qwen3-14B`. It is made if it does not exist.
3. **Check size** shows the total size, the number of files, the licence and the largest files, and whether the repository asks for access. Nothing is downloaded yet.
4. **Download** starts it. A bar shows progress and **Cancel** stops it. Start the same download again and it continues where it stopped: finished files are kept.

What it does when it finishes:

- A **model** appears in the model lists of the recipe form, under *Yours*.
- A **dataset** is registered as a dataset version (below), with the repository and revision as its source, so you can always see where it came from.

The download runs on the machine that serves the page (a login node), as you, and writes with your permissions and your disk quota. Only Hugging Face is contacted; a web address cannot be typed in. System folders are refused.
A download above 50 GB asks you to confirm, and the free space in the folder is checked first. One download runs at a time. On a node with no internet, the page says so instead of waiting.

### Signing in to Hugging Face

Most models need no key. A key is needed for private repositories and for models that ask you to accept a licence first (Llama, for example):

1. Create a key with *read* access at huggingface.co/settings/tokens, and accept the licence on the model's page.
2. In **Downloads**, paste the key under *Hugging Face access* and press **Sign in**. Bhaskera checks it with Hugging Face and shows your Hugging Face name.

The key is kept in a file readable by you only (`<run folder>/.cache/huggingface/token`, the place `huggingface-cli login` uses), and is never shown again, written to a log or put in a command line; a download gets it through its environment. **Sign out** deletes the file.
A key set in the environment (`HF_TOKEN`) is used too, and Sign out does not remove that.

## Dataset versions

A dataset version is a name, a number and a fingerprint of the content. It answers "which data made this model?" exactly, even after the files were edited.

In **Data** on the Train page, press **Register data**, give a name, and choose a tokenised cache folder or your raw data file or folder. Bhaskera reads the files once (in the background, with progress) and records the fingerprint, the number of rows, the sequence length and tokenizer (for a tokenised cache), the size, the source and an optional note.

- The same content again keeps its version. Changed content becomes the next number (**New version** on the row prefills the dialog). Old versions stay.
- A version is **unchanged**, **changed** (the files differ from what was registered) or **folder missing**. This is checked without reading the files, and **Verify** reads them again and compares the fingerprint.
- When you train, the run records the version of its data (`dataset.name` and `dataset.version` on the run in MLflow). The runs table shows it next to the run name.
- **Training on data that changed after it was registered is refused**, with the name and number it was registered as, until you register it again as a new version (or restore the files). A recipe whose data was never registered is not affected.
- **Forget** removes one version from the list. The files are never touched.
- Runs made before this have no version and show none. Nothing is guessed for them.

From a terminal:

```bash
$ python -m bhaskera.dataversions add --name support --path /scratch/<project>/<username>/data/train.jsonl --note "first cut"
$ python -m bhaskera.dataversions list
$ python -m bhaskera.dataversions verify support          # or support:2
$ python -m bhaskera.downloads run --kind model --repo Qwen/Qwen3-14B --target /scratch/<project>/<username>/models/Qwen3-14B
```

The list is one small file, `.bhaskera/datasets.json` in your run folder. It needs no server.
