# Downloads and Dataset Versions

Before a model can be trained, the model and the data have to be on disk (the machines that run training have no internet), and later you should be able to say which data a model was trained on. This page covers getting data in, in the formats you have; naming Hugging Face models and datasets in a recipe; downloading from the terminal or the home page; and giving data names and versions.

## What data formats work

The tokenise step (`./submit.sh tokenize`) reads your raw data in any of these, as a file, a folder of files, or a pattern such as `/data/shards/*.csv`:

| Format | Notes |
|---|---|
| **JSONL** (`.jsonl`, `.ndjson`, one JSON row per line) | The usual choice, and the only one the helper tools `data_tools inspect|dedupe|filter|split` read. Best for very large data. |
| **JSON** (`.json`) | One row per line, or one array `[{...}, {...}]`. An array file is read whole into memory, so prefer JSONL for large data. |
| **CSV / TSV** (`.csv`, `.tsv`) | One row per line with a header. Every cell is read as text, and an empty cell stays empty text. A cell that holds a list as JSON text (the `messages` or `conversations` column of a chat dataset) is decoded. |
| **Parquet** (`.parquet`) | Read as it is. |
| **Arrow / Feather** (`.arrow`, `.feather`) | The Arrow IPC file or stream format. |
| **Plain text** (`.txt`, `.text`, `.md`) | Text with no columns, for continued pre-training or a book or code corpus. See [Plain text](#plain-text) below. |
| **Compressed** (`.gz`) | Any of the above with `.gz` added, for example `data.csv.gz`. |
| **A Hugging Face dataset saved to disk** (a folder from `save_to_disk`) | One split per folder: point `train_path` at the split's folder. |

One path must hold one format (JSON lines and JSON arrays may be mixed). What the *columns* must be is a separate choice, `data.format`. Four layouts are built in, and they cover most public datasets:

| `data.format` | The columns each row needs | Typical sources |
|---|---|---|
| `chatml` | `messages`: a list of `{role, content}` turns (`data.format_options.messages_field` names another column) | OpenAI-style chat files, UltraChat, Nemotron chat, most chat datasets on Hugging Face |
| `alpaca` | `instruction`, `output`, and optionally `input` | Alpaca and its many copies |
| `sharegpt` | `conversations`: a list of `{from, value}` turns | ShareGPT, Vicuna and OpenHermes style files |
| `prompt_completion` | `prompt` and `completion`; `format_options: {prompt_field: ..., completion_field: ...}` names other columns | Question and answer sets (`question`/`answer`), Dolly (`instruction`/`response`), OpenAI's older fine-tuning files |

Text with no columns at all (books, code, articles) needs no format: see [Plain text](#plain-text).

A layout that is none of these can be added without changing Bhaskera: write a small Python function that turns one row into one string, register it with `@register_format("my_format")` (the example is at the top of `bhaskera/data/formats/__init__.py`), and set `data.format: my_format`. This route is not tested on the cluster, because the function must also be importable by the tokenising workers. A few lines of Python or `pandas` that rename the columns into one of the layouts above are enough.

Verified on PARAM Rudra: the same 2,000 instruction rows as CSV, TSV and a JSON array each tokenised to the same cache (2,000 rows at sequence length 512, format `alpaca`).

!!! note "No validation file"
    A recipe without `val_path` tokenises the training data only, and says so. Training then runs without an evaluation set. Add a `val_path` and tokenise again to get one.

### Plain text

For text that is not question and answer, point `train_path` at `.txt` files and leave out `data.format`:

```yaml
data:
  name: local
  train_path: /scratch/<project>/<username>/data/books/      # a folder of .txt files
  val_path:   /scratch/<project>/<username>/data/val.txt
  seq_len: 4096
  format_options:
    text_split: file        # file | paragraphs | lines
    chunk: true
```

- `text_split` says what one training example is: a whole **file** (the default), a **paragraph** (pieces separated by a blank line) or one **line**. Empty pieces are dropped.
- `chunk: true` cuts every example into `seq_len` pieces and ends each with the model's end-of-text token. Without it, an example longer than `seq_len` is cut off and the rest is lost, which for whole books means almost all of the text. Use `chunk: true` for files and long paragraphs.
- Text files are read one file per task, so a large corpus is spread over the workers; a single file is read by one worker, so split a very large file into several. Files are read as UTF-8.
- Set these two keys in the recipe file; the Create recipe dialog does not ask for them.

### Data you tokenised yourself

A folder of Parquet files that already hold token ids works as `data.tokenized_path`, without Bhaskera's tokenise step. Each row needs `input_ids` and `labels`, and `attention_mask` (or set `data.pack_sequences: true` to have an all-ones mask made). Name differently-named columns with `data.column_map`. A `metadata.json` next to the files is optional but worth writing: without it Bhaskera cannot check that `seq_len` and the tokenizer match your model (it warns), and the web page lists only folders that have one.
The rows are used as they are and never re-padded, so `data.seq_len` must equal their length.

## Name a Hugging Face model or dataset in a recipe

Instead of downloading first and pasting a folder, a recipe can name what it needs:

```yaml
model:
  name: Qwen/Qwen3-1.7B                  # a Hugging Face model
data:
  name: local
  format: alpaca
  train_path: hf://tatsu-lab/alpaca      # a Hugging Face dataset; hf://owner/name@revision pins a version
```

`./submit.sh` downloads them first, on a machine that has internet, into `models/` and `data/hf/` of your run folder, and runs a **resolved copy** of the recipe (`.bhaskera/resolved/<recipe>.yaml`) with the real folders in it. Your recipe is not changed. A download that finished before is not repeated. For a dataset that has train and validation (or test) parts, the files are sorted into `_splits/train` and `_splits/val` next to it, and `val_path` is filled in when your recipe has none. `data: {name: owner/dataset}` is the same as `name: local` with that `train_path`.
`DRY_RUN=1 ./submit.sh ...` only says what would be downloaded.

Some login nodes have no internet. This is handled: the download runs on a login node that has it (see below).

## Download from the terminal

```bash
$ python -m bhaskera.downloads preview --kind model --repo Qwen/Qwen3-14B        # size, files, licence; downloads nothing
$ python -m bhaskera.downloads get --kind model --repo Qwen/Qwen3-14B --target /scratch/<project>/<username>/models/Qwen3-14B
$ python -m bhaskera.downloads get --kind dataset --repo tatsu-lab/alpaca --target /scratch/<project>/<username>/data/alpaca
```

`get` can be run again to continue an interrupted download. A model that asks you to accept a licence (Llama, for example) needs `huggingface-cli login`, or **Sign in** on the home page, first.

### When this machine has no internet

On PARAM Rudra only some login nodes can reach the internet (login01, login02, login03 and login05 can; the others cannot), and which one you land on varies. If the machine you are on cannot reach Hugging Face, `get`, `preview`, the home page and `./submit.sh` run the work on one that can, over SSH, and say so (`downloading on login01 ...`). The nodes to use are the `download_hosts` of the cluster profile (or `BHASKERA_DOWNLOAD_HOSTS`, a comma-separated list). If none can be reached, the message names this machine and what to set. The target folder must be on the shared file system, since the download runs on another node. A key that is set only as `HF_TOKEN` in your shell is not passed on to the other node: sign in once (`huggingface-cli login`, or the home page), which keeps the key in a file all nodes can read.

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
```

The list is one small file, `.bhaskera/datasets.json` in your run folder. It needs no server.
