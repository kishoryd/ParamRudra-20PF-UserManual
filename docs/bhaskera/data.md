---
icon: material/database
---

# Data Format

Tokenising reads your raw data in two steps of choice: the **file type** (how the rows are stored) and the **column layout** (`data.format`: what the columns mean). Plain text needs neither.

## 1. File type

Give `train_path` (and `val_path`) as a file, a folder, or a pattern such as `/data/shards/*.csv`. **One path holds one file type** (JSON lines and JSON arrays may mix); add `.gz` to any of them.

| File type | Notes |
|---|---|
| **JSONL** `.jsonl` `.ndjson` | The usual choice and the best for large data. The only type the `data_tools` helpers read. |
| **JSON** `.json` | One row per line, or one array `[{...}]`. An array is read whole into memory, so prefer JSONL when large. |
| **CSV / TSV** `.csv` `.tsv` | Header row; every cell is text. A cell holding a list as JSON text (`messages`, `conversations`) is decoded. |
| **Parquet** `.parquet`, **Arrow / Feather** `.arrow` `.feather` | Read as they are. |
| **Plain text** `.txt` `.text` `.md` | Documents with no columns: see [Plain text](#plain-text). |
| **Hugging Face dataset on disk** (`save_to_disk` folder) | Point at one split's folder. |

## 2. Column layout: `data.format`

| `data.format` | Each row needs | Typical sources |
|---|---|---|
| `chatml` | `messages`: list of `{role, content}` | OpenAI chat files, UltraChat, Nemotron |
| `alpaca` | `instruction`, `output`, optional `input` | Alpaca |
| `sharegpt` | `conversations`: list of `{from, value}` | ShareGPT, Vicuna, OpenHermes |
| `prompt_completion` | `prompt`, `completion` | Q&A sets, OpenAI's older files |
| `dolly` | `instruction`, optional `context`, `response` | Databricks Dolly |
| `qa` | `question`, `answer`, optional `context` | SQuAD, TriviaQA |
| `scored_response` | `prompt`, `response`, `score` | Rated or reward-scored answers |
| `tool_calling` | `messages` with `tool_calls` and `tool` turns, plus `tools` | Glaive, xLAM, Hermes function calling |
| `reasoning` | `messages` with `reasoning_content`, or `prompt` + `reasoning` + `answer` | OpenThoughts, DeepSeek-R1 distills |

Other column names are set under `data.format_options`: `messages_field`, and `<name>_field` for the others (for example `{prompt_field: question, completion_field: answer}`).

!!! tip "Which to choose"
    A chat model: use a chat layout (`chatml`, `sharegpt`, `tool_calling`, `reasoning`), or add `use_chat_template: true` to `dolly`, `qa` or `prompt_completion` so the model's own chat template is used. Otherwise those three write plain labelled text (`### Instruction:`, `Question:` / `Answer:`).

!!! warning "Good to know"
    - **`scored_response`** keeps only rows with `score >= min_score` (`format_options: {min_score: 0.7}`): fine-tuning on the good answers. It is **not** reward-model training or reinforcement learning; ExaAI does neither.
    - **`tool_calling`** needs a model whose chat template supports tools (Qwen3, Llama 3.1+, Mistral and others). If tool schemas differ between rows, store `tools` as JSON text.
    - **`reasoning`** writes `<think>…</think>` before the answer (`think_open` / `think_close` change the markers). Reasoning is long: use a generous `seq_len`, since longer rows are cut.
    - **No `val_path`:** only the training data is tokenised and training runs without an evaluation set.

Verified on PARAM Rudra: the same 2,000 rows as CSV, TSV and a JSON array tokenised to identical caches; tool calling, `prompt_completion` and plain text went through a real tokenise run.

## Plain text

For books, code or articles, leave out `data.format`:

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

- `text_split` sets what one example is: a whole **file** (default), a **paragraph** (blank-line separated) or a **line**.
- `chunk: true` cuts each example into `seq_len` pieces ending in the end-of-text token. Without it a long example is cut after `seq_len` tokens, which for a book loses almost all of it.
- One file is read by one worker, so split a very large file. Files are read as UTF-8. These two keys go in the recipe file; Create recipe does not ask for them.

## Data you tokenised yourself

A folder of Parquet files with `input_ids` and `labels` (and `attention_mask`, or set `data.pack_sequences: true`) works as `data.tokenized_path`, without ExaAI's tokenise step. Rename columns with `data.column_map`. Write a `metadata.json` (`{"model_name": "<model folder>", "seq_len": 4096}`) next to the files: without it ExaAI cannot check that the length and tokenizer match, and the dashboard does not list the folder. Rows are never re-padded, so `data.seq_len` must equal their length.


Getting data in from Hugging Face, and naming and versioning it: [Downloads and Dataset Versions](downloads.md).
