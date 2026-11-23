# Multimodal Fine-tuning

Bhaskera also fine-tunes models that read **images, video and audio** next to text, with the same strategies, recipes
and cluster setup as text models. The output is text.

| Input | Model family | Tested with |
|---|---|---|
| Image | Qwen3-VL (2B to 32B) | Qwen3-VL-2B and 8B |
| Video | Qwen3-VL (same model) | Qwen3-VL-2B |
| Audio | Qwen2.5-Omni (thinker) | Qwen2.5-Omni-3B |

Supported strategies: DDP, FSDP2 and HSDP. The Pipeline strategy, QLoRA and sequence packing are not available for these
models; the run is refused before it starts, with the reason.

## Data

One JSON object per line. `content` is a list of parts, also for plain text:

```json
{"messages": [
  {"role": "user", "content": [
    {"type": "image", "image": "/scratch/<project>/<username>/data/pokemon/00001.png"},
    {"type": "text",  "text": "Describe this Pokemon in one short sentence."}]},
  {"role": "assistant", "content": [{"type": "text", "text": "a green dragon with wings"}]}]}
```

`{"type": "video", "video": path}` and `{"type": "audio", "audio": path}` work the same way. Images, video and audio stay
on disk; only their paths and token counts are cached, and they are decoded while training.

## Run

Start from the multimodal starters in the Bhaskera repository (`configs/multimodal/`): one for DDP, FSDP2 and HSDP, and a
tokenizer config for vision (`tokenize_vision.yaml`, image and video) and one for audio (`tokenize_audio.yaml`).

```bash
$ python -m bhaskera.launcher.tokenize --config configs/multimodal/tokenize_vision.yaml --split both
$ sbatch scripts/submit_multimodal.sh configs/multimodal/fsdp.yaml
```

Put the two printed cache paths into the training config first. Important settings: `data.modalities` (any of `text`,
`image`, `video`, `audio`), `data.image_max_pixels` (about 1 million pixels is at most 1024 visual tokens per image) and
`model.freeze_encoders` (the vision and audio towers are frozen by default). Every setting, and how the encoders are split
across GPUs, is in the guide in the repository: `docs/multimodal/README.md`.
