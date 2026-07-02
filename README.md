# Confucius4-TTS-mlx

An [MLX](https://github.com/ml-explore/mlx) port of
[netease-youdao/Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS) —
a multilingual, cross-lingual, **zero-shot voice-cloning** TTS (14 languages incl.
Vietnamese) — running on **Apple Silicon**. The official model is CUDA-only; this
re-implements the pipeline in MLX (torch-free inference) and was validated
numerically against the original PyTorch model at every stage.

> The model has been integrated into mlx-audio, so you can use it there ! 

## Pipeline

```
ref.wav ──► w2v-bert-2.0 (semantic) + CAMPPlus (speaker) + mel  [frontend]
text ─────► T2S  GPT-2 autoregressive decoder (KV-cached)
            S2A  conditional flow-matching (DiT + WaveNet, 25-step Euler + CFG)
            BigVGAN v2 vocoder (anti-aliased snakebeta)  ──► 22.05 kHz wav
```

All stages run in MLX. `torch` is used **only** in `confucius4/convert.py` (weight
conversion), never at inference. Frontend DSP (SeamlessM4T-style fbank, mel) is numpy.

## Validation (MLX vs original PyTorch)

| Stage | metric |
|---|---|
| T2S | argmax agreement 99.2% (teacher-forcing) |
| S2A mel | rel. error 0.77% |
| BigVGAN waveform | correlation 0.9998 |
| w2v-bert hidden[17] | mean abs 0.003 |
| ECAPA / CAMPPlus speaker | rel 5e-4 / cos 0.997 |

Reproducibility scripts in [`tools/`](tools/).

## Benchmark (Apple M5, 24 GB)

~3.8 s of audio end-to-end in ~8 s (RTF ~2.1) fp32; ~6 s (RTF ~1.7) int8.

## Weights

Prebuilt MLX weights on the Hub:
- fp32: [`beyoru/Confucius4-TTS-mlx`](https://huggingface.co/beyoru/Confucius4-TTS-mlx)
- int8 (recommended, ~2.8 GB): [`beyoru/Confucius4-TTS-mlx-int8`](https://huggingface.co/beyoru/Confucius4-TTS-mlx-int8)

Or build them yourself (needs `torch`/`transformers`, conversion only):

```bash
python -m confucius4.convert --out ./confucius4-model               # fp32
python -m confucius4.convert --out ./confucius4-int8 --quantize int8 # int8 (recommended)
```

## Usage

The `confucius4` package is an [mlx-audio](https://github.com/Blaizzy/mlx-audio)
model. Until PR #799 merges, install mlx-audio from the PR branch:

```bash
pip install "git+https://github.com/Hert4/mlx-audio.git@add-confucius4-tts"
```

```python
from mlx_audio.tts.utils import load
model = load("beyoru/Confucius4-TTS-mlx-int8")
for r in model.generate("Xin chào", ref_audio="voice.wav", lang="vi"):
    ...  # r.audio at 22050 Hz
```

## Quantization

8-bit (group 64) on the T2S body matmuls and the w2v-bert encoder linears;
`semantic_head` + norms + embeddings stay fp32 (8-bit on the head audibly degrades
pronunciation). int8 ≈ half the RAM, ~25% faster, quality preserved. int4 is
available (`--quantize int4`) but not recommended — negligible gain over int8 here
(the text-embedding table dominates the fp32 remainder; decode is bound by S2A +
vocoder, not the T2S matmuls).

## Attribution & license

- Model & architecture: [netease-youdao/Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS), Apache-2.0
- Vocoder: [NVIDIA BigVGAN v2](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x)
- Speaker encoder: 3D-Speaker CAMPPlus ([funasr](https://huggingface.co/funasr/campplus))
- MLX port by [Hert4](https://github.com/Hert4), released under Apache-2.0 (see `LICENSE`).
