# tools — reproducibility / validation

The MLX port was built **validate-driven**: at each stage the PyTorch model is run
to produce ground-truth tensors, then the MLX implementation is diffed against them.
These scripts document that process.

They require the **original** [Confucius4-TTS](https://github.com/netease-youdao/Confucius4-TTS)
torch repo on `PYTHONPATH` (for the ground-truth dumps), plus `torch` + `transformers`,
and a reference voice clip saved as `reference.wav`. Run from the repo root.

- `dump_t2s_gt.py` / `validate_tf.py` — T2S logits/tokens (MLX vs torch, teacher-forcing)
- `dump_s2a_gt.py` / `validate_probe.py` / `validate_mel.py` — S2A DiT probe + full mel
- `dump_w2vbert.py` — w2v-bert hidden_states[17] ground truth
- `dump_prefix_gt.py` — T2S prefix (text_projector + ECAPA speaker encoder)

Results achieved are summarized in the top-level `README.md`.
