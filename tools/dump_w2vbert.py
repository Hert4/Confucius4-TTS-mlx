"""B4: dump w2v-bert-2.0 weight structure + ground-truth (input_features -> hidden_states[17]).
    cd reference && ../.venv/bin/python dump_w2vbert.py
"""
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torchaudio
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

REF = "reference.wav"
OUT = Path("b4"); OUT.mkdir(parents=True, exist_ok=True)

fe = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval()

# weight structure (encoder layer 0 + projection + a couple layers)
g = defaultdict(list)
for k, v in model.state_dict().items():
    g[re.sub(r'\.\d+\.', '.#.', k)].append(tuple(v.shape))
print("=== w2v-bert weight groups ===")
for k in sorted(g):
    print(f"{k:60s} x{len(g[k]):<3d} {g[k][0]}")

# ground-truth
wav, sr = torchaudio.load(REF)
if wav.shape[0] > 1:
    wav = wav.mean(0, keepdim=True)
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
inp = fe(wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    out = model(input_features=inp["input_features"], attention_mask=inp["attention_mask"],
                output_hidden_states=True)
h17 = out.hidden_states[17]
print("input_features", tuple(inp["input_features"].shape), "h17", tuple(h17.shape))
np.savez(OUT / "w2v_gt.npz",
         input_features=inp["input_features"].numpy().astype(np.float32),
         attention_mask=inp["attention_mask"].numpy(),
         h17=h17.numpy().astype(np.float32))
print("saved", OUT / "w2v_gt.npz")
