"""B1 validation: teacher-forcing logits MLX vs torch ground-truth (gt.npz).

Dung prefix embeds (cond_emb,text_emb) tu torch dump de co lap GPT-2 core MLX.
"""
import sys
import numpy as np
import mlx.core as mx

sys.path.insert(0, ".")
from confucius_mlx.t2s_mlx import T2SMLX

gt = np.load("b1/gt.npz")
cond_emb = mx.array(gt["cond_emb"])     # (1,1,D)
text_emb = mx.array(gt["text_emb"])     # (1,Tt,D)
sem_ids = mx.array(gt["sem_ids"])       # (1,1+Tsem) int
S = gt["S"][0]                          # (Tsem,)
P = int(gt["P"][0])
ref_logits = gt["logits"]              # (1, L, 8194) torch

m = T2SMLX()
se = m.semantic_embed(sem_ids)                          # (1,1+Tsem,D)
inputs_embeds = mx.concatenate([cond_emb, text_emb, se], axis=1)
logits = m.logits_from_embeds(inputs_embeds)            # (1, L, 8194)
mx.eval(logits)
lg = np.array(logits)

# numeric diff vs torch
d = np.abs(lg - ref_logits)
print(f"logits shape mlx={lg.shape} torch={ref_logits.shape}")
print(f"max abs diff = {d.max():.4e}   mean abs diff = {d.mean():.4e}")

# token-level: argmax at semantic positions should reproduce S
sem_logits = lg[:, P:, :]
pred = sem_logits[0, :-1].argmax(-1)
match = (pred == S).mean()
print(f"greedy argmax self-match (mlx vs S) = {match:.4f}  ({(pred==S).sum()}/{len(S)})")

# also compare mlx argmax vs torch argmax over all positions
amlx = lg[0].argmax(-1)
ator = ref_logits[0].argmax(-1)
print(f"argmax agreement mlx-vs-torch = {(amlx==ator).mean():.4f}")
