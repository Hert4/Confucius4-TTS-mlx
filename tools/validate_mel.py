"""Validate full S2A mel (MLX solve_euler) vs torch (s2a_gt.npz)."""
import sys, time
import numpy as np
import mlx.core as mx

sys.path.insert(0, ".")
from confucius_mlx.s2a_mlx import S2AEstimator

gt = np.load("b2/s2a_gt.npz")
est = S2AEstimator()
z = mx.array(gt["z"]); prompt = mx.array(gt["prompt"]); mu = mx.array(gt["mu"])
spks = mx.array(gt["style"]); t_span = mx.array(gt["t_span"])
T_ref = int(gt["T_ref"][0])

t0 = time.time()
full = est.solve_euler(z, prompt, mu, spks, t_span, cfg=0.7)
mx.eval(full)
dt = time.time() - t0
mel = np.array(full)[:, :, T_ref:]
ref = gt["mel"]
d = np.abs(mel - ref)
print(f"mel mlx {mel.shape} torch {ref.shape}  ({dt:.1f}s, 25 steps)")
print(f"max abs diff = {d.max():.4e}   mean abs diff = {d.mean():.4e}")
print(f"ref range [{ref.min():.3f},{ref.max():.3f}]  rel = {d.max()/(abs(ref).max()+1e-9):.2e}")
