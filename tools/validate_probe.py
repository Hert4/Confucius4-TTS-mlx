"""Validate one DiT estimator call (MLX) vs torch probe (s2a_gt.npz)."""
import sys
import numpy as np
import mlx.core as mx

sys.path.insert(0, ".")
from confucius_mlx.s2a_mlx import S2AEstimator

gt = np.load("b2/s2a_gt.npz")
est = S2AEstimator()
x0 = mx.array(gt["probe_x0"])          # (1,80,T)
mu = mx.array(gt["mu"])                 # (1,T,512)
t0 = mx.array(gt["probe_t0"])          # (1,)
spks = mx.array(gt["style"])           # (1,192)
cond = mx.array(gt["probe_promptx"])   # (1,80,T)

out = est.forward(x0, mu, t0, spks, cond)
mx.eval(out)
o = np.array(out)
ref = gt["probe_out"]
d = np.abs(o - ref)
print("mlx", o.shape, "torch", ref.shape)
print(f"max abs diff = {d.max():.4e}   mean abs diff = {d.mean():.4e}")
print(f"ref range [{ref.min():.3f},{ref.max():.3f}]  rel = {d.max()/(abs(ref).max()+1e-9):.2e}")
