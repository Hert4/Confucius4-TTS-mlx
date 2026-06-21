"""GT for T2S prefix: cond_vec/token_ids -> cond_emb, text_emb (torch)."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from confuciustts.cli.inference import ConfuciusTTS
from confuciustts.utils.text_utils import LANGUAGE_TOKEN_MAP

REF = "reference.wav"
TEXT = "Xin chào, đây là bản thử nghiệm giọng nói tiếng Việt trên máy Mac."
OUT = Path("b4"); OUT.mkdir(exist_ok=True)

m = ConfuciusTTS(config_path="config/inference_config.yaml", device="cpu")
text = m.normalizer.normalize(TEXT, language="vi")
wav_16k, _ = m._load_prompt(REF)
cond_vec = m._extract_semantic(wav_16k)
tok = m.tokenizer.encode(f"You are a helpful assistant. {LANGUAGE_TOKEN_MAP['vi']}:{text}", return_tensors="pt")
m.t2s_model.store_conditioning(cond_vec, tok)
np.savez(OUT / "prefix_gt.npz",
         cond_vec=cond_vec.detach().numpy().astype(np.float32),
         token_ids=tok.numpy(),
         cond_emb=m.t2s_model.cached_condition_emb.detach().numpy().astype(np.float32),
         text_emb=m.t2s_model.cached_text_emb.detach().numpy().astype(np.float32))
print("cond_vec", tuple(cond_vec.shape), "tok", tuple(tok.shape),
      "cond_emb", tuple(m.t2s_model.cached_condition_emb.shape),
      "text_emb", tuple(m.t2s_model.cached_text_emb.shape))
print("saved", OUT / "prefix_gt.npz")
