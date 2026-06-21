"""B1 ground-truth dumper: chay T2S torch o che do GREEDY tat dinh, dump
prefix embeds + token sequence + teacher-forcing logits de doi chieu voi MLX.

    cd reference && ../.venv/bin/python dump_t2s_gt.py
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from confuciustts.cli.inference import ConfuciusTTS
from confuciustts.utils.text_utils import LANGUAGE_TOKEN_MAP

REF_WAV = "reference.wav"
TEXT = "Xin chào, đây là bản thử nghiệm giọng nói tiếng Việt trên máy Mac."
LANG = "vi"
OUT = Path("b1")
OUT.mkdir(parents=True, exist_ok=True)
BOS, EOS = 8192, 8193

torch.manual_seed(0)
m = ConfuciusTTS(config_path="config/inference_config.yaml", device="cpu")
t2s = m.t2s_model
t2s.eval()

# --- build prefix inputs exactly like _synth_segment ---
text = m.normalizer.normalize(TEXT, language=LANG)
wav_16k, _ = m._load_prompt(REF_WAV)
cond_vec = m._extract_semantic(wav_16k)              # (1, T_feat, 1024) condition_vector
lang_token = LANGUAGE_TOKEN_MAP.get(LANG)
formatted = f"You are a helpful assistant. {lang_token}:{text}"
token_ids = m.tokenizer.encode(formatted, return_tensors="pt")  # (1, T_text)
print("text tokens:", token_ids.shape, "cond_vec:", cond_vec.shape)

# --- GREEDY deterministic generate (no sampling, no rep penalty) ---
with torch.no_grad():
    out = t2s.generate(
        text_inputs=token_ids, condition_vector=cond_vec,
        max_length=512, num_beams=1, do_sample=False,
        repetition_penalty=1.0, return_latent=True,
    )
S = out["semantic_codes"]            # (1, T_sem) generated tokens
latent = out["latent"]               # (1, T_sem, 1280)
print("greedy semantic_codes:", S.shape, "latent:", latent.shape)
print("first 20 tokens:", S[0, :20].tolist())

# cached prefix (set by store_conditioning inside generate)
cond_emb = t2s.cached_condition_emb   # (1,1,1280)
text_emb = t2s.cached_text_emb        # (1,T_text,1280)

# --- teacher-forcing forward over [cond, text, BOS+S] -> logits ---
sem_ids = torch.cat([torch.full((1, 1), BOS, dtype=token_ids.dtype), S], dim=1)  # (1,1+T_sem)
with torch.no_grad():
    se = t2s.semantic_embedding(sem_ids)
    se = t2s.semantic_position_embedding(se)
    inputs_embeds = torch.cat([cond_emb, text_emb, se], dim=1)
    tro = t2s.transformer(inputs_embeds=inputs_embeds,
                          attention_mask=torch.ones(inputs_embeds.shape[:2], dtype=torch.long),
                          use_cache=False, return_dict=True)
    h = t2s.final_norm(tro.last_hidden_state)
    logits = t2s.semantic_head(h)     # (1, P+1+T_sem, 8194)

P = cond_emb.shape[1] + text_emb.shape[1]      # prefix length = 1 + T_text
sem_logits = logits[:, P:, :]                  # logits at semantic positions (predict next)
# sanity: argmax at semantic positions 0..T_sem-1 should reproduce S
pred = sem_logits[0, :-1].argmax(-1)           # predictions after BOS..S[-2]
match = (pred == S[0]).float().mean().item()
print(f"teacher-forcing self-match (argmax==S): {match:.4f}")

np.savez(
    OUT / "gt.npz",
    cond_emb=cond_emb.numpy(),
    text_emb=text_emb.numpy(),
    sem_ids=sem_ids.numpy(),
    S=S.numpy(),
    logits=logits.numpy().astype(np.float32),
    sem_logits=sem_logits.numpy().astype(np.float32),
    P=np.array([P]),
)
print("saved", OUT / "gt.npz", "logits", logits.shape)
