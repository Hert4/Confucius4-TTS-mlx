"""B2 ground-truth dumper for S2A. Co dinh noise z -> dump mu/prompt/spks/z/mel,
+ 1 probe estimator(DiT) de unit-test rieng phan kho nhat.

    cd reference && ../.venv/bin/python dump_s2a_gt.py
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from confuciustts.cli.inference import ConfuciusTTS
from confuciustts.utils.text_utils import LANGUAGE_TOKEN_MAP

REF = "reference.wav"
TEXT = "Xin chào, đây là bản thử nghiệm giọng nói tiếng Việt trên máy Mac."
LANG = "vi"
OUT = Path("b2"); OUT.mkdir(parents=True, exist_ok=True)
N_STEPS, CFG = 25, 0.7

torch.manual_seed(0)
m = ConfuciusTTS(config_path="config/inference_config.yaml", device="cpu")

text = m.normalizer.normalize(TEXT, language=LANG)
wav_16k, wav_tgt = m._load_prompt(REF)
cond_vec = m._extract_semantic(wav_16k)
style = m._extract_style(wav_16k)
ref_mel = m._ref_mel(wav_tgt)
formatted = f"You are a helpful assistant. {LANGUAGE_TOKEN_MAP[LANG]}:{text}"
tok = m.tokenizer.encode(formatted, return_tensors="pt")
out = m.t2s_model.generate(text_inputs=tok, condition_vector=cond_vec, max_length=512,
                           num_beams=1, do_sample=False, repetition_penalty=1.0, return_latent=True)
codes, latent = out["semantic_codes"], out["latent"]

s2a = m.s2a_model
# ---- replicate inference preamble to build mu ----
with torch.no_grad():
    semantic_emb = s2a.input_embedding(codes).transpose(1, 2)
    text_cond = s2a.encoder_proj(torch.cat([latent, semantic_emb], dim=-1))
    target_len = torch.tensor([int(codes.shape[1] * 1.72)])
    cond_target, _ = s2a.length_regulator(text_cond, target_len)
    T_ref = ref_mel.size(1)
    prompt_condition = s2a.prompt_cond.expand(1, T_ref, -1)
    mu = torch.cat([prompt_condition, cond_target], dim=1)        # (1, Ttot, 512)
    total_len = torch.tensor([T_ref]) + target_len
    prompt = ref_mel.transpose(1, 2).contiguous()                # (1,80,T_ref)

    Ttot = mu.size(1)
    # ---- fixed noise + linear schedule (t_scheduler='linear') ----
    torch.manual_seed(0)
    z = torch.randn(1, 80, Ttot)
    t_span = torch.linspace(0, 1, N_STEPS + 1)

    # ---- DiT probe: one conditional estimator call (step 0 inputs) ----
    prompt_x = torch.zeros_like(z); prompt_x[..., :T_ref] = prompt[..., :T_ref]
    x0 = z.clone(); x0[..., :T_ref] = 0
    mask = (torch.arange(Ttot)[None] < total_len[:, None])        # (1,Ttot) bool
    t0 = t_span[0:1]
    probe = s2a.decoder.estimator(x0, mask, mu, t0, style, prompt_x)   # (1,80,Ttot)

    # ---- full flow via solve_euler (z fixed) ----
    mel_full = s2a.decoder.solve_euler(z.clone(), t_span=t_span, x_lens=total_len,
                                       prompt=prompt, mu=mu, spks=style, cfg_rate=CFG)
    if isinstance(mel_full, (list, tuple)):
        mel_full = mel_full[-1]
    mel = mel_full[:, :, T_ref:]

def npy(t):
    return t.detach().cpu().numpy()

np.savez(OUT / "s2a_gt.npz",
         codes=npy(codes), latent=npy(latent).astype(np.float32),
         ref_mel=npy(ref_mel).astype(np.float32), style=npy(style).astype(np.float32),
         mu=npy(mu).astype(np.float32), prompt=npy(prompt).astype(np.float32),
         z=npy(z).astype(np.float32), t_span=npy(t_span).astype(np.float32),
         total_len=npy(total_len), T_ref=np.array([T_ref]),
         probe_x0=npy(x0).astype(np.float32), probe_t0=npy(t0).astype(np.float32),
         probe_promptx=npy(prompt_x).astype(np.float32),
         probe_out=npy(probe).astype(np.float32),
         mel=npy(mel).astype(np.float32), mel_full=npy(mel_full).astype(np.float32))
print("codes", codes.shape, "mu", tuple(mu.shape), "T_ref", T_ref, "Ttot", Ttot)
print("probe_out", tuple(probe.shape), "mel", tuple(mel.shape))
print("saved", OUT / "s2a_gt.npz")
