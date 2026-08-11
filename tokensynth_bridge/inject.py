"""9차 Phase 2-B — 임베딩 주입 경로.

기존 tokensynth_paper 코드는 건드리지 않는다. `TokenSynth.synthesize()`(model.py)는
이미 `clap_embedding: torch.Tensor[1,512]`를 직접 인자로 받는 공개 API라서, 몽키패치
없이 임의의 512차원 벡터를 그대로 주입할 수 있다.

세 지점(코드 근거, tokensynth_paper/src/tokensynth/):

  1) CLAP 임베딩 생성    clap.py  CLAP.encode_audio()/encode_text()
     — L2 정규화는 이 래퍼가 하는 게 아니라 laion_clap 라이브러리 내부
       (`laion_clap.clap_module.model.CLAP.get_audio_embedding`의
       `audio_embeds = F.normalize(audio_embeds, dim=-1)`)에서 일어난다. 우리
       01_embed.py의 embed_batch()도 같은 laion_clap 호출을 거치므로 정규화 지점은
       동일 — 확인 완료(둘 다 노름 1.000000).
  2) transformer 전달    model.py  TokenSynth.forward()
       `clap_proj = self.clap_projection(clap_embedding).unsqueeze(1)`
       `embedding = torch.cat((clap_proj, tok_embedding), dim=1)`
  3) projection layer     model.py  TokenSynth.__init__()
       `self.clap_projection = nn.Sequential(nn.Linear(512,1024), nn.ReLU(),
       nn.Linear(1024, hparams.embed_dim))`  — 논문 III-A의 2-layer MLP,
       512 -> 1024 -> embed_dim(=1024)과 정확히 일치.

★ Phase 2-A 실측: TokenSynth 자체 경로(clap.encode_audio, 16k->48k 업샘플·정규화
  없음·원본 길이)와 우리 파이프라인(01_embed.py, 48k 직접·피크 0.7 정규화·4.0초
  고정)으로 같은 참조 오디오를 인코딩하면 cos=0.7035(무작위 기준선 0.003±0.053 대비는
  훨씬 높지만 1.0과는 거리가 멂) — 차이가 실질적이다. 그래서 이 모듈은 8차 학습
  임베딩과 정확히 같은 전처리(우리 파이프라인)만 쓴다. TokenSynth의 clap.encode_audio는
  Phase 2-B 검증(기존 경로 재현)에서만, 참고용으로 딱 한 번 쓴다.
"""
import random
from pathlib import Path

import numpy as np
import librosa
import torch

# ---- 01_embed.py와 동일한 상수/전처리 (8차 학습 임베딩과 동일 파이프라인) ----
OUR_SAMPLE_RATE = 48000
OUR_DURATION_SEC = 4.0
OUR_NUM_SAMPLES = int(OUR_SAMPLE_RATE * OUR_DURATION_SEC)
OUR_PEAK_TARGET = 0.7
OUR_SILENCE_THRESHOLD = 1e-4


def our_load_and_preprocess(path):
    y, _ = librosa.load(path, sr=OUR_SAMPLE_RATE, mono=True)
    if len(y) < OUR_NUM_SAMPLES:
        y = np.pad(y, (0, OUR_NUM_SAMPLES - len(y)))
    else:
        y = y[:OUR_NUM_SAMPLES]
    peak = float(np.abs(y).max())
    if peak < OUR_SILENCE_THRESHOLD:
        return None
    y = (y * (OUR_PEAK_TARGET / peak)).astype(np.float32)
    return y


def extract_embedding_our_pipeline(clap_wrapper, path, device):
    """8차 학습 임베딩과 동일한 전처리로 CLAP 임베딩을 뽑는다 (TokenSynth 자체
    clap.encode_audio()는 쓰지 않음 — Phase 2-A에서 cos=0.70로 유의미하게 다름을 확인)."""
    y = our_load_and_preprocess(path)
    if y is None:
        raise ValueError(f"{path}가 무음입니다.")
    tensor = torch.tensor(y.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        emb = clap_wrapper.clap.get_audio_embedding_from_data(tensor, use_tensor=True)
    return emb  # torch.Tensor (1,512), CLAP 내부에서 이미 L2 정규화됨


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synthesize_from_embedding(synth, clap_emb, midi_fname, seed=None, normalize="none",
                               target_norm=1.0, **generation_kwargs):
    """임의의 512차원 벡터를 CLAP 임베딩 자리에 주입해 오디오 토큰을 생성한다.

    Args:
        synth: TokenSynth 인스턴스 (from_pretrained로 로드된 것)
        clap_emb: (512,) 또는 (1,512) numpy array 또는 torch.Tensor
        midi_fname: MIDI 파일 경로
        seed: 정수면 생성 전 전역 시드 고정 (자기회귀 샘플링이라 시드 없이는
              매 호출마다 다른 결과가 나온다)
        normalize: "none"(입력 그대로, 노름 민감도 실험용 기본값) |
                   "unit"(L2 정규화, 노름 1.0) | "target"(target_norm으로 스케일)
        **generation_kwargs: top_p/top_k/guidance_scale 등 synth.synthesize()에 그대로 전달

    Returns:
        torch.Tensor: 생성된 오디오 토큰 [1, seq_len, 9] (synth.synthesize()와 동일 포맷)
    """
    if seed is not None:
        set_all_seeds(seed)

    device = synth.device
    if isinstance(clap_emb, np.ndarray):
        t = torch.tensor(clap_emb, dtype=torch.float32, device=device)
    else:
        t = clap_emb.detach().to(device=device, dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)

    if normalize == "unit":
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
    elif normalize == "target":
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8) * target_norm
    elif normalize != "none":
        raise ValueError(f"알 수 없는 normalize 옵션: {normalize}")

    with torch.no_grad():
        tokens = synth.synthesize(t, midi_fname, **generation_kwargs)
    return tokens
