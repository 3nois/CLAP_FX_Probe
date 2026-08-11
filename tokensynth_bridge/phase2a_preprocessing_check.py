"""9차 Phase 2-A — 전처리 대조: 우리 파이프라인 vs TokenSynth의 clap.encode_audio().

코드 대조(README 표)로 이미 몇 개 항목이 다름을 확인했다 — 이 스크립트는 그 차이가
임베딩에 실질적 영향을 주는지 실측한다. 같은 오디오 파일을 두 경로로 인코딩해
코사인 유사도를 비교한다. 새 렌더링 없음, 기존 참조 오디오만 사용.
"""
import numpy as np
import librosa
import torch

from tokensynth import CLAP

REF_AUDIO = "tokensynth_paper/media/reference_audio.wav"

# ---- 우리 경로 (01_embed.py와 동일 상수/로직) ----
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
    y = y * (OUR_PEAK_TARGET / peak)
    return y.astype(np.float32)


def main():
    device = torch.device("cpu")
    print("CLAP 로딩 중 (TokenSynth 래퍼, 체크포인트는 우리 것과 SHA256 동일 확인됨)...")
    clap = CLAP(device=device)

    # ---- 경로 1: TokenSynth 자체 경로 (16k 로드 -> 48k 업샘플, 정규화 없음) ----
    emb_tokensynth = clap.encode_audio(REF_AUDIO)  # torch.Tensor (1,512)

    # ---- 경로 2: 우리 파이프라인 경로 (48k 직접 로드, 4.0초 고정, 피크 0.7 정규화) ----
    y_ours = our_load_and_preprocess(REF_AUDIO)
    assert y_ours is not None, "무음으로 판정됨 — 예상 밖"
    tensor_ours = torch.tensor(y_ours.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        emb_ours = clap.clap.get_audio_embedding_from_data(tensor_ours, use_tensor=True)

    e1 = emb_tokensynth.detach().cpu().numpy().reshape(-1)
    e2 = emb_ours.detach().cpu().numpy().reshape(-1)
    cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12))

    print(f"\nTokenSynth 경로 임베딩 노름: {np.linalg.norm(e1):.6f}")
    print(f"우리 경로 임베딩 노름:      {np.linalg.norm(e2):.6f}")
    print(f"cos(TokenSynth 경로, 우리 경로) = {cos:.6f}")
    print(f"L2 거리 = {np.linalg.norm(e1 - e2):.6f}")

    # 참고용 무작위 기준선(같은 512차원 구 위 두 무작위 단위벡터의 기대 코사인 ~0, std~1/sqrt(512))
    rng = np.random.RandomState(0)
    rv = rng.normal(size=(1000, 512))
    rv /= np.linalg.norm(rv, axis=1, keepdims=True)
    random_cos = rv @ rv[0]
    print(f"\n(참고) 무작위 두 단위벡터 코사인 분포: mean={random_cos.mean():.4f} std={random_cos.std():.4f}")


if __name__ == "__main__":
    main()
