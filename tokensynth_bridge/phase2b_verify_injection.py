"""9차 Phase 2-B 검증 — 주입 경로가 기존 경로와 같은 결과를 내는가.

참조 오디오에서 뽑은 임베딩(TokenSynth 자체 clap.encode_audio() 경로 — Phase 1과 정확히
동일)을 그대로 synthesize_from_embedding()에 주입했을 때, synth.synthesize()를 직접
부르는 것과 같은 결과가 나오는지 확인한다. 자기회귀 샘플링이라 시드를 고정해야 비교가
성립한다.

  (0) 시드 고정 자체가 재현성을 주는지 sanity check (직접 경로를 같은 시드로 2번)
  (1) 직접 경로 vs 주입 경로 — 같은 임베딩·같은 시드
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject import synthesize_from_embedding, set_all_seeds

from tokensynth import TokenSynth, CLAP

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENSYNTH_MEDIA = REPO_ROOT / "tokensynth_paper" / "media"
REF_AUDIO = str(TOKENSYNTH_MEDIA / "reference_audio.wav")
MIDI = str(TOKENSYNTH_MEDIA / "input_midi.mid")
SEED = 42


def tokens_equal(a, b):
    if a.shape != b.shape:
        return False, f"shape 다름 {a.shape} vs {b.shape}"
    diff = (a != b).sum().item()
    return diff == 0, f"불일치 토큰 수 = {diff}/{a.numel()}"


def main():
    device = torch.device("cpu")
    print("모델 로딩 중...")
    synth = TokenSynth.from_pretrained(aug=True, device=device)
    clap = CLAP(device=device)

    print("\n참조 오디오 임베딩 추출 (TokenSynth 자체 경로, Phase 1과 동일)...")
    with torch.no_grad():
        emb = clap.encode_audio(REF_AUDIO)
    print(f"  norm={emb.norm().item():.6f}")

    # ---- (0) 시드 고정 자체의 재현성 sanity check ----
    print("\n(0) 시드 고정 재현성 sanity check (직접 경로 2회, 같은 시드)...")
    set_all_seeds(SEED)
    with torch.no_grad():
        tok_a = synth.synthesize(emb, MIDI, top_k=10)
    set_all_seeds(SEED)
    with torch.no_grad():
        tok_b = synth.synthesize(emb, MIDI, top_k=10)
    ok0, msg0 = tokens_equal(tok_a, tok_b)
    print(f"  직접경로(1회) vs 직접경로(2회, 같은 시드): 동일={ok0}  {msg0}")

    # ---- (1) 직접 경로 vs 주입 경로 ----
    print("\n(1) 직접 경로 vs 주입 경로 (같은 임베딩·같은 시드)...")
    set_all_seeds(SEED)
    with torch.no_grad():
        tok_direct = synth.synthesize(emb, MIDI, top_k=10)

    tok_injected = synthesize_from_embedding(synth, emb, MIDI, seed=SEED, normalize="none", top_k=10)

    ok1, msg1 = tokens_equal(tok_direct, tok_injected)
    print(f"  직접경로 vs 주입경로: 동일={ok1}  {msg1}")

    # ---- 다른 시드면 달라지는지도 확인 (시드가 실제로 작동하는지) ----
    tok_diff_seed = synthesize_from_embedding(synth, emb, MIDI, seed=SEED + 1, normalize="none", top_k=10)
    ok2, msg2 = tokens_equal(tok_direct, tok_diff_seed)
    print(f"\n(참고) 직접경로 vs 주입경로(다른 시드): 동일={ok2} (다르면 정상) {msg2}")

    print("\n=== 결론 ===")
    if ok0 and ok1:
        print("★ 주입 경로가 기존 경로와 완전히 동일한 결과를 낸다 — 주입 지점 정상.")
    elif ok0 and not ok1:
        print("★ 시드 고정 자체는 재현되는데 주입 경로만 다르다 — 주입 지점에 문제 있음. 원인 조사 필요.")
    else:
        print("★ 시드 고정만으로는 재현이 안 된다(비결정적 연산 존재 가능) — 비교 방법 자체를 재검토해야 함.")


if __name__ == "__main__":
    main()
