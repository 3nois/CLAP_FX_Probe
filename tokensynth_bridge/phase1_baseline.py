"""9차 Phase 1 — TokenSynth 기본 추론 재현 (환경 검증).

tokensynth_paper/quickstart.py의 최소 경로(참조 오디오 조건, 텍스트/가이던스 없음)만
재현한다. 별도 디렉터리(tokensynth_bridge/)에서 작업하며 tokensynth_paper/의 파일은
건드리지 않는다 — 입력은 media/에서 읽기만 하고, 출력은 out/audio/에 새로 쓴다.

CPU를 기본으로 한다(M5에서 MPS가 자기회귀 생성 경로를 막을 가능성이 높다고 보고됨).
생성 시간을 측정해 이후 청취 검증 세트(Phase 4) 규모를 정하는 근거로 삼는다.
"""
import argparse
import time
from pathlib import Path

import torch
from tokensynth import TokenSynth, CLAP, DACDecoder
import audiofile

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENSYNTH_MEDIA = REPO_ROOT / "tokensynth_paper" / "media"
OUT_AUDIO = REPO_ROOT / "out" / "audio"


def main():
    parser = argparse.ArgumentParser(description="9차 Phase 1 — TokenSynth 기본 추론 재현")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--ref-audio", type=str, default=str(TOKENSYNTH_MEDIA / "reference_audio.wav"))
    parser.add_argument("--midi", type=str, default=str(TOKENSYNTH_MEDIA / "input_midi.mid"))
    parser.add_argument("--out", type=str, default=str(OUT_AUDIO / "phase1_baseline.wav"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.device == "mps":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = torch.device(args.device)
    print(f"device = {device}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("TokenSynth(aug) 로딩 중...")
    synth = TokenSynth.from_pretrained(aug=True, device=device)
    print(f"  {time.time()-t0:.1f}초")

    t0 = time.time()
    print("CLAP 로딩 중...")
    clap = CLAP(device=device)
    print(f"  {time.time()-t0:.1f}초")

    t0 = time.time()
    print("DACDecoder 로딩 중...")
    decoder = DACDecoder(device=device)
    print(f"  {time.time()-t0:.1f}초")

    with torch.no_grad():
        t0 = time.time()
        timbre_audio = clap.encode_audio(args.ref_audio)
        t_embed = time.time() - t0
        print(f"CLAP 임베딩 추출: {t_embed:.2f}초, shape={tuple(timbre_audio.shape)}, "
              f"norm={timbre_audio.norm(dim=-1).item():.4f}")

        t0 = time.time()
        tokens_audio = synth.synthesize(timbre_audio, args.midi, top_k=args.top_k)
        t_synth = time.time() - t0
        print(f"토큰 생성: {t_synth:.1f}초, tokens shape={tuple(tokens_audio.shape)}")

        t0 = time.time()
        audio = decoder.decode(tokens_audio)
        t_decode = time.time() - t0
        print(f"DAC 디코딩: {t_decode:.2f}초")

    audiofile.write(args.out, audio.cpu().numpy(), 16000)
    total = t_embed + t_synth + t_decode
    print(f"\n저장: {args.out}")
    print(f"총 소요(임베딩+생성+디코딩): {total:.1f}초 (임베딩 {t_embed:.2f}s / 생성 {t_synth:.1f}s / 디코딩 {t_decode:.2f}s)")
    print(f"오디오 길이: {audio.shape[-1]/16000:.2f}초, device={device}")


if __name__ == "__main__":
    main()
