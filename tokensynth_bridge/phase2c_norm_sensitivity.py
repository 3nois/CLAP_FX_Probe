"""9차 Phase 2-C — 노름 민감도 실험.

CLAP 임베딩은 항상 노름 1.0(L2 정규화, laion_clap 내부에서 강제됨 — Phase 2-A에서
확인). TokenSynth는 이런 단위노름 임베딩만 보고 학습됐을 것이므로, e_dry_hat =
e_wet + β·v_to_dry처럼 노름이 1.0을 벗어나는 입력을 넣었을 때 모델이 어떻게
반응하는지는 코드로 알 수 없고 실측해야 한다.

방향은 참조 오디오 임베딩 방향으로 고정하고 노름만 바꿔가며 생성한다. 같은 시드·같은
MIDI로 노름 외 변동을 없앤다.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import audiofile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject import synthesize_from_embedding, set_all_seeds, extract_embedding_our_pipeline, our_load_and_preprocess

from tokensynth import TokenSynth, CLAP, DACDecoder

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENSYNTH_MEDIA = REPO_ROOT / "tokensynth_paper" / "media"
REF_AUDIO = str(TOKENSYNTH_MEDIA / "reference_audio.wav")
MIDI = str(TOKENSYNTH_MEDIA / "input_midi.mid")
OUT_AUDIO_DIR = REPO_ROOT / "out" / "audio"
SEED = 42
NORMS = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]


def main():
    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print("모델 로딩 중...")
    synth = TokenSynth.from_pretrained(aug=True, device=device)
    clap = CLAP(device=device)
    decoder = DACDecoder(device=device)

    print("참조 임베딩 추출 (우리 파이프라인, 8차 학습과 동일 전처리)...")
    e_ref = extract_embedding_our_pipeline(clap, REF_AUDIO, device)  # (1,512), norm=1.0
    direction = e_ref / e_ref.norm(dim=-1, keepdim=True)
    print(f"  방향 노름(확인용)={direction.norm().item():.6f}")

    results = {}
    ref_reembed = None  # norm=1.0 결과의 재임베딩 (기준)

    for norm in NORMS:
        emb_scaled = direction * norm
        t0 = time.time()
        tokens = synthesize_from_embedding(synth, emb_scaled, MIDI, seed=SEED, normalize="none", top_k=10)
        t_gen = time.time() - t0

        with torch.no_grad():
            audio = decoder.decode(tokens)
        audio_np = audio.cpu().numpy()

        fname = OUT_AUDIO_DIR / f"phase2_norm_{norm:.1f}.wav"
        audiofile.write(str(fname), audio_np, 16000)

        n_tokens = tokens.shape[1]
        duration_sec = audio_np.shape[-1] / 16000
        rms = float(np.sqrt(np.mean(audio_np.astype(np.float64) ** 2)))
        peak = float(np.abs(audio_np).max())

        # 재임베딩(우리 파이프라인, 16kHz 생성물을 파일로 저장했으니 그 파일을 다시 로드)
        with torch.no_grad():
            reembed = extract_embedding_our_pipeline(clap, str(fname), device)
        if norm == 1.0:
            ref_reembed = reembed
        results[norm] = {
            "n_tokens": int(n_tokens), "duration_sec": duration_sec, "rms": rms, "peak": peak,
            "gen_time_sec": t_gen, "reembed": reembed, "wav_path": str(fname),
        }
        print(f"  norm={norm:.1f}: tokens={n_tokens} dur={duration_sec:.2f}s rms={rms:.4f} peak={peak:.4f} "
              f"gen_time={t_gen:.1f}s -> {fname.name}")

    print("\n=== norm=1.0 기준 재임베딩 코사인 (정량화) ===")
    for norm in NORMS:
        r = results[norm]
        cos = float(torch.nn.functional.cosine_similarity(r["reembed"], ref_reembed).item())
        r["cos_to_norm1"] = cos
        flag = ""
        if r["duration_sec"] < 1.0:
            flag += " ⚠매우 짧음(붕괴 의심)"
        if r["rms"] < 1e-3:
            flag += " ⚠거의 무음"
        print(f"  norm={norm:.1f}: cos(재임베딩, norm1.0 재임베딩)={cos:.4f}  tokens={r['n_tokens']}  dur={r['duration_sec']:.2f}s{flag}")

    print("\n저장된 wav:")
    for norm in NORMS:
        print(f"  {results[norm]['wav_path']}")

    # 저장용으로 텐서 제거
    import json
    out = {
        str(norm): {k: v for k, v in r.items() if k != "reembed"}
        for norm, r in results.items()
    }
    results_dir = REPO_ROOT / "out" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results_9_phase2c_norm.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {results_dir / 'results_9_phase2c_norm.json'}")


if __name__ == "__main__":
    main()
