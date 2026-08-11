"""9차 Phase 3-3 — OOD 확인: TokenSynth가 이펙트(임베딩 방향 차이)에 실제로 반응하는가.

★ 방향이 여기서 갈린다. e_dry_true를 넣어도 dry하게 안 들리면 8차 예측이 아무리
정확해도 소용없다(4차에서 우려했던 지점).

3소스 x 3이펙트, 동일 MIDI, 시드 고정으로 4가지를 만든다.
    (a) 원본 wet 오디오(실제 이펙트 렌더링 결과, TokenSynth 생성 아님) — 참조
    (b) 원본 dry 오디오(레벨0 렌더링 결과, TokenSynth 생성 아님) — 참조
        ★ reverb/highshelf는 레벨0이 완전한 무효과가 아니다(wet_level=0.4 고정,
          gain=-9dB) — 7~9차 전체가 일관되게 써온 "dry" 정의를 그대로 따른다.
    (c) e_wet을 그대로 주입해 생성 — β=0 기준(아무 조작 안 함)
    (d) e_dry_true(레벨0 임베딩)를 주입해 생성 — 상한(가장 좋은 경우)

(c)/(d) 생성물을 다시 CLAP(TokenSynth 공간)에 넣어 cos(e_regen, e_dry_true)와
cos(e_regen, e_wet)을 잰다. (d)가 (c)보다 dry 쪽으로 유의하게 이동하면 3-4 진행,
구분이 안 되면 TokenSynth가 조건화 신호에 반응하지 않는다는 뜻이라 멈춘다.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import librosa
import torch
import audiofile
from pedalboard import Distortion, HighShelfFilter, Pedalboard, Reverb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject import synthesize_from_embedding, set_all_seeds

from tokensynth import TokenSynth, CLAP, DACDecoder

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENSYNTH_MEDIA = REPO_ROOT / "tokensynth_paper" / "media"
MIDI = str(TOKENSYNTH_MEDIA / "input_midi.mid")
OUT_AUDIO_DIR = REPO_ROOT / "out" / "audio"
NSYNTH_AUDIO_DIR = REPO_ROOT / "nsynth-test" / "audio"

SEED = 42
SAMPLE_RATE = 48000
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
PEAK_TARGET_A = 0.7
SILENCE_PEAK_THRESHOLD = 1e-4
TS_ENCODE_SR = 16000

SOURCES = [
    ("bass_electronic_018-022-100.wav", "bass"),
    ("vocal_acoustic_000-050-025.wav", "vocal"),
    ("guitar_acoustic_010-021-127.wav", "guitar"),
]
EFFECT_NAMES = ["reverb", "distortion", "highshelf"]
LEVEL_VALUES = {"reverb": [0.0, 0.5], "distortion": [0.0, 15.0], "highshelf": [-9.0, 9.0]}  # [level0, level2]만
REVERB_WET_LEVEL, REVERB_DRY_LEVEL, HIGHSHELF_CUTOFF_HZ = 0.4, 0.6, 4000.0


def render_reverb(y, room_size):
    board = Pedalboard([Reverb(room_size=room_size, damping=0.5, wet_level=REVERB_WET_LEVEL,
                                dry_level=REVERB_DRY_LEVEL, width=1.0, freeze_mode=0.0)])
    return board(y, SAMPLE_RATE)


def render_distortion(y, drive_db):
    return Pedalboard([Distortion(drive_db=drive_db)])(y, SAMPLE_RATE)


def render_highshelf(y, gain_db):
    return Pedalboard([HighShelfFilter(cutoff_frequency_hz=HIGHSHELF_CUTOFF_HZ, gain_db=gain_db)])(y, SAMPLE_RATE)


RENDER_FN = {"reverb": render_reverb, "distortion": render_distortion, "highshelf": render_highshelf}


def load_and_preprocess_A(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) < NUM_SAMPLES:
        y = np.pad(y, (0, NUM_SAMPLES - len(y)))
    else:
        y = y[:NUM_SAMPLES]
    peak = float(np.abs(y).max())
    if peak < SILENCE_PEAK_THRESHOLD:
        return None
    return (y * (PEAK_TARGET_A / peak)).astype(np.float32)


def apply_condition_A(wet):
    peak = float(np.abs(wet).max())
    if peak > 1.0:
        wet = wet * (0.99 / peak)
    return wet.astype(np.float32)


def to_ts_space_48k(wet_48k):
    y16 = librosa.resample(wet_48k, orig_sr=SAMPLE_RATE, target_sr=TS_ENCODE_SR)
    return librosa.resample(y16, orig_sr=TS_ENCODE_SR, target_sr=SAMPLE_RATE).astype(np.float32)


def embed_ts(clap_wrapper, device, wet_48k_or_16k, src_sr):
    """TokenSynth 공간 임베딩. src_sr=48000이면 48k->16k->48k 왕복, 16000이면 16k->48k만."""
    if src_sr == 48000:
        y_for_embed = to_ts_space_48k(wet_48k_or_16k)
    elif src_sr == 16000:
        y_for_embed = librosa.resample(wet_48k_or_16k, orig_sr=16000, target_sr=48000).astype(np.float32)
    else:
        raise ValueError(src_sr)
    tensor = torch.tensor(y_for_embed.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        emb = clap_wrapper.clap.get_audio_embedding_from_data(tensor, use_tensor=True)
    return emb


def cos(a, b):
    a = a.detach().cpu().numpy().reshape(-1)
    b = b.detach().cpu().numpy().reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    parser = argparse.ArgumentParser(description="9차 Phase 3-3 — OOD 확인")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)
    device = torch.device(args.device)

    print("모델 로딩 중...")
    synth = TokenSynth.from_pretrained(aug=True, device=device)
    clap = CLAP(device=device)
    decoder = DACDecoder(device=device)

    rows = []
    for (fname, fam), effect in itertools.product(SOURCES, EFFECT_NAMES):
        tag = f"{fam}_{effect}"
        print(f"\n--- {tag} ---")
        y_dry = load_and_preprocess_A(NSYNTH_AUDIO_DIR / fname)
        assert y_dry is not None

        level0, level2 = LEVEL_VALUES[effect]
        wet_dry_raw = apply_condition_A(RENDER_FN[effect](y_dry, level0))   # (b) 원본 dry(레벨0)
        wet_wet_raw = apply_condition_A(RENDER_FN[effect](y_dry, level2))   # (a) 원본 wet(레벨2)

        # (a),(b) 저장 (16kHz로 리샘플해 나머지 결과물과 같은 포맷으로 저장)
        wav_a = librosa.resample(wet_wet_raw, orig_sr=SAMPLE_RATE, target_sr=16000)
        wav_b = librosa.resample(wet_dry_raw, orig_sr=SAMPLE_RATE, target_sr=16000)
        audiofile.write(str(OUT_AUDIO_DIR / f"phase3_3_{tag}_a_orig_wet.wav"), wav_a, 16000)
        audiofile.write(str(OUT_AUDIO_DIR / f"phase3_3_{tag}_b_orig_dry.wav"), wav_b, 16000)

        e_wet = embed_ts(clap, device, wet_wet_raw, 48000)
        e_dry_true = embed_ts(clap, device, wet_dry_raw, 48000)
        cos_wet_dry_orig = cos(e_wet, e_dry_true)

        # (c) e_wet 주입
        tok_c = synthesize_from_embedding(synth, e_wet, MIDI, seed=SEED, normalize="none", top_k=args.top_k)
        with torch.no_grad():
            audio_c = decoder.decode(tok_c)
        audio_c_np = audio_c.cpu().numpy()
        audiofile.write(str(OUT_AUDIO_DIR / f"phase3_3_{tag}_c_inject_ewet.wav"), audio_c_np, 16000)
        e_regen_c = embed_ts(clap, device, audio_c_np, 16000)

        # (d) e_dry_true 주입
        tok_d = synthesize_from_embedding(synth, e_dry_true, MIDI, seed=SEED, normalize="none", top_k=args.top_k)
        with torch.no_grad():
            audio_d = decoder.decode(tok_d)
        audio_d_np = audio_d.cpu().numpy()
        audiofile.write(str(OUT_AUDIO_DIR / f"phase3_3_{tag}_d_inject_edrytrue.wav"), audio_d_np, 16000)
        e_regen_d = embed_ts(clap, device, audio_d_np, 16000)

        row = {
            "source": fam, "effect": effect,
            "cos_origwet_origdry": cos_wet_dry_orig,
            "c_cos_regen_drytrue": cos(e_regen_c, e_dry_true), "c_cos_regen_wet": cos(e_regen_c, e_wet),
            "d_cos_regen_drytrue": cos(e_regen_d, e_dry_true), "d_cos_regen_wet": cos(e_regen_d, e_wet),
        }
        row["dry_shift_drytrue_axis"] = row["d_cos_regen_drytrue"] - row["c_cos_regen_drytrue"]
        row["dry_shift_wet_axis"] = row["c_cos_regen_wet"] - row["d_cos_regen_wet"]  # 양수면 d가 wet에서 멀어짐(=dry쪽)
        rows.append(row)
        print(f"  원본 cos(wet,dry)={cos_wet_dry_orig:.4f}  |  "
              f"(c)regen: cos_to_dry={row['c_cos_regen_drytrue']:.4f} cos_to_wet={row['c_cos_regen_wet']:.4f}  |  "
              f"(d)regen: cos_to_dry={row['d_cos_regen_drytrue']:.4f} cos_to_wet={row['d_cos_regen_wet']:.4f}  |  "
              f"dry_shift={row['dry_shift_drytrue_axis']:+.4f}")

    shifts_dry_axis = np.array([r["dry_shift_drytrue_axis"] for r in rows])
    shifts_wet_axis = np.array([r["dry_shift_wet_axis"] for r in rows])
    n_positive_dry = int((shifts_dry_axis > 0).sum())
    n_positive_wet = int((shifts_wet_axis > 0).sum())

    rng = np.random.RandomState(0)
    boot_means = [shifts_dry_axis[rng.choice(len(shifts_dry_axis), len(shifts_dry_axis), replace=True)].mean() for _ in range(2000)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    verdict = (
        "(d)가 (c)보다 dry 쪽으로 유의하게 이동 — 3-4 진행 가능"
        if ci_lo > 0 else
        "구분 안 됨(부트스트랩 CI가 0 포함) — TokenSynth가 조건화 신호에 약하게만 반응. 신중히 판단 필요"
    )

    results = {
        "meta": {"sources": SOURCES, "effects": EFFECT_NAMES, "seed": SEED, "midi": MIDI, "top_k": args.top_k},
        "depends_on_surrogate": "none",
        "rows": rows,
        "summary": {
            "mean_dry_shift": float(shifts_dry_axis.mean()), "dry_shift_ci95": [float(ci_lo), float(ci_hi)],
            "n_positive_dry_shift": n_positive_dry, "n_total": len(rows),
            "mean_wet_axis_shift": float(shifts_wet_axis.mean()), "n_positive_wet_axis_shift": n_positive_wet,
            "verdict": verdict,
        },
    }
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results_9_phase3_3.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 3-3 요약 ===")
    print(f"dry_shift(=cos_to_dry[d]-cos_to_dry[c]) 평균={shifts_dry_axis.mean():+.4f}  "
          f"부트스트랩 95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  양수인 조합={n_positive_dry}/{len(rows)}")
    print(f"wet_axis_shift(=cos_to_wet[c]-cos_to_wet[d]) 평균={shifts_wet_axis.mean():+.4f}  양수인 조합={n_positive_wet}/{len(rows)}")
    print(f"\n판정: {verdict}")
    print(f"\n저장: {results_dir / 'results_9_phase3_3.json'}")
    print(f"wav 12개(3소스x3이펙트x{{a,b,c,d}}... 정확히는 소스x이펙트당 4개 = 36개): {OUT_AUDIO_DIR}/phase3_3_*.wav")
    print("★ 여기서 멈춥니다. 반드시 청취 확인 후(파일 목록 위) 3-4 진행 여부를 결정하세요.")


if __name__ == "__main__":
    main()
