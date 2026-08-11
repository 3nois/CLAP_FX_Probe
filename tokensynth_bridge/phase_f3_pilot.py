"""9차 F-3 (수정: MIDI 3변형 포함) — 파일럿: MIDI 재설계가 재구성 충실도를 올리는가.

F-4(전체 재생성) 전에 8소스 x (기존MIDI 1 + 신규MIDI 3변형) x 조건(c)만 = 32생성으로
소규모 검증한다. "MIDI가 결과를 좌우하는가"를 분리하기 위해 신규 MIDI는 상행/하행/
지그재그 3개 윤곽을 쓴다(같은 4개 pitch, 순서만 다름 — midi_gen.py 참고).

cos(e_regen, e_wet)이 신규 MIDI(3변형 평균)에서 유의하게 오르면 F-4로 진행하고,
안 오르면 MIDI가 원인이 아니라는 뜻이라 여기서 멈춘다. 3변형 간 분산도 함께
보고한다 — 분산이 크면 지금까지 단일 MIDI로 잰 값들이 불안정했다는 뜻이다.

highshelf(가족 무관, F-2에서 전 패밀리 허용)로 이펙트를 통일해 MIDI 변수만 분리한다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import librosa
import torch
import audiofile
from pedalboard import HighShelfFilter, Pedalboard
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject import synthesize_from_embedding
from midi_gen import generate_midi_variants_for_source, N_VARIANTS

from tokensynth import TokenSynth, CLAP, DACDecoder

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENSYNTH_MEDIA = REPO_ROOT / "tokensynth_paper" / "media"
OLD_MIDI = str(TOKENSYNTH_MEDIA / "input_midi.mid")
OUT_AUDIO_DIR = REPO_ROOT / "out" / "audio"
NSYNTH_AUDIO_DIR = REPO_ROOT / "nsynth-test" / "audio"
NEW_MIDI_DIR = REPO_ROOT / "tokensynth_bridge" / "generated_midi"

GEN_SEED = 42
SAMPLE_RATE = 48000
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
PEAK_TARGET_A = 0.7
SILENCE_PEAK_THRESHOLD = 1e-4
HIGHSHELF_CUTOFF_HZ = 4000.0
HIGHSHELF_GAIN_LEVEL2 = 9.0

PILOT_SOURCES = [
    ("bass_synthetic_034-062-100.wav", "bass"),
    ("brass_acoustic_006-070-100.wav", "brass"),
    ("flute_acoustic_002-077-100.wav", "flute"),
    ("guitar_acoustic_010-021-127.wav", "guitar"),
    ("keyboard_acoustic_004-022-050.wav", "keyboard"),
    ("mallet_acoustic_047-065-025.wav", "mallet"),
    ("organ_electronic_001-060-100.wav", "organ"),
    ("vocal_acoustic_000-050-025.wav", "vocal"),
]


def parse_nsynth_filename(fname):
    stem = Path(fname).stem
    parts = stem.rsplit("-", 2)
    instrument, pitch_str, velocity_str = parts
    return instrument, int(pitch_str), int(velocity_str)


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


def render_highshelf(y, gain_db):
    return Pedalboard([HighShelfFilter(cutoff_frequency_hz=HIGHSHELF_CUTOFF_HZ, gain_db=gain_db)])(y, SAMPLE_RATE)


def embed_ts_48k(clap_wrapper, device, y48):
    y16 = librosa.resample(y48, orig_sr=SAMPLE_RATE, target_sr=16000)
    y48r = librosa.resample(y16, orig_sr=16000, target_sr=SAMPLE_RATE).astype(np.float32)
    tensor = torch.tensor(y48r.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        return clap_wrapper.clap.get_audio_embedding_from_data(tensor, use_tensor=True)


def embed_ts_16k(clap_wrapper, device, y16):
    y48 = librosa.resample(y16, orig_sr=16000, target_sr=SAMPLE_RATE).astype(np.float32)
    tensor = torch.tensor(y48.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        return clap_wrapper.clap.get_audio_embedding_from_data(tensor, use_tensor=True)


def cos_np(a, b):
    a = np.asarray(a).reshape(-1); b = np.asarray(b).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    parser = argparse.ArgumentParser(description="9차 F-3(수정) — MIDI 3변형 파일럿(32생성)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    t_start = time.time()
    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("모델 로딩 중...")
    synth = TokenSynth.from_pretrained(aug=True, device=device)
    clap = CLAP(device=device)
    decoder = DACDecoder(device=device)

    rows = []
    for fname, fam in PILOT_SOURCES:
        instrument, pitch, velocity = parse_nsynth_filename(fname)
        y_dry = load_and_preprocess_A(NSYNTH_AUDIO_DIR / fname)
        if y_dry is None:
            print(f"  경고: {fname} 무음, 건너뜀")
            continue
        wet_raw = apply_condition_A(render_highshelf(y_dry, HIGHSHELF_GAIN_LEVEL2))
        e_wet = embed_ts_48k(clap, device, wet_raw)

        variants = generate_midi_variants_for_source(pitch, velocity, seed=pitch, out_dir=NEW_MIDI_DIR, tag=f"{fam}_{pitch}")

        row = {"family": fam, "filename": fname, "pitch": pitch, "velocity": velocity, "variants_info": variants}

        tok = synthesize_from_embedding(synth, e_wet, OLD_MIDI, seed=GEN_SEED, normalize="none", top_k=args.top_k)
        with torch.no_grad():
            audio_old = decoder.decode(tok).cpu().numpy()
        wav_old = OUT_AUDIO_DIR / f"phase_f3_{fam}_midi_old.wav"
        audiofile.write(str(wav_old), audio_old, 16000)
        row["midi_old_cos_to_wet"] = cos_np(embed_ts_16k(clap, device, audio_old), e_wet)
        row["midi_old_wav"] = str(wav_old)

        new_cos_vals = []
        for variant in range(N_VARIANTS):
            midi_path = variants[variant]["path"]
            tok = synthesize_from_embedding(synth, e_wet, midi_path, seed=GEN_SEED, normalize="none", top_k=args.top_k)
            with torch.no_grad():
                audio_new = decoder.decode(tok).cpu().numpy()
            wav_new = OUT_AUDIO_DIR / f"phase_f3_{fam}_midi_new_v{variant}.wav"
            audiofile.write(str(wav_new), audio_new, 16000)
            c = cos_np(embed_ts_16k(clap, device, audio_new), e_wet)
            row[f"midi_new_v{variant}_cos_to_wet"] = c
            row[f"midi_new_v{variant}_wav"] = str(wav_new)
            new_cos_vals.append(c)
        row["midi_new_mean_cos_to_wet"] = float(np.mean(new_cos_vals))
        row["midi_new_variant_std"] = float(np.std(new_cos_vals))

        rows.append(row)
        print(f"  {fam:<10} pitch={pitch} 기존={row['midi_old_cos_to_wet']:.4f}  "
              f"신규(v0/v1/v2)={new_cos_vals[0]:.4f}/{new_cos_vals[1]:.4f}/{new_cos_vals[2]:.4f}  "
              f"신규평균={row['midi_new_mean_cos_to_wet']:.4f}(std={row['midi_new_variant_std']:.4f})  "
              f"Δ={row['midi_new_mean_cos_to_wet']-row['midi_old_cos_to_wet']:+.4f}")

    old_vals = np.array([r["midi_old_cos_to_wet"] for r in rows])
    new_mean_vals = np.array([r["midi_new_mean_cos_to_wet"] for r in rows])
    diff = new_mean_vals - old_vals
    n_improved = int((diff > 0).sum())

    all_variant_stds = np.array([r["midi_new_variant_std"] for r in rows])
    between_source_std = float(new_mean_vals.std())

    try:
        stat, pval = wilcoxon(new_mean_vals, old_vals, alternative="greater")
    except ValueError:
        stat, pval = None, None

    variance_flag = "MIDI 변형 간 분산이 소스 간 분산 대비 크다 — MIDI 설계가 불안정" \
        if all_variant_stds.mean() > 0.5 * between_source_std else \
        "MIDI 변형 간 분산이 소스 간 분산보다 작다 — 변형 선택이 결과를 크게 흔들지 않음"

    if pval is not None and pval < 0.05:
        verdict = "신규 MIDI(3변형 평균)에서 유의하게 상승 — F-4 전체 재생성으로 진행"
    else:
        verdict = "유의한 차이 없음 — MIDI가 주원인이 아닐 가능성. 3번(학습분포 밖) 문제일 수 있음. 신중 검토 필요"

    results = {
        "meta": {"n_pilot_sources": len(rows), "effect": "highshelf", "gain_db": HIGHSHELF_GAIN_LEVEL2,
                  "seed": GEN_SEED, "old_midi": OLD_MIDI, "n_variants": N_VARIANTS, "elapsed_sec": time.time() - t_start},
        "depends_on_surrogate": "none",
        "rows": rows,
        "summary": {
            "mean_old": float(old_vals.mean()), "mean_new_avg_of_variants": float(new_mean_vals.mean()),
            "mean_diff": float(diff.mean()), "n_improved": n_improved, "n_total": len(rows),
            "wilcoxon_stat": stat, "wilcoxon_p_greater": pval, "verdict": verdict,
            "variant_std_within_source_mean": float(all_variant_stds.mean()),
            "between_source_std_of_new_mean": between_source_std,
            "variance_flag": variance_flag,
        },
    }
    results_dir = Path(args.out) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results_9_phase_f3_pilot.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== F-3(수정) 파일럿 결과 ===")
    print(f"기존 MIDI 평균 cos(e_regen,e_wet) = {old_vals.mean():.4f}")
    print(f"신규 MIDI(3변형 평균) 평균 = {new_mean_vals.mean():.4f}")
    print(f"평균 개선 = {diff.mean():+.4f}  (상승한 소스 {n_improved}/{len(rows)})")
    print(f"Wilcoxon(신규>기존) p = {pval}")
    print(f"\nMIDI 변형 간 분산(소스 내, 평균 std) = {all_variant_stds.mean():.4f}")
    print(f"소스 간 분산(신규평균 기준 std) = {between_source_std:.4f}")
    print(f"분산 판정: {variance_flag}")
    print(f"\n판정: {verdict}")
    print(f"\n저장: {results_dir/'results_9_phase_f3_pilot.json'}")
    print(f"wav {len(rows)*4}개: {OUT_AUDIO_DIR}/phase_f3_*_midi_{{old,new_v0,new_v1,new_v2}}.wav")
    print(f"생성 MIDI: {NEW_MIDI_DIR}/*_v{{0,1,2}}.mid")
    print("★ 여기서 멈춥니다. F-4 전체 재생성 진행 여부를 결정하세요.")


if __name__ == "__main__":
    main()
