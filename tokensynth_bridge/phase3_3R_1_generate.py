"""9차 Phase 3-3R-1 — 확장 검증: 변위 방향 일치도.

Phase 3-3(9쌍)에서 정량 지표가 "구분 안 됨"으로 나왔으나 사용자 청취로는 (c)/(d)가
명확히 구분됐다 — 지표가 잘못된 것을 재고 있었다. 재생성물이 원본 wet/dry 양쪽에서
멀다(cos 0.25~0.65)는 사실 자체가 재구성 충실도 문제이지 wet/dry 판별 문제가
아니다. **절대 위치가 아니라 변위 방향**을 봐야 한다:

    v_generated = e_regen(d) - e_regen(c)      생성물 사이 변위
    v_original  = e_dry_true - e_wet            원본 사이 변위
    directional_agreement = cos(v_generated, v_original)

50소스(10패밀리×5) × 3이펙트 × 2조건(c,d) = 300생성으로 확대해 검정력을 키운다.
소스는 Phase 3-1(out/caches/oat_emb_ts.npz)에서 뽑고, e_wet/e_dry_true는 이미
캐시된 TokenSynth 공간 임베딩을 그대로 쓴다(재추출 안 함 — 3-1과 동일 렌더링·
동일 시드로 만들어졌으므로 정확히 같은 값). (a)/(b) 원본 wet/dry는 오디오
파형이 필요하므로 pedalboard로 다시 렌더링해 파일로만 저장한다(임베딩은 재추출
안 함).

기존 out/results/results_9_phase3_3.json은 건드리지 않는다 — 새 파일.
"""
import argparse
import collections
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
import audiofile
from pedalboard import Distortion, HighShelfFilter, Pedalboard, Reverb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject import synthesize_from_embedding

from tokensynth import TokenSynth, CLAP, DACDecoder

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
COLORS = {"reverb": "#2a78d6", "distortion": "#eb6834", "highshelf": "#1baf7a", "null": "#e34948", "baseline": "#898781"}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENSYNTH_MEDIA = REPO_ROOT / "tokensynth_paper" / "media"
MIDI = str(TOKENSYNTH_MEDIA / "input_midi.mid")
OUT_AUDIO_DIR = REPO_ROOT / "out" / "audio"
NSYNTH_AUDIO_DIR = REPO_ROOT / "nsynth-test" / "audio"

GEN_SEED = 42
N_PER_FAMILY = 5
SELECT_SEED = 1  # 3-1의 소스풀(seed=0)에서 50개를 다시 뽑는 시드 — 전체 선정 시드(0)와 구분

SAMPLE_RATE = 48000
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
PEAK_TARGET_A = 0.7
SILENCE_PEAK_THRESHOLD = 1e-4
TS_ENCODE_SR = 16000

NSYNTH_SOURCE_TYPES = {"acoustic", "electronic", "synthetic"}
EFFECT_NAMES = ["reverb", "distortion", "highshelf"]
LEVEL_RAW = {"reverb": [0.0, 0.5], "distortion": [0.0, 15.0], "highshelf": [-9.0, 9.0]}  # [level0, level2] 원시값
REVERB_WET_LEVEL, REVERB_DRY_LEVEL, HIGHSHELF_CUTOFF_HZ = 0.4, 0.6, 4000.0

# 3-1(phase3_1_reextract.py)과 완전히 동일 — 같은 seed=0으로 같은 1,200소스 목록을 재현
N_SOURCES_PER_FAMILY_TARGET = 120
N_SOURCES_PER_FAMILY_MIN = 60
RESELECT_SEED = 0


def parse_instrument_family(instrument):
    tokens = instrument.split("_")
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] in NSYNTH_SOURCE_TYPES:
            family = "_".join(tokens[:i])
            return family if family else instrument
    return instrument


def parse_nsynth_filename(path):
    parts = path.stem.rsplit("-", 2)
    if len(parts) != 3:
        return path.stem, None, None
    instrument, pitch_str, velocity_str = parts
    try:
        return instrument, int(pitch_str), int(velocity_str)
    except ValueError:
        return instrument, None, None


def select_sources(audio_dir, n_target, n_min, seed):
    files = sorted(audio_dir.glob("*.wav"))
    by_family = collections.defaultdict(list)
    for f in files:
        instrument, pitch, velocity = parse_nsynth_filename(f)
        if pitch is None:
            continue
        fam = parse_instrument_family(instrument)
        by_family[fam].append((f.name, instrument))
    rng = np.random.RandomState(seed)
    selected = []
    for fam in sorted(by_family.keys()):
        pool = by_family[fam]
        if len(pool) < n_min:
            continue
        n_take = min(n_target, len(pool))
        idx = rng.choice(len(pool), size=n_take, replace=False)
        for i in sorted(idx.tolist()):
            fname, instrument = pool[i]
            selected.append((fname, instrument, fam))
    return selected


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


def embed_ts_from_16k(clap_wrapper, device, y16k):
    y48 = librosa.resample(y16k, orig_sr=16000, target_sr=48000).astype(np.float32)
    tensor = torch.tensor(y48.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        return clap_wrapper.clap.get_audio_embedding_from_data(tensor, use_tensor=True)


def cos_np(a, b):
    a = np.asarray(a).reshape(-1); b = np.asarray(b).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def bootstrap_ci_by_source(values, src_ids, seed, n_boot=2000):
    values = np.asarray(values)
    sources = np.unique(src_ids)
    rng = np.random.RandomState(seed)
    src_to_rows = {s: np.where(src_ids == s)[0] for s in sources}
    means = []
    for _ in range(n_boot):
        boot = rng.choice(sources, size=len(sources), replace=True)
        rows = np.concatenate([src_to_rows[s] for s in boot])
        means.append(values[rows].mean())
    means = np.array(means)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser(description="9차 Phase 3-3R-1 — 확장 검증(300생성)")
    parser.add_argument("--oat-emb-ts", type=str, default="out/caches/oat_emb_ts.npz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.out)
    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("oat_emb_ts.npz 로딩 및 3-1 소스 목록 재현 중...")
    d = np.load(args.oat_emb_ts, allow_pickle=False)
    emb = d["emb"]  # (1200,3,3,512), TokenSynth 공간
    src_id_arr = d["src_id"]
    family_arr = d["instrument_family"]

    full_selected = select_sources(NSYNTH_AUDIO_DIR, N_SOURCES_PER_FAMILY_TARGET, N_SOURCES_PER_FAMILY_MIN, RESELECT_SEED)
    assert len(full_selected) == emb.shape[0], f"소스 재현 불일치: {len(full_selected)} vs {emb.shape[0]}"

    # 50소스(10패밀리x5) 선정 — src_id_arr의 위치가 곧 full_selected의 인덱스
    rng = np.random.RandomState(SELECT_SEED)
    families_sorted = sorted(set(family_arr.tolist()))
    chosen_positions = []  # emb/full_selected 상의 위치(=src_id 값)
    for fam in families_sorted:
        fam_positions = np.where(family_arr == fam)[0]
        chosen = rng.choice(fam_positions, size=min(N_PER_FAMILY, len(fam_positions)), replace=False)
        chosen_positions.extend(sorted(chosen.tolist()))
    print(f"선정된 소스 {len(chosen_positions)}개 (패밀리별 {N_PER_FAMILY}개)")

    print("CLAP/TokenSynth/DAC 로딩 중...")
    synth = TokenSynth.from_pretrained(aug=True, device=device)
    clap = CLAP(device=device)
    decoder = DACDecoder(device=device)

    rows = []
    n_total = len(chosen_positions) * len(EFFECT_NAMES)
    done = 0
    for pos in chosen_positions:
        fname, instrument, fam = full_selected[pos]
        y_dry = load_and_preprocess_A(NSYNTH_AUDIO_DIR / fname)
        if y_dry is None:
            print(f"  경고: {fname} 무음, 건너뜀")
            continue
        for effect in EFFECT_NAMES:
            done += 1
            tag = f"{pos}_{fam}_{effect}"
            level0_raw, level2_raw = LEVEL_RAW[effect]

            # (a)/(b) 원본 wet/dry — 파형 렌더링 후 파일로만 저장 (임베딩은 캐시 재사용)
            wet_dry_raw = apply_condition_A(RENDER_FN[effect](y_dry, level0_raw))
            wet_wet_raw = apply_condition_A(RENDER_FN[effect](y_dry, level2_raw))
            wav_a = librosa.resample(wet_wet_raw, orig_sr=SAMPLE_RATE, target_sr=16000)
            wav_b = librosa.resample(wet_dry_raw, orig_sr=SAMPLE_RATE, target_sr=16000)
            audiofile.write(str(OUT_AUDIO_DIR / f"phase3r_{tag}_a_orig_wet.wav"), wav_a, 16000)
            audiofile.write(str(OUT_AUDIO_DIR / f"phase3r_{tag}_b_orig_dry.wav"), wav_b, 16000)

            ei = EFFECT_NAMES.index(effect)
            e_wet = emb[pos, ei, 2, :]         # 캐시 재사용 (3-1과 동일 렌더링)
            e_dry_true = emb[pos, ei, 0, :]

            # (c) e_wet 주입
            tok_c = synthesize_from_embedding(synth, e_wet, MIDI, seed=GEN_SEED, normalize="none", top_k=args.top_k)
            with torch.no_grad():
                audio_c = decoder.decode(tok_c).cpu().numpy()
            audiofile.write(str(OUT_AUDIO_DIR / f"phase3r_{tag}_c.wav"), audio_c, 16000)
            e_regen_c = embed_ts_from_16k(clap, device, audio_c).cpu().numpy().reshape(-1)

            # (d) e_dry_true 주입
            tok_d = synthesize_from_embedding(synth, e_dry_true, MIDI, seed=GEN_SEED, normalize="none", top_k=args.top_k)
            with torch.no_grad():
                audio_d = decoder.decode(tok_d).cpu().numpy()
            audiofile.write(str(OUT_AUDIO_DIR / f"phase3r_{tag}_d.wav"), audio_d, 16000)
            e_regen_d = embed_ts_from_16k(clap, device, audio_d).cpu().numpy().reshape(-1)

            v_generated = e_regen_d - e_regen_c
            v_original = e_dry_true - e_wet
            directional_agreement = cos_np(v_generated, v_original)
            magnitude_ratio = float(np.linalg.norm(v_generated) / (np.linalg.norm(v_original) + 1e-12))
            separation = cos_np(e_regen_c, e_regen_d)
            recon_c = cos_np(e_regen_c, e_wet)
            recon_d = cos_np(e_regen_d, e_dry_true)

            rows.append({
                "src_pos": int(pos), "family": fam, "effect": effect, "instrument": instrument, "filename": fname,
                "directional_agreement": directional_agreement, "magnitude_ratio": magnitude_ratio,
                "separation": separation, "reconstruction_c": recon_c, "reconstruction_d": recon_d,
                "wav_c": f"phase3r_{tag}_c.wav", "wav_d": f"phase3r_{tag}_d.wav",
                "wav_a": f"phase3r_{tag}_a_orig_wet.wav", "wav_b": f"phase3r_{tag}_b_orig_dry.wav",
            })
            if done % 10 == 0 or done == n_total:
                elapsed = time.time() - t_start
                eta = elapsed / done * (n_total - done)
                print(f"  [{done}/{n_total}] {tag}: dir_agree={directional_agreement:+.4f}  "
                      f"(경과 {elapsed/60:.1f}분, 예상 잔여 {eta/60:.1f}분)")

    # ---- 집계 ----
    da = np.array([r["directional_agreement"] for r in rows])
    src_ids_for_boot = np.array([r["src_pos"] for r in rows])
    overall_mean, overall_lo, overall_hi = bootstrap_ci_by_source(da, src_ids_for_boot, seed=0)

    by_effect = {}
    for effect in EFFECT_NAMES:
        sub = [r for r in rows if r["effect"] == effect]
        vals = np.array([r["directional_agreement"] for r in sub])
        srcs = np.array([r["src_pos"] for r in sub])
        m, lo, hi = bootstrap_ci_by_source(vals, srcs, seed=0)
        by_effect[effect] = {"mean": m, "ci95": [lo, hi], "n": len(sub)}

    by_family = {}
    for fam in families_sorted:
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        vals = np.array([r["directional_agreement"] for r in sub])
        srcs = np.array([r["src_pos"] for r in sub])
        m, lo, hi = bootstrap_ci_by_source(vals, srcs, seed=0)
        recon_c_mean = float(np.mean([r["reconstruction_c"] for r in sub]))
        recon_d_mean = float(np.mean([r["reconstruction_d"] for r in sub]))
        by_family[fam] = {"directional_agreement_mean": m, "ci95": [lo, hi], "n": len(sub),
                           "reconstruction_c_mean": recon_c_mean, "reconstruction_d_mean": recon_d_mean}

    verdict = (
        "CI가 0을 넘고 양수 — 조건화가 의도한 방향으로 작동"
        if overall_lo > 0 else
        ("유의하게 음수 — 반대로 움직임" if overall_hi < 0 else "0 근처 — 방향이 무작위(CI가 0 포함)")
    )

    results = {
        "meta": {
            "n_sources": len(chosen_positions), "n_per_family": N_PER_FAMILY, "n_pairs": len(rows),
            "midi": MIDI, "gen_seed": GEN_SEED, "select_seed": SELECT_SEED, "top_k": args.top_k,
            "elapsed_sec": time.time() - t_start,
        },
        "depends_on_surrogate": "none",
        "rows": rows,
        "directional_agreement": {"overall": {"mean": overall_mean, "ci95": [overall_lo, overall_hi], "n": len(rows)},
                                   "by_effect": by_effect, "by_family": by_family, "verdict": verdict},
        "magnitude_ratio_mean": float(np.mean([r["magnitude_ratio"] for r in rows])),
        "separation_mean": float(np.mean([r["separation"] for r in rows])),
        "reconstruction_note": "판정에 쓰지 않음 — 재구성 충실도는 wet/dry와 무관한 별개 문제(NSynth 단일음이 TokenSynth 학습분포 밖일 가능성). by_family에 참고용으로만 기록.",
    }
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results_9_phase3_3R.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- 그림 ----
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.hist(da, bins=30, color=COLORS["baseline"], zorder=3)
    ax.axvline(0, color=COLORS["null"], linestyle="--", linewidth=1, label="0 (무작위)")
    ax.axvline(overall_mean, color=COLORS["reverb"], linestyle="-", linewidth=1.5, label=f"평균={overall_mean:.3f}")
    ax.axvspan(overall_lo, overall_hi, color=COLORS["reverb"], alpha=0.15, label=f"95% CI [{overall_lo:.3f},{overall_hi:.3f}]")
    ax.set_xlabel("directional_agreement = cos(v_generated, v_original)")
    ax.set_ylabel("빈도 (150쌍)")
    ax.set_title("Phase 3-3R — 변위 방향 일치도")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_directional.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fams = list(by_family.keys())
    means = [by_family[f]["directional_agreement_mean"] for f in fams]
    los = [by_family[f]["ci95"][0] for f in fams]
    his = [by_family[f]["ci95"][1] for f in fams]
    x = np.arange(len(fams))
    yerr = np.array([np.array(means) - np.array(los), np.array(his) - np.array(means)])
    ax.bar(x, means, yerr=np.clip(yerr, 0, None), capsize=3, color=COLORS["baseline"], zorder=3)
    ax.axhline(0, color=COLORS["null"], linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.set_ylabel("directional_agreement (패밀리별)")
    ax.set_title("Phase 3-3R — 패밀리별 방향 일치도")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_by_family.png")
    plt.close(fig)

    print("\n=== Phase 3-3R-2 요약 (★ 핵심) ===")
    print(f"directional_agreement 전체: 평균={overall_mean:+.4f}  95% CI=[{overall_lo:+.4f},{overall_hi:+.4f}]  n={len(rows)}")
    print(f"판정: {verdict}")
    print("\n이펙트별:")
    for e, v in by_effect.items():
        print(f"  {e:<12} 평균={v['mean']:+.4f}  CI={v['ci95']}  n={v['n']}")
    print("\n패밀리별 (재구성 충실도 참고):")
    for f, v in by_family.items():
        print(f"  {f:<12} dir_agree={v['directional_agreement_mean']:+.4f} CI={v['ci95']}  "
              f"recon_c={v['reconstruction_c_mean']:.3f} recon_d={v['reconstruction_d_mean']:.3f}")
    print(f"\nmagnitude_ratio 평균={results['magnitude_ratio_mean']:.4f}  separation 평균={results['separation_mean']:.4f}")
    print(f"\n저장: {results_dir/'results_9_phase3_3R.json'}, {figures_dir/'phase3_directional.png'}, {figures_dir/'phase3_by_family.png'}")
    print(f"wav {len(rows)*4}개: {OUT_AUDIO_DIR}/phase3r_*.wav")
    print("★ 여기서 멈춥니다. directional_agreement CI를 확인한 뒤 블라인드 세트 준비 여부를 결정하세요.")


if __name__ == "__main__":
    main()
