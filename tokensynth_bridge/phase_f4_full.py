"""9차 F-4 (수정: MIDI 3변형 + 분산 분해) — 전체 재생성.

F-3(수정) 파일럿에서 신규 MIDI(3변형 평균)가 재구성 충실도를 유의하게 올림을
확인한 뒤 실행한다. F-2 조합 필터를 적용한 (소스×이펙트) 조합마다 MIDI 3변형
(상행/하행/지그재그, midi_gen.py) × 조건 2(c: e_wet 주입, d: e_dry_true 주입)를
전부 생성해 directional_agreement를 다시 낸다.

소스는 Phase 3-3R-1과 동일한 50소스 풀(out/caches/oat_emb_ts.npz, 10패밀리×5)에서
가져오되, e_wet/e_dry_true 임베딩은 그 캐시를 그대로 재사용한다(재추출 안 함).
MIDI는 소스당 한 번만 생성해 이펙트 간 공유한다(프레이즈는 소스 고유 속성이지
이펙트 속성이 아니므로).

주 지표는 MIDI 3변형을 평균한 값으로 보고하고, 변형별 값도 함께 낸다(재현성
확인용). directional_agreement의 총분산을 소스 간/MIDI 간(변형 정체성의 주효과)/
잔차로 이원분산분석(two-way ANOVA) 방식으로 분해한다.

기존 out/results/results_9_phase3_3R.json, out/audio/phase3r_*.wav는 건드리지
않는다 — 전부 새 파일(midi_new_v{0,1,2} 접미사).
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
from midi_gen import generate_midi_variants_for_source, N_VARIANTS
from phase_f2_filter import is_allowed, exclusion_reason

from tokensynth import TokenSynth, CLAP, DACDecoder

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False
INK_SECONDARY = "#52514e"; GRID_COLOR = "#e1e0d9"
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
OUT_AUDIO_DIR = REPO_ROOT / "out" / "audio"
NSYNTH_AUDIO_DIR = REPO_ROOT / "nsynth-test" / "audio"
NEW_MIDI_DIR = REPO_ROOT / "tokensynth_bridge" / "generated_midi"

GEN_SEED = 42
N_PER_FAMILY = 5
SELECT_SEED = 1  # Phase 3-3R-1과 동일 — 같은 50소스 재현
SAMPLE_RATE = 48000
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
PEAK_TARGET_A = 0.7
SILENCE_PEAK_THRESHOLD = 1e-4
TS_ENCODE_SR = 16000
NSYNTH_SOURCE_TYPES = {"acoustic", "electronic", "synthetic"}
EFFECT_NAMES = ["reverb", "distortion", "highshelf"]
LEVEL_RAW = {"reverb": [0.0, 0.5], "distortion": [0.0, 15.0], "highshelf": [-9.0, 9.0]}
REVERB_WET_LEVEL, REVERB_DRY_LEVEL, HIGHSHELF_CUTOFF_HZ = 0.4, 0.6, 4000.0
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


def two_way_anova_ss(grid):
    """grid: dict (combo_id, variant) -> value, 균형설계(모든 combo가 같은 변형 수)를 가정."""
    combos = sorted(set(k[0] for k in grid))
    variants = sorted(set(k[1] for k in grid))
    n_c, n_v = len(combos), len(variants)
    mat = np.array([[grid[(c, v)] for v in variants] for c in combos])  # (n_combo, n_variant)
    grand_mean = mat.mean()
    combo_means = mat.mean(axis=1)
    variant_means = mat.mean(axis=0)
    ss_total = float(np.sum((mat - grand_mean) ** 2))
    ss_source = float(n_v * np.sum((combo_means - grand_mean) ** 2))
    ss_midi = float(n_c * np.sum((variant_means - grand_mean) ** 2))
    ss_residual = ss_total - ss_source - ss_midi
    return {
        "ss_total": ss_total, "ss_source_between": ss_source, "ss_midi_variant": ss_midi, "ss_residual": ss_residual,
        "pct_source_between": ss_source / ss_total if ss_total > 1e-12 else None,
        "pct_midi_variant": ss_midi / ss_total if ss_total > 1e-12 else None,
        "pct_residual": ss_residual / ss_total if ss_total > 1e-12 else None,
        "n_combos": n_c, "n_variants": n_v,
    }


def main():
    parser = argparse.ArgumentParser(description="9차 F-4(수정) — 조합필터+MIDI3변형 전체 재생성")
    parser.add_argument("--oat-emb-ts", type=str, default="out/caches/oat_emb_ts.npz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.out)
    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("oat_emb_ts.npz 로딩 및 소스 목록 재현...")
    d = np.load(args.oat_emb_ts, allow_pickle=False)
    emb = d["emb"]
    family_arr = d["instrument_family"]
    full_selected = select_sources(NSYNTH_AUDIO_DIR, N_SOURCES_PER_FAMILY_TARGET, N_SOURCES_PER_FAMILY_MIN, RESELECT_SEED)
    assert len(full_selected) == emb.shape[0]

    rng = np.random.RandomState(SELECT_SEED)
    families_sorted = sorted(set(family_arr.tolist()))
    chosen_positions = []
    for fam in families_sorted:
        fam_positions = np.where(family_arr == fam)[0]
        chosen = rng.choice(fam_positions, size=min(N_PER_FAMILY, len(fam_positions)), replace=False)
        chosen_positions.extend(sorted(chosen.tolist()))
    print(f"소스 {len(chosen_positions)}개 (3-3R-1과 동일)")

    # F-2 필터로 유효 (소스,이펙트) 조합 구성 + 제외 기록
    combos = []
    exclusions = []
    for pos in chosen_positions:
        fname, instrument, fam = full_selected[pos]
        for effect in EFFECT_NAMES:
            if is_allowed(effect, fam):
                combos.append((pos, effect))
            else:
                exclusions.append({"src_pos": pos, "family": fam, "effect": effect, "reason": exclusion_reason(effect, fam)})
    print(f"F-2 필터 적용 후 유효 조합 {len(combos)}개 (제외 {len(exclusions)}개)")
    n_jobs = len(combos) * N_VARIANTS * 2
    print(f"총 생성 {n_jobs}회 예상")

    print("CLAP/TokenSynth/DAC 로딩 중...")
    synth = TokenSynth.from_pretrained(aug=True, device=device)
    clap = CLAP(device=device)
    decoder = DACDecoder(device=device)

    # 소스당 MIDI 3변형 — 이펙트 간 공유(소스 속성)
    midi_cache = {}
    for pos in chosen_positions:
        fname, instrument, fam = full_selected[pos]
        _, pitch, velocity = parse_nsynth_filename(Path(fname))
        midi_cache[pos] = generate_midi_variants_for_source(pitch, velocity, seed=pitch, out_dir=NEW_MIDI_DIR, tag=f"pos{pos}_{fam}")

    rows = []
    done = 0
    for pos, effect in combos:
        fname, instrument, fam = full_selected[pos]
        ei = EFFECT_NAMES.index(effect)
        e_wet = emb[pos, ei, 2, :]
        e_dry_true = emb[pos, ei, 0, :]
        variant_da = {}
        for variant in range(N_VARIANTS):
            done += 1
            midi_path = midi_cache[pos][variant]["path"]
            tag = f"{pos}_{fam}_{effect}_v{variant}"

            tok_c = synthesize_from_embedding(synth, e_wet, midi_path, seed=GEN_SEED, normalize="none", top_k=args.top_k)
            with torch.no_grad():
                audio_c = decoder.decode(tok_c).cpu().numpy()
            audiofile.write(str(OUT_AUDIO_DIR / f"phase_f4_{tag}_c.wav"), audio_c, 16000)
            e_regen_c = embed_ts_from_16k(clap, device, audio_c).cpu().numpy().reshape(-1)

            tok_d = synthesize_from_embedding(synth, e_dry_true, midi_path, seed=GEN_SEED, normalize="none", top_k=args.top_k)
            with torch.no_grad():
                audio_d = decoder.decode(tok_d).cpu().numpy()
            audiofile.write(str(OUT_AUDIO_DIR / f"phase_f4_{tag}_d.wav"), audio_d, 16000)
            e_regen_d = embed_ts_from_16k(clap, device, audio_d).cpu().numpy().reshape(-1)

            v_generated = e_regen_d - e_regen_c
            v_original = e_dry_true - e_wet
            da = cos_np(v_generated, v_original)
            variant_da[variant] = da

            if done % 20 == 0 or done == n_jobs:
                elapsed = time.time() - t_start
                eta = elapsed / done * (n_jobs - done)
                print(f"  [{done}/{n_jobs}] {tag}: dir_agree={da:+.4f}  (경과 {elapsed/60:.1f}분, 잔여 {eta/60:.1f}분)")

        mean_da = float(np.mean(list(variant_da.values())))
        std_da = float(np.std(list(variant_da.values())))
        rows.append({
            "src_pos": pos, "family": fam, "effect": effect, "filename": fname,
            "directional_agreement_by_variant": variant_da,
            "directional_agreement_mean": mean_da, "directional_agreement_std_across_variants": std_da,
        })

    # ---- 집계 (변형 평균 기준) ----
    da_mean = np.array([r["directional_agreement_mean"] for r in rows])
    src_ids = np.array([r["src_pos"] for r in rows])
    overall_mean, overall_lo, overall_hi = bootstrap_ci_by_source(da_mean, src_ids, seed=0)

    by_effect = {}
    for effect in EFFECT_NAMES:
        sub = [r for r in rows if r["effect"] == effect]
        if not sub:
            continue
        vals = np.array([r["directional_agreement_mean"] for r in sub])
        srcs = np.array([r["src_pos"] for r in sub])
        m, lo, hi = bootstrap_ci_by_source(vals, srcs, seed=0)
        by_effect[effect] = {"mean": m, "ci95": [lo, hi], "n": len(sub)}

    by_family = {}
    for fam in families_sorted:
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        vals = np.array([r["directional_agreement_mean"] for r in sub])
        srcs = np.array([r["src_pos"] for r in sub])
        m, lo, hi = bootstrap_ci_by_source(vals, srcs, seed=0)
        by_family[fam] = {"mean": m, "ci95": [lo, hi], "n": len(sub)}

    # ---- 분산 분해 (이원분산분석) ----
    grid = {}
    for r in rows:
        combo_id = f"{r['src_pos']}_{r['effect']}"
        for variant, val in r["directional_agreement_by_variant"].items():
            grid[(combo_id, variant)] = val
    anova = two_way_anova_ss(grid)

    verdict = (
        "CI가 0을 넘고 양수 — 조건화가 의도한 방향으로 작동" if overall_lo > 0 else
        ("유의하게 음수 — 반대로 움직임" if overall_hi < 0 else "0 근처 — CI가 0 포함")
    )

    results = {
        "meta": {
            "n_combos": len(combos), "n_variants": N_VARIANTS, "n_jobs": n_jobs,
            "exclusions": exclusions, "n_excluded": len(exclusions),
            "gen_seed": GEN_SEED, "select_seed": SELECT_SEED, "top_k": args.top_k,
            "elapsed_sec": time.time() - t_start,
        },
        "depends_on_surrogate": "none",
        "rows": rows,
        "directional_agreement": {"overall": {"mean": overall_mean, "ci95": [overall_lo, overall_hi], "n": len(rows)},
                                   "by_effect": by_effect, "by_family": by_family, "verdict": verdict},
        "variance_decomposition": anova,
    }
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results_9_phase_f4.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.hist(da_mean, bins=30, color=COLORS["baseline"], zorder=3)
    ax.axvline(0, color=COLORS["null"], linestyle="--", linewidth=1, label="0")
    ax.axvline(overall_mean, color=COLORS["reverb"], linewidth=1.5, label=f"평균={overall_mean:.3f}")
    ax.axvspan(overall_lo, overall_hi, color=COLORS["reverb"], alpha=0.15, label=f"95% CI")
    ax.set_xlabel("directional_agreement (MIDI 3변형 평균)")
    ax.set_ylabel(f"빈도 ({len(rows)}조합)")
    ax.set_title("F-4 — 변위 방향 일치도 (조합필터+MIDI 3변형)")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_f4_directional.png")
    plt.close(fig)

    print("\n=== F-4 결과 요약 ===")
    print(f"directional_agreement(변형평균) 전체: 평균={overall_mean:+.4f}  CI=[{overall_lo:+.4f},{overall_hi:+.4f}]  n={len(rows)}")
    print(f"판정: {verdict}")
    print("\n이펙트별:")
    for e, v in by_effect.items():
        print(f"  {e:<12} 평균={v['mean']:+.4f}  CI={v['ci95']}  n={v['n']}")
    print("\n분산 분해(이원분산분석, SS 비율):")
    print(f"  소스 간(source_between) = {anova['pct_source_between']:.3f}")
    print(f"  MIDI 변형(midi_variant)  = {anova['pct_midi_variant']:.3f}")
    print(f"  잔차(residual)           = {anova['pct_residual']:.3f}")
    print(f"\n저장: {results_dir/'results_9_phase_f4.json'}, {figures_dir/'phase_f4_directional.png'}")
    print(f"wav {n_jobs}개: {OUT_AUDIO_DIR}/phase_f4_*.wav")
    print("★ 여기서 멈춥니다.")


if __name__ == "__main__":
    main()
