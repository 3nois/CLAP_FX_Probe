"""CLAP FX Probe — 20_family_cosine_oat.py (6차 후속: 원래 질문 — "악기마다 손잡이가 다른가")

out/oat_emb.npz(19_oat_render.py, 1,200소스 x 3레벨 x 3이펙트, 조건A)만 읽는다.
재렌더링 0회, 야코비안·대리모델·유한차분 전부 쓰지 않는다 — 2차 임베딩에서 뺄셈만
한다(v = e(최고 레벨) − e(최저 레벨)). 모든 결과에 depends_on_surrogate="none".

  과제 A: 소스별 차이 벡터 (+ highshelf 반쪽 스윙, 비선형성 보조 확인)
  과제 B: within/between/random 분해 — "악기마다 다른가"의 직접적 답 (★ 주 검정)
  과제 C: 패밀리 평균 코사인 + split-half 감쇠 보정, 3차 값과 비교

결과 해석은 이 스크립트가 단정하지 않는다. README 6차 후속 절의 판정 기준표를 따를 것.
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
COLORS = {"reverb": "#2a78d6", "distortion": "#eb6834", "highshelf": "#1baf7a", "random": "#898781", "null": "#e34948"}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


ZERO_NORM_EPS = 1e-4
EFFECT_NAMES = ["reverb", "distortion", "highshelf"]


def unit(v, axis=-1, eps=1e-12):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.clip(n, eps, None)


def cosine_matrix(vecs):
    """vecs: (n, 512) 이미 단위벡터. 반환 (n,n) 코사인 행렬."""
    return vecs @ vecs.T


def bootstrap_within_between(unit_vecs, family_labels, n_boot, seed):
    """소스 단위 부트스트랩. 반환: within_mean_dist, between_mean_dist, gap_dist (각 len=n_boot)."""
    n = unit_vecs.shape[0]
    rng = np.random.RandomState(seed)
    fam_arr = np.array(family_labels)
    within_means, between_means, gaps = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        V = unit_vecs[idx]
        F = fam_arr[idx]
        C = cosine_matrix(V)
        iu = np.triu_indices(n, k=1)
        same = F[iu[0]] == F[iu[1]]
        w = C[iu][same]
        b = C[iu][~same]
        if len(w) == 0 or len(b) == 0:
            continue
        within_means.append(float(w.mean()))
        between_means.append(float(b.mean()))
        gaps.append(float(w.mean() - b.mean()))
    return np.array(within_means), np.array(between_means), np.array(gaps)


def ci95(arr):
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def split_half_correction(v_by_family, n_reps, seed):
    """v_by_family: {family: (n_i, 512) raw(비정규화) v 배열}.
    반환: self_cosine_dist[fam] (len n_reps), corrected_dist[(fam_i,fam_j)] (len n_reps),
          cross_cosine[(fam_i,fam_j)] (scalar, 전체 평균 기준)."""
    families = sorted(v_by_family.keys())
    full_mean = {f: v_by_family[f].mean(axis=0) for f in families}
    cross_cosine = {}
    for fi, fj in itertools.combinations(families, 2):
        a, b = full_mean[fi], full_mean[fj]
        cross_cosine[(fi, fj)] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    rng = np.random.RandomState(seed)
    self_cosine_dist = {f: [] for f in families}
    corrected_dist = {(fi, fj): [] for fi, fj in itertools.combinations(families, 2)}
    for r in range(n_reps):
        half_mean = {}
        for f in families:
            arr = v_by_family[f]
            n_i = arr.shape[0]
            perm = rng.permutation(n_i)
            half_a_idx, half_b_idx = perm[: n_i // 2], perm[n_i // 2:]
            ma = arr[half_a_idx].mean(axis=0)
            mb = arr[half_b_idx].mean(axis=0)
            self_c = float(np.dot(ma, mb) / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-12))
            self_cosine_dist[f].append(self_c)
            half_mean[f] = self_c
        for fi, fj in itertools.combinations(families, 2):
            si, sj = half_mean[fi], half_mean[fj]
            denom = np.sqrt(max(si, 1e-6) * max(sj, 1e-6)) if si > 0 and sj > 0 else np.nan
            corrected = cross_cosine[(fi, fj)] / denom if denom and denom > 1e-6 else np.nan
            corrected_dist[(fi, fj)].append(corrected)
    self_cosine_dist = {f: np.array(v) for f, v in self_cosine_dist.items()}
    corrected_dist = {k: np.array(v) for k, v in corrected_dist.items()}
    return cross_cosine, self_cosine_dist, corrected_dist, families


def split_half_correction_unitavg(v_by_family, n_reps, seed):
    """결함 21 수정 — split_half_correction과 동일하지만 family 평균을
    '먼저 정규화(unit) -> 평균'으로 낸다(raw 평균이 아님). 나란히 비교용.
    v_by_family: {family: (n_i, 512) raw 배열} — 내부에서 unit()을 적용."""
    families = sorted(v_by_family.keys())
    v_unit_by_family = {f: unit(v_by_family[f]) for f in families}
    full_mean = {f: v_unit_by_family[f].mean(axis=0) for f in families}
    cross_cosine = {}
    for fi, fj in itertools.combinations(families, 2):
        a, b = full_mean[fi], full_mean[fj]
        cross_cosine[(fi, fj)] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    rng = np.random.RandomState(seed)
    self_cosine_dist = {f: [] for f in families}
    corrected_dist = {(fi, fj): [] for fi, fj in itertools.combinations(families, 2)}
    for r in range(n_reps):
        half_mean = {}
        for f in families:
            arr = v_unit_by_family[f]
            n_i = arr.shape[0]
            perm = rng.permutation(n_i)
            half_a_idx, half_b_idx = perm[: n_i // 2], perm[n_i // 2:]
            ma = arr[half_a_idx].mean(axis=0)
            mb = arr[half_b_idx].mean(axis=0)
            self_c = float(np.dot(ma, mb) / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-12))
            self_cosine_dist[f].append(self_c)
            half_mean[f] = self_c
        for fi, fj in itertools.combinations(families, 2):
            si, sj = half_mean[fi], half_mean[fj]
            denom = np.sqrt(max(si, 1e-6) * max(sj, 1e-6)) if si > 0 and sj > 0 else np.nan
            corrected = cross_cosine[(fi, fj)] / denom if denom and denom > 1e-6 else np.nan
            corrected_dist[(fi, fj)].append(corrected)
    self_cosine_dist = {f: np.array(v) for f, v in self_cosine_dist.items()}
    corrected_dist = {k: np.array(v) for k, v in corrected_dist.items()}
    return cross_cosine, self_cosine_dist, corrected_dist, families


def main():
    parser = argparse.ArgumentParser(description="6차 후속 — family cosine (OAT diff-vector, 원래 질문)")
    parser.add_argument("--oat-emb", type=str, default="out/oat_emb.npz")
    parser.add_argument("--results", type=str, default="out/results.json")
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--n-splithalf-reps", type=int, default=100)
    parser.add_argument("--n-random-pairs", type=int, default=20000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    print(f"로딩 중: {args.oat_emb}")
    d = np.load(args.oat_emb, allow_pickle=False)
    emb = d["emb"]  # (n, 3, 3, 512)
    src_id = d["src_id"]
    family = d["instrument_family"]
    n_sources = emb.shape[0]
    families_sorted = sorted(set(family.tolist()))
    print(f"소스 {n_sources}개, 패밀리 {len(families_sorted)}개: {families_sorted}")

    # ---- 과제 A: 차이 벡터 ----
    v_full = {}       # effect -> (n,512) raw
    v_half_pos = {}   # highshelf만: 0->+9
    v_half_neg = {}   # highshelf만: -9->0
    nonlinearity = {}  # effect -> cos(e1-e0, e2-e1) per source
    excluded_zero_norm = {}
    for ei, effect in enumerate(EFFECT_NAMES):
        e0, e1, e2 = emb[:, ei, 0], emb[:, ei, 1], emb[:, ei, 2]
        v = e2 - e0
        norms = np.linalg.norm(v, axis=1)
        keep = norms > ZERO_NORM_EPS
        excluded_zero_norm[effect] = int((~keep).sum())
        v_full[effect] = v[keep]
        d1 = e1 - e0
        d2 = e2 - e1
        n1, n2 = np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1)
        nl_keep = (n1 > ZERO_NORM_EPS) & (n2 > ZERO_NORM_EPS)
        cos_nl = np.sum(d1[nl_keep] * d2[nl_keep], axis=1) / (n1[nl_keep] * n2[nl_keep])
        nonlinearity[effect] = cos_nl

    v_pos = emb[:, 2, 2] - emb[:, 2, 1]  # highshelf idx=2: 0->+9
    v_neg = emb[:, 2, 1] - emb[:, 2, 0]  # highshelf: -9->0
    v_half_pos["highshelf"] = v_pos
    v_half_neg["highshelf"] = v_neg
    cos_pos_neg = float(np.mean([
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        for a, b in zip(v_pos, v_neg) if np.linalg.norm(a) > ZERO_NORM_EPS and np.linalg.norm(b) > ZERO_NORM_EPS
    ]))

    # family label 배열은 excluded 소스를 뺀 버전으로 effect별로 따로 관리
    family_kept = {}
    for ei, effect in enumerate(EFFECT_NAMES):
        e0, e2 = emb[:, ei, 0], emb[:, ei, 2]
        norms = np.linalg.norm(e2 - e0, axis=1)
        keep = norms > ZERO_NORM_EPS
        family_kept[effect] = family[keep]

    print("과제 A 완료:")
    for effect in EFFECT_NAMES:
        print(f"  {effect}: v_full n={v_full[effect].shape[0]} (제외 {excluded_zero_norm[effect]}), "
              f"비선형성 cos(d1,d2) median={np.median(nonlinearity[effect]):.4f}")
    print(f"  highshelf half-swing cos(v+, v-) mean={cos_pos_neg:.4f}")

    # ---- 과제 B: within/between/random 분해 (★ 주 검정) ----
    print("\n과제 B 계산 중 (within/between/random, source-level 부트스트랩)...")
    within_between_result = {}
    rng_global = np.random.RandomState(args.seed)
    # 무작위 기준선(공용, 이펙트 무관 — 512차원 벡터 기하 특성)
    rv = rng_global.normal(size=(args.n_random_pairs, 2, 512))
    rv = rv / np.linalg.norm(rv, axis=2, keepdims=True)
    random_cos = np.sum(rv[:, 0] * rv[:, 1], axis=1)

    for effect in EFFECT_NAMES:
        U = unit(v_full[effect])
        fam_labels = family_kept[effect]
        w_dist, b_dist, gap_dist = bootstrap_within_between(U, fam_labels, args.n_boot, args.seed)
        n = U.shape[0]
        C = cosine_matrix(U)
        iu = np.triu_indices(n, k=1)
        same = fam_labels[iu[0]] == fam_labels[iu[1]]
        w_point = C[iu][same]
        b_point = C[iu][~same]
        within_between_result[effect] = {
            "n_sources": int(n),
            "within_point_mean": float(w_point.mean()), "within_point_n_pairs": int(len(w_point)),
            "between_point_mean": float(b_point.mean()), "between_point_n_pairs": int(len(b_point)),
            "within_boot_mean": float(w_dist.mean()), "within_boot_ci": list(ci95(w_dist)),
            "between_boot_mean": float(b_dist.mean()), "between_boot_ci": list(ci95(b_dist)),
            "gap_boot_mean": float(gap_dist.mean()), "gap_boot_ci": list(ci95(gap_dist)),
            "random_baseline_mean": float(random_cos.mean()), "random_baseline_std": float(random_cos.std()),
            "verdict": (
                "within > between — 악기 패밀리 구조 실재"
                if ci95(gap_dist)[0] > 0 else
                "within ≈ between (CI가 0 포함) — 패밀리 무관, 소스 단위 변동"
            ),
        }
        print(f"  {effect}: within={w_dist.mean():.4f} CI={ci95(w_dist)}  between={b_dist.mean():.4f} CI={ci95(b_dist)}  "
              f"gap={gap_dist.mean():.4f} CI={ci95(gap_dist)}  -> {within_between_result[effect]['verdict']}")

    # ---- 그림: within/between/random ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
    for ax, effect in zip(axes, EFFECT_NAMES):
        r = within_between_result[effect]
        U = unit(v_full[effect])
        fam_labels = family_kept[effect]
        n = U.shape[0]
        C = cosine_matrix(U)
        iu = np.triu_indices(n, k=1)
        same = fam_labels[iu[0]] == fam_labels[iu[1]]
        w_point, b_point = C[iu][same], C[iu][~same]
        ax.hist(random_cos, bins=60, density=True, alpha=0.4, color=COLORS["random"], label="random")
        ax.hist(b_point, bins=60, density=True, alpha=0.5, color="#c3c2b7", label="between")
        ax.hist(w_point, bins=60, density=True, alpha=0.5, color=COLORS[effect], label="within")
        ax.set_title(f"{effect}\ngap={r['gap_boot_mean']:.3f} CI={[round(x,3) for x in r['gap_boot_ci']]}", fontsize=9)
        ax.set_xlabel("cosine")
        ax.legend(frameon=False, fontsize=7)
        style_axis(ax)
    axes[0].set_ylabel("밀도")
    fig.suptitle("과제 B — within/between/random 분포 (소스 쌍별 코사인, 원래 질문의 직접적 답)")
    fig.tight_layout()
    fig.savefig(out_dir / "family_within_between.png")
    plt.close(fig)

    # ---- 과제 C: 패밀리 평균 코사인 + split-half 보정 ----
    print("\n과제 C 계산 중 (split-half 감쇠 보정)...")
    family_cosine_result = {}
    family_cosine_matrix = {}
    for effect in EFFECT_NAMES:
        v_by_family = {}
        fam_labels = family_kept[effect]
        vecs = v_full[effect]
        for f in families_sorted:
            mask = fam_labels == f
            if mask.sum() >= 2:
                v_by_family[f] = vecs[mask]
        cross_cosine, self_cosine_dist, corrected_dist, fams_used = split_half_correction(v_by_family, args.n_splithalf_reps, args.seed)

        self_summary = {f: {"mean": float(v.mean()), "ci": list(ci95(v)), "n_sources": int(v_by_family[f].shape[0])}
                         for f, v in self_cosine_dist.items()}
        low_reliability = [f for f, s in self_summary.items() if s["mean"] < 0.7]

        pair_summary = {}
        for (fi, fj), arr in corrected_dist.items():
            valid = arr[~np.isnan(arr)]
            pair_summary[f"{fi}|{fj}"] = {
                "cross_cosine_raw": cross_cosine[(fi, fj)],
                "corrected_mean": float(np.nanmean(arr)) if len(valid) else None,
                "corrected_ci": list(ci95(valid)) if len(valid) else [None, None],
            }

        corrected_vals_all = [pair_summary[k]["corrected_mean"] for k in pair_summary if pair_summary[k]["corrected_mean"] is not None]
        corrected_vals_excl = [
            pair_summary[f"{fi}|{fj}"]["corrected_mean"] for fi, fj in itertools.combinations(fams_used, 2)
            if fi not in low_reliability and fj not in low_reliability and pair_summary[f"{fi}|{fj}"]["corrected_mean"] is not None
        ]

        family_cosine_result[effect] = {
            "self_cosine": self_summary,
            "low_reliability_families_self_below_0.7": low_reliability,
            "pairwise": pair_summary,
            "corrected_mean_all_pairs": float(np.mean(corrected_vals_all)) if corrected_vals_all else None,
            "corrected_mean_excl_low_reliability": float(np.mean(corrected_vals_excl)) if corrected_vals_excl else None,
            "raw_cross_cosine_mean_all_pairs": float(np.mean([cross_cosine[k] for k in cross_cosine])),
        }

        mat = np.full((len(fams_used), len(fams_used)), np.nan)
        mat_corrected = np.full((len(fams_used), len(fams_used)), np.nan)
        for a_i, fi in enumerate(fams_used):
            mat[a_i, a_i] = 1.0
            mat_corrected[a_i, a_i] = 1.0
        for (fi, fj), cval in cross_cosine.items():
            i_, j_ = fams_used.index(fi), fams_used.index(fj)
            mat[i_, j_] = mat[j_, i_] = cval
            mat_corrected[i_, j_] = mat_corrected[j_, i_] = pair_summary[f"{fi}|{fj}"]["corrected_mean"]
        family_cosine_matrix[effect] = {"families": fams_used, "raw": mat.tolist(), "corrected": mat_corrected.tolist()}

        print(f"  {effect}: 원값 평균={family_cosine_result[effect]['raw_cross_cosine_mean_all_pairs']:.4f}  "
              f"보정값 평균={family_cosine_result[effect]['corrected_mean_all_pairs']:.4f}  "
              f"저신뢰 패밀리(self<0.7)={low_reliability}")

    # ---- 그림: 히트맵 ----
    fig, axes = plt.subplots(3, 2, figsize=(11, 15), dpi=150)
    for row, effect in enumerate(EFFECT_NAMES):
        fams = family_cosine_matrix[effect]["families"]
        for col, key in enumerate(["raw", "corrected"]):
            ax = axes[row, col]
            mat = np.array(family_cosine_matrix[effect][key])
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=60, ha="right", fontsize=6)
            ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=6)
            ax.set_title(f"{effect} — {'원값' if key=='raw' else '보정값'}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("과제 C — 패밀리 간 코사인 히트맵 (원값 vs split-half 보정값)")
    fig.tight_layout()
    fig.savefig(out_dir / "family_cosine_heatmap.png")
    plt.close(fig)

    # ---- 3차 값과 비교 ----
    comparison_with_3rd = {}
    third_round_rep_param = {"reverb": "reverb.room_size", "distortion": "distortion.drive_db", "highshelf": "highshelf.gain_db"}
    try:
        with open(args.results) as f:
            r3 = json.load(f)
        for effect, pname in third_round_rep_param.items():
            v3 = r3["params"].get(pname, {}).get("jacobian_family_cosine", {})
            comparison_with_3rd[effect] = {
                "3rd_round_representative_param": pname,
                "3rd_round_cosine_mean": v3.get("cosine_mean"),
                "3rd_round_basis": "surrogate Jacobian (jacrev), 대리모델 의존 — 불신 판정",
                "6th_round_oat_raw_mean": family_cosine_result[effect]["raw_cross_cosine_mean_all_pairs"],
                "6th_round_oat_corrected_mean": family_cosine_result[effect]["corrected_mean_all_pairs"],
                "6th_round_basis": "OAT 차이벡터(레벨 극단값), 대리모델·야코비안 미사용",
            }
    except FileNotFoundError:
        comparison_with_3rd = {"error": f"{args.results} 없음"}

    # ---- 그림: 원값 vs 보정값 vs 3차 ----
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    x = np.arange(len(EFFECT_NAMES))
    width = 0.25
    raw_vals = [family_cosine_result[e]["raw_cross_cosine_mean_all_pairs"] for e in EFFECT_NAMES]
    corr_vals = [family_cosine_result[e]["corrected_mean_all_pairs"] for e in EFFECT_NAMES]
    third_vals = [comparison_with_3rd.get(e, {}).get("3rd_round_cosine_mean") for e in EFFECT_NAMES]
    ax.bar(x - width, raw_vals, width, label="6차 OAT 원값", color="#c3c2b7", zorder=3)
    ax.bar(x, corr_vals, width, label="6차 OAT 보정값", color=[COLORS[e] for e in EFFECT_NAMES], zorder=3)
    ax.bar(x + width, [v if v is not None else 0 for v in third_vals], width, label="3차 (대리모델, 불신)", color="#e0e0e0", edgecolor="black", zorder=3)
    ax.axhline(0.5, color=COLORS["null"], linestyle="--", linewidth=1, label="0.5 (악기별 손잡이 필요 경계)")
    ax.axhline(0.8, color="#2a9d5c", linestyle=":", linewidth=1, label="0.8 (공통 손잡이 경계)")
    ax.set_xticks(x); ax.set_xticklabels(EFFECT_NAMES)
    ax.set_ylabel("family 간 평균 코사인")
    ax.set_title("과제 C — 원값 vs 감쇠보정값 vs 3차(대리모델) 비교")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "family_cosine_corrected.png")
    plt.close(fig)

    # ---- 저장 ----
    results7 = {
        "meta": {
            "source_axis": {
                "reverb": {"swept_param": "room_size", "levels": emb.shape, "note": "wet_level=0.4/dry_level=0.6/freeze_mode=0 고정"},
                "distortion": {"swept_param": "drive_db"},
                "highshelf": {"swept_param": "gain_db", "note": "cutoff=4000Hz 고정, q 기본값"},
            },
            "n_sources_total": int(n_sources),
            "n_sources_per_family_target": None,
            "n_excluded_zero_norm": excluded_zero_norm,
            "condition": "A (조건C 아님 — 1~5차와 비교 가능하도록)",
            "highshelf_half_swing_cos_pos_neg": cos_pos_neg,
            "seed": args.seed, "n_boot": args.n_boot, "n_splithalf_reps": args.n_splithalf_reps,
        },
        "depends_on_surrogate": "none",
        "nonlinearity": {
            effect: {
                "cos_d1_d2_median": float(np.median(nonlinearity[effect])),
                "cos_d1_d2_mean": float(np.mean(nonlinearity[effect])),
                "cos_d1_d2_std": float(np.std(nonlinearity[effect])),
                "n": int(len(nonlinearity[effect])),
                "depends_on_surrogate": "none",
            } for effect in EFFECT_NAMES
        },
        "within_between": {**within_between_result, "depends_on_surrogate": "none"},
        "family_cosine": {**family_cosine_result, "depends_on_surrogate": "none"},
        "family_cosine_matrix": family_cosine_matrix,
        "comparison_with_3rd": comparison_with_3rd,
    }
    results7_path = out_dir / "results_7.json"
    with open(results7_path, "w") as f:
        json.dump(results7, f, indent=2, ensure_ascii=False)

    print("\n=== 과제 B 요약 (원래 질문의 직접적 답) ===")
    for effect in EFFECT_NAMES:
        r = within_between_result[effect]
        print(f"  {effect:<12} within={r['within_boot_mean']:.4f} CI={r['within_boot_ci']}  "
              f"between={r['between_boot_mean']:.4f} CI={r['between_boot_ci']}  "
              f"gap CI={r['gap_boot_ci']}  -> {r['verdict']}")
    print(f"\n저장: {results7_path}, {out_dir / 'family_within_between.png'}, "
          f"{out_dir / 'family_cosine_heatmap.png'}, {out_dir / 'family_cosine_corrected.png'}")
    print("★ 과제 B를 먼저 완료했습니다. 여기서 멈춥니다.")


if __name__ == "__main__":
    main()
