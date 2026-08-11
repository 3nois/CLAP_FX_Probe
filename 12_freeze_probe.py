"""CLAP FX Probe — 12_freeze_probe.py (5차 후속 과제 1: 프로브 R²를 freeze로 재층화)

pedalboard Reverb의 freeze_mode=1은 wet_level/room_size/damping/width를 완전히
무시한다(부동소수점까지 동일 — 11_fd_phase1_cache_diag.py에서 확인). 3차 LHS는
freeze_mode를 Bernoulli(0.5)로 뽑았으므로, reverb 25,600행 중 절반(12,463행)이
"이펙트가 사실상 안 걸린" 조건이었다. 3·4차가 인용해온 reverb 프로브 R²
(room_size 0.155, wet_level 0.056, damping 0.010)는 이 무효 표본에 의해
과소평가됐을 가능성이 있다.

재렌더링 없음 — 3차 embeddings.npz만 읽는다. 방법론은 04_probe.py의
multivariate_probe_for_effect(다변량 Ridge, source-level 부트스트랩 OOB CI)를
그대로 재사용한다. Y에서 freeze_mode는 뺀다 — freeze=0/1 부분표본 안에서는
freeze_mode가 상수라 회귀 타깃으로 의미가 없다.

비교 두 가지를 낸다:
  (a) freeze=0만 사용 (N이 절반으로 줄어듦) vs distortion/highshelf 전체 N
  (b) 세 이펙트를 공통 N(가장 작은 쪽에 맞춤)으로 무작위 서브샘플링한 공정 비교

결과 해석은 이 스크립트가 단정하지 않는다. README의 판정 기준표를 따를 것.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"reverb": "#2a78d6", "distortion": "#eb6834", "highshelf": "#1baf7a", "baseline": "#898781"}
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 04_probe.py와 동일한 방법론 (X, Y, groups를 직접 받도록 일반화)
# ---------------------------------------------------------------------------
def held_out_r2_multi(X, Y, groups, seed, n_splits=5, test_size=0.3):
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    scores = []
    for train_idx, test_idx in gss.split(X, Y, groups):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], Y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(r2_score(Y[test_idx], pred, multioutput="raw_values"))
    scores = np.array(scores)
    return scores.mean(axis=0), scores.std(axis=0)


def bootstrap_r2_ci_multi(X, Y, groups, seed, n_boot=500, ci=0.95):
    unique_srcs = np.unique(groups)
    n = len(unique_srcs)
    src_to_rows = {s: np.where(groups == s)[0] for s in unique_srcs}
    rng = np.random.RandomState(seed)

    all_scores = []
    for _ in range(n_boot):
        boot_srcs = rng.choice(unique_srcs, size=n, replace=True)
        oob_srcs = np.setdiff1d(unique_srcs, boot_srcs)
        if len(oob_srcs) < 3:
            continue
        train_idx = np.concatenate([src_to_rows[s] for s in boot_srcs])
        test_idx = np.concatenate([src_to_rows[s] for s in oob_srcs])
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], Y[train_idx])
        pred = model.predict(X[test_idx])
        all_scores.append(r2_score(Y[test_idx], pred, multioutput="raw_values"))

    all_scores = np.array(all_scores)
    lo = np.percentile(all_scores, (1 - ci) / 2 * 100, axis=0)
    hi = np.percentile(all_scores, (1 + ci) / 2 * 100, axis=0)
    return all_scores.mean(axis=0), lo, hi, int(len(all_scores))


def probe_xyz(X, Y, groups, param_names, seed, n_boot):
    r2_mean, r2_std = held_out_r2_multi(X, Y, groups, seed)
    boot_mean, ci_lo, ci_hi, n_boot_used = bootstrap_r2_ci_multi(X, Y, groups, seed, n_boot=n_boot)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(Y))
    r2_shuf_mean, _ = held_out_r2_multi(X, Y[perm], groups, seed)

    result = {}
    for i, pname in enumerate(param_names):
        result[pname] = {
            "probe_r2": float(r2_mean[i]), "probe_r2_std": float(r2_std[i]),
            "probe_r2_ci_low": float(ci_lo[i]), "probe_r2_ci_high": float(ci_hi[i]),
            "probe_r2_shuffled": float(r2_shuf_mean[i]),
            "n_rows": int(len(Y)), "n_sources": int(len(np.unique(groups))), "n_boot_used": n_boot_used,
        }
    return result


def subsample_rows(X, Y, groups, n_target, seed):
    n = len(Y)
    if n <= n_target:
        return X, Y, groups
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=n_target, replace=False)
    return X[idx], Y[idx], groups[idx]


def main():
    parser = argparse.ArgumentParser(description="5차 후속 과제 1 — freeze_mode로 재층화한 프로브 R² (재렌더링 없음)")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    out_dir = Path(args.out)
    d = np.load(emb_path, allow_pickle=False)
    config = json.load(open(emb_path.parent / "embed_config.json"))
    theta_slots = {e: tuple(v) for e, v in config["theta_slots"].items()}
    param_order = config["param_order"]

    reverb_mask = d["effect"] == "reverb"
    start, end = theta_slots["reverb"]
    reverb_theta = d["theta_norm"][reverb_mask][:, start:end]
    freeze_idx = param_order["reverb"].index("freeze_mode")
    freeze = reverb_theta[:, freeze_idx]
    cont_params = [p for p in param_order["reverb"] if p != "freeze_mode"]
    cont_idx = [param_order["reverb"].index(p) for p in cont_params]

    X_reverb_all = d["embeddings"][reverb_mask]
    Y_reverb_all = reverb_theta[:, cont_idx]
    groups_reverb_all = d["src_id"][reverb_mask]

    m0 = freeze == 0.0
    m1 = freeze == 1.0
    print(f"reverb 전체 {len(freeze)}행 — freeze=0: {m0.sum()}, freeze=1: {m1.sum()}")

    print("\n(a) 비matched — freeze=0 / freeze=1 / pooled(전체, freeze 열 제외) 프로브 계산 중...")
    res_freeze0 = probe_xyz(X_reverb_all[m0], Y_reverb_all[m0], groups_reverb_all[m0], cont_params, args.seed, args.n_boot)
    res_freeze1 = probe_xyz(X_reverb_all[m1], Y_reverb_all[m1], groups_reverb_all[m1], cont_params, args.seed, args.n_boot)
    res_pooled = probe_xyz(X_reverb_all, Y_reverb_all, groups_reverb_all, cont_params, args.seed, args.n_boot)

    def get_effect_full(effect_name):
        mask = d["effect"] == effect_name
        s, e = theta_slots[effect_name]
        X = d["embeddings"][mask]
        Y = d["theta_norm"][mask][:, s:e]
        groups = d["src_id"][mask]
        return X, Y, groups

    X_dist, Y_dist, g_dist = get_effect_full("distortion")
    X_hs, Y_hs, g_hs = get_effect_full("highshelf")
    print("distortion / highshelf 프로브 재계산 중 (동일 방법론, 참고용)...")
    res_dist_full = probe_xyz(X_dist, Y_dist, g_dist, param_order["distortion"], args.seed, args.n_boot)
    res_hs_full = probe_xyz(X_hs, Y_hs, g_hs, param_order["highshelf"], args.seed, args.n_boot)

    print("\n(b) matched-N 공정 비교 계산 중...")
    n_freeze0 = int(m0.sum())
    n_dist = X_dist.shape[0]
    n_hs = X_hs.shape[0]
    n_matched = min(n_freeze0, n_dist, n_hs)
    print(f"N 후보: reverb freeze=0={n_freeze0}, distortion={n_dist}, highshelf={n_hs} -> matched N={n_matched}")

    Xr, Yr, gr = subsample_rows(X_reverb_all[m0], Y_reverb_all[m0], groups_reverb_all[m0], n_matched, args.seed)
    Xd, Yd, gd = subsample_rows(X_dist, Y_dist, g_dist, n_matched, args.seed)
    Xh, Yh, gh = subsample_rows(X_hs, Y_hs, g_hs, n_matched, args.seed)

    res_freeze0_matched = probe_xyz(Xr, Yr, gr, cont_params, args.seed, args.n_boot)
    res_dist_matched = probe_xyz(Xd, Yd, gd, param_order["distortion"], args.seed, args.n_boot)
    res_hs_matched = probe_xyz(Xh, Yh, gh, param_order["highshelf"], args.seed, args.n_boot)

    # ---- 이펙트 간 순서 ----
    def mean_r2(res, params):
        return float(np.mean([res[p]["probe_r2"] for p in params]))

    order_before = sorted(
        [("reverb(pooled,3차와동일조건)", mean_r2(res_pooled, cont_params)),
         ("distortion", mean_r2(res_dist_full, param_order["distortion"])),
         ("highshelf", mean_r2(res_hs_full, param_order["highshelf"]))],
        key=lambda x: -x[1])
    order_after_freeze0 = sorted(
        [("reverb(freeze=0)", mean_r2(res_freeze0, cont_params)),
         ("distortion", mean_r2(res_dist_full, param_order["distortion"])),
         ("highshelf", mean_r2(res_hs_full, param_order["highshelf"]))],
        key=lambda x: -x[1])
    order_matched = sorted(
        [("reverb(freeze=0,matched)", mean_r2(res_freeze0_matched, cont_params)),
         ("distortion(matched)", mean_r2(res_dist_matched, param_order["distortion"])),
         ("highshelf(matched)", mean_r2(res_hs_matched, param_order["highshelf"]))],
        key=lambda x: -x[1])

    # ---- 그림 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    ax = axes[0]
    x = np.arange(len(cont_params))
    width = 0.25
    for off, (label, res, color) in enumerate([
        ("pooled(3차와 동일)", res_pooled, COLORS["baseline"]),
        ("freeze=0", res_freeze0, COLORS["reverb"]),
        ("freeze=1", res_freeze1, "#c3c2b7"),
    ]):
        r2s = [max(res[p]["probe_r2"], 0.0) for p in cont_params]
        lo = [max(res[p]["probe_r2_ci_low"], 0.0) for p in cont_params]
        hi = [res[p]["probe_r2_ci_high"] for p in cont_params]
        yerr = np.array([[r - l for r, l in zip(r2s, lo)], [h - r for r, h in zip(r2s, hi)]])
        yerr = np.clip(yerr, 0, None)
        ax.bar(x + (off - 1) * width, r2s, width, yerr=yerr, capsize=2, label=label, color=color, zorder=3)
    resolution_floor = None
    results_path = out_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            r4 = json.load(f)
        resolution_floor = r4.get("resolution_floor", {}).get("value")
    if resolution_floor is not None:
        ax.axhline(resolution_floor, color="black", linestyle="--", linewidth=1, label=f"4차 해상도 바닥={resolution_floor:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(cont_params, rotation=20, ha="right")
    ax.set_ylabel("Held-out R²")
    ax.set_title("reverb 프로브 R² — freeze 층화 전후")
    ax.legend(frameon=False, fontsize=7)
    style_axis(ax)

    ax = axes[1]
    labels = ["reverb\n(freeze=0,\nmatched)", "distortion\n(matched)", "highshelf\n(matched)"]
    vals = [mean_r2(res_freeze0_matched, cont_params), mean_r2(res_dist_matched, param_order["distortion"]),
            mean_r2(res_hs_matched, param_order["highshelf"])]
    colors3 = [COLORS["reverb"], COLORS["distortion"], COLORS["highshelf"]]
    ax.bar(np.arange(3), vals, color=colors3, zorder=3)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"평균 R² (실제 파라미터, N={n_matched} 공통)")
    ax.set_title("이펙트 간 순서 — 공정 비교 (matched N)")
    style_axis(ax)

    fig.suptitle(f"과제 1 — freeze_mode 재층화 (reverb freeze=0 N={n_freeze0}, distortion N={n_dist}, highshelf N={n_hs})")
    fig.tight_layout()
    fig.savefig(out_dir / "freeze_stratified_probe.png")
    plt.close(fig)

    # ---- results_6.json ----
    results6_path = out_dir / "results_6.json"
    results6 = {}
    if results6_path.exists():
        with open(results6_path) as f:
            results6 = json.load(f)

    results6.setdefault("meta", {})
    results6["meta"].update({
        "task1_seed": args.seed, "task1_n_boot": args.n_boot,
        "task1_n_reverb_total": int(len(freeze)), "task1_n_freeze0": int(m0.sum()), "task1_n_freeze1": int(m1.sum()),
        "task1_n_distortion": n_dist, "task1_n_highshelf": n_hs, "task1_n_matched": n_matched,
        "task1_resolution_floor_ref_from_results5": resolution_floor,
    })
    results6["task1_freeze_stratified_probe"] = {
        "reverb_pooled_4cont_only": res_pooled,
        "reverb_freeze0": res_freeze0,
        "reverb_freeze1": res_freeze1,
        "distortion_full": res_dist_full,
        "highshelf_full": res_hs_full,
        "matched_N": {
            "n_matched": n_matched,
            "reverb_freeze0": res_freeze0_matched,
            "distortion": res_dist_matched,
            "highshelf": res_hs_matched,
        },
        "effect_order": {
            "before_freeze_split_pooled_vs_full": [o[0] for o in order_before],
            "after_freeze0_vs_full": [o[0] for o in order_after_freeze0],
            "matched_N": [o[0] for o in order_matched],
        },
    }
    with open(results6_path, "w") as f:
        json.dump(results6, f, indent=2, ensure_ascii=False)

    # ---- 콘솔 보고 ----
    print("\n=== 과제 1 결과 ===")
    print(f"\nreverb 연속 파라미터 4개 (freeze_mode 제외) R² 비교:")
    print(f"{'param':<12}{'pooled(전체)':>16}{'freeze=0':>16}{'freeze=1':>16}")
    for p in cont_params:
        print(f"{p:<12}{res_pooled[p]['probe_r2']:>16.4f}{res_freeze0[p]['probe_r2']:>16.4f}{res_freeze1[p]['probe_r2']:>16.4f}")
    print(f"\n(참고) 4차 3차 원본 R² (5열 포함 버전, resolution_floor={resolution_floor}):")
    print("  wet_level=0.0564 room_size=0.1549 damping=0.0099 width=0.0087")

    print(f"\nCI (freeze=0, N={n_freeze0}):")
    for p in cont_params:
        r = res_freeze0[p]
        print(f"  {p:<12} R²={r['probe_r2']:.4f}  CI=[{r['probe_r2_ci_low']:.4f}, {r['probe_r2_ci_high']:.4f}]")

    print(f"\nmatched-N (N={n_matched}) 이펙트 간 평균 R²:")
    print(f"  reverb(freeze=0,matched)  = {mean_r2(res_freeze0_matched, cont_params):.4f}")
    print(f"  distortion(matched)       = {mean_r2(res_dist_matched, param_order['distortion']):.4f}")
    print(f"  highshelf(matched)        = {mean_r2(res_hs_matched, param_order['highshelf']):.4f}")
    print(f"\n순서(전체 N, freeze 분리 전 pooled): {[o[0] for o in order_before]}")
    print(f"순서(freeze=0 vs 나머지 전체 N):      {[o[0] for o in order_after_freeze0]}")
    print(f"순서(matched N):                      {[o[0] for o in order_matched]}")

    print(f"\n저장: {results6_path}, {out_dir / 'freeze_stratified_probe.png'}")
    print("★ 여기서 멈춥니다. 결과를 판정한 뒤 과제 2 진행 여부를 결정하세요.")


if __name__ == "__main__":
    main()
