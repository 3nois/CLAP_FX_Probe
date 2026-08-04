"""CLAP FX Probe — 04_probe.py (3차)

이펙트별로 임베딩(512) → θ 전체(5/1/3차원)를 동시에 회귀하는 다변량 Ridge 프로브.
파라미터별 R²와 source-level 부트스트랩 95% CI를 낸다 — 어느 성분이 읽히고 어느
성분이 안 읽히는지가 한 번에 나온다.

★ width(reverb 스테레오 폭)는 음성 통제다. 파이프라인이 오디오를 모노로 변환하므로
파형이 동일해야 하고, R²(width) ≈ 0이어야 한다. 유의하게 0을 초과하면 파이프라인
누수이며 1·2차 결과까지 재검토 대상이다 — width_control.png와 결과를 반드시 먼저 볼 것.

악기 패밀리 통제(NMI 주 지표, 7클래스 서브샘플)는 2차 설계를 그대로 유지한다.

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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, r2_score
from sklearn.model_selection import GroupShuffleSplit

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

EFFECTS = ["reverb", "distortion", "highshelf"]
COLORS = {"reverb": "#2a78d6", "distortion": "#eb6834", "highshelf": "#1baf7a", "baseline": "#898781"}
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
NEGATIVE_CONTROL_COLOR = "#e34948"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load_embeddings(path: Path):
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_config(out_dir: Path):
    with open(out_dir / "embed_config.json") as f:
        return json.load(f)


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
    """소스 단위 부트스트랩(복원추출) + out-of-bag 평가. 열(파라미터)별로 CI를 낸다."""
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


def multivariate_probe_for_effect(effect_name, d, theta_slots, param_order, seed, n_boot):
    mask = d["effect"] == effect_name
    X = d["embeddings"][mask]
    start, end = theta_slots[effect_name]
    Y = d["theta_norm"][mask][:, start:end]
    groups = d["src_id"][mask]

    r2_mean, r2_std = held_out_r2_multi(X, Y, groups, seed)
    boot_mean, ci_lo, ci_hi, n_boot_used = bootstrap_r2_ci_multi(X, Y, groups, seed, n_boot=n_boot)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(Y))
    Y_shuffled = Y[perm]
    r2_shuffled_mean, r2_shuffled_std = held_out_r2_multi(X, Y_shuffled, groups, seed)

    result = {}
    for i, pname in enumerate(param_order):
        result[pname] = {
            "probe_r2": float(r2_mean[i]),
            "probe_r2_std": float(r2_std[i]),
            "probe_r2_ci_low": float(ci_lo[i]),
            "probe_r2_ci_high": float(ci_hi[i]),
            "probe_r2_ci95": [float(ci_lo[i]), float(ci_hi[i])],
            "probe_r2_ci_method": "source-level bootstrap (resample src_id with replacement, OOB test)",
            "probe_r2_shuffled": float(r2_shuffled_mean[i]),
            "probe_r2_shuffled_std": float(r2_shuffled_std[i]),
        }
    return result


def classification_metrics(X, y, groups, seed, n_splits=5, test_size=0.3):
    """held-out accuracy / chance-normalized accuracy / NMI (src_id 기준 GroupShuffleSplit). 2차와 동일."""
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    accs, nmis = [], []
    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes if n_classes > 0 else None

    for train_idx, test_idx in gss.split(X, y, groups):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        nmis.append(normalized_mutual_info_score(y[test_idx], pred))

    acc_mean = float(np.mean(accs))
    acc_chance_norm = float((acc_mean - chance) / (1 - chance)) if chance is not None and chance < 1 else None
    return {
        "accuracy": acc_mean,
        "accuracy_std": float(np.std(accs)),
        "acc_chance_normalized": acc_chance_norm,
        "nmi": float(np.mean(nmis)),
        "nmi_std": float(np.std(nmis)),
        "chance_level": chance,
        "n_classes": n_classes,
    }


def instrument_family_control(d, seed):
    dry_mask = d["effect"] == "dry"
    X = d["embeddings"][dry_mask]
    families = d["instrument_family"][dry_mask]
    groups = d["src_id"][dry_mask]

    unique = np.unique(families)
    full = classification_metrics(X, families, groups, seed)
    full["families_used"] = sorted(unique.tolist())
    full["n_sources_used"] = int(len(families))

    rng = np.random.RandomState(seed)
    if len(unique) >= 7:
        chosen = rng.choice(unique, size=7, replace=False)
        sub_mask = np.isin(families, chosen)
        subsampled = classification_metrics(X[sub_mask], families[sub_mask], groups[sub_mask], seed)
        subsampled["families_used"] = sorted(chosen.tolist())
        subsampled["n_sources_used"] = int(sub_mask.sum())
    else:
        subsampled = {
            "note": f"사용 가능한 패밀리가 {len(unique)}개뿐이라(<7) 7종 서브샘플링 불가",
            "families_used": sorted(unique.tolist()),
        }
    return full, subsampled


def plot_param_profile(probe_results, param_order_by_effect, negative_control_params, out_path):
    keys, r2s, ci_lo, ci_hi, colors = [], [], [], [], []
    for e in EFFECTS:
        for p in param_order_by_effect[e]:
            key = f"{e}.{p}"
            r = probe_results[e][p]
            keys.append(key)
            r2s.append(max(r["probe_r2"], 0.0))
            ci_lo.append(max(r["probe_r2_ci_low"], 0.0))
            ci_hi.append(r["probe_r2_ci_high"])
            colors.append(NEGATIVE_CONTROL_COLOR if key in negative_control_params else COLORS[e])

    yerr = np.array([[r - lo, hi - r] for r, lo, hi in zip(r2s, ci_lo, ci_hi)]).T
    yerr = np.clip(yerr, 0, None)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    x = np.arange(len(keys))
    ax.bar(x, r2s, yerr=yerr, capsize=3, color=colors, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Held-out R² (부트스트랩 95% CI, 0 미만은 0으로 표시)")
    ax.set_title("파라미터별 인코딩 강도 — 다변량 프로브")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_width_control(probe_results, jacobian_norms, out_path):
    """★ 최우선 확인 — width(모노라 원리적으로 못 읽어야 함) vs reverb의 다른 파라미터들."""
    reverb_params = ["wet_level", "room_size", "damping", "width", "freeze_mode"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    r2s = [max(probe_results["reverb"][p]["probe_r2"], 0.0) for p in reverb_params]
    shuffled = [max(probe_results["reverb"][p]["probe_r2_shuffled"], 0.0) for p in reverb_params]
    colors = [NEGATIVE_CONTROL_COLOR if p == "width" else COLORS["reverb"] for p in reverb_params]

    x = np.arange(len(reverb_params))
    width_bar = 0.35
    ax1.bar(x - width_bar / 2, r2s, width_bar, color=colors, label="실제 레이블", zorder=3)
    ax1.bar(x + width_bar / 2, shuffled, width_bar, color=COLORS["baseline"], label="셔플 통제", zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(reverb_params, rotation=30, ha="right")
    ax1.set_ylabel("Held-out R²")
    ax1.set_title("① 다변량 프로브 R² — width vs 나머지 reverb 파라미터")
    ax1.legend(frameon=False, fontsize=8)
    style_axis(ax1)

    if jacobian_norms is not None:
        norms = [jacobian_norms.get(f"reverb.{p}", 0.0) for p in reverb_params]
        colors2 = [NEGATIVE_CONTROL_COLOR if p == "width" else COLORS["reverb"] for p in reverb_params]
        ax2.bar(np.arange(len(reverb_params)), norms, color=colors2, zorder=3)
        ax2.set_xticks(np.arange(len(reverb_params)))
        ax2.set_xticklabels(reverb_params, rotation=30, ha="right")
        ax2.set_ylabel("야코비안 열 노름 평균")
        ax2.set_title("② 대리모델 야코비안 크기 — width vs 나머지")
        style_axis(ax2)
    else:
        ax2.text(0.5, 0.5, "03_jacobian.py를 먼저 실행하면\n야코비안 비교도 여기 표시됩니다", ha="center", va="center")
        ax2.axis("off")

    fig.suptitle("width 음성 통제 ★최우선 확인 — 유의하게 0을 초과하면 파이프라인 누수")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="다변량 Ridge 프로브 + width 음성 통제 + 악기 패밀리 통제 (3차)")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path}가 없습니다. 먼저 01_embed.py를 실행하세요.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_embeddings(emb_path)
    config = load_config(emb_path.parent)
    theta_slots = {e: tuple(v) for e, v in config["theta_slots"].items()}
    param_order_by_effect = config["param_order"]
    negative_control_params = config["negative_control_params"]

    print("이펙트별 다변량 프로브 계산 중 (부트스트랩 CI 포함, 시간이 걸릴 수 있음)...")
    probe_results = {
        e: multivariate_probe_for_effect(e, d, theta_slots, param_order_by_effect[e], args.seed, args.n_boot) for e in EFFECTS
    }

    print("악기 패밀리 통제 계산 중...")
    family_full, family_subsampled = instrument_family_control(d, args.seed)

    results_path = out_dir / "results.json"
    results_json = {}
    if results_path.exists():
        with open(results_path) as f:
            results_json = json.load(f)

    jacobian_norms = None
    if "params" in results_json:
        jacobian_norms = {k: v.get("jacobian_norm_mean") for k, v in results_json["params"].items()}

    print("그림 저장 중...")
    plot_param_profile(probe_results, param_order_by_effect, negative_control_params, out_dir / "param_profile.png")
    plot_width_control(probe_results, jacobian_norms, out_dir / "width_control.png")

    params = results_json.setdefault("params", {})
    for e in EFFECTS:
        for p in param_order_by_effect[e]:
            key = f"{e}.{p}"
            entry = params.setdefault(key, {})
            entry["effect"] = e
            entry["name"] = p
            entry["range"] = config["param_space"][e][p]
            entry["is_negative_control"] = key in negative_control_params
            entry.update(probe_results[e][p])

    results_json["controls"] = results_json.get("controls", {})
    results_json["controls"]["instrument_family"] = family_full
    results_json["controls"]["instrument_family_7class_subsampled"] = family_subsampled

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    width_r2 = probe_results["reverb"]["width"]["probe_r2"]
    print(f"완료: {results_path}, {out_dir}/param_profile.png, {out_dir}/width_control.png")
    print(f"★ width 음성 통제 probe_r2 = {width_r2:.4f} (0에 가까워야 함 — width_control.png 확인)")


if __name__ == "__main__":
    main()
