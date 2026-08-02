"""CLAP FX Probe — 02_analyze.py

01_embed.py가 만든 임베딩으로 세 가지 측정(선형 프로브, 차이 벡터 방향/크기, 단조성)과
세 가지 통제(레이블 셔플, 무작위 벡터, 악기 분류 상한)를 계산한다.

(d) 사상 모델 학습과 H0~H6 위계 판정은 03_mapping.py에서 이어서 처리한다
(이 스크립트가 만든 results.json에 이어 붙임).

이 스크립트는 수치만 산출한다. "정보가 있다/없다"의 해석 기준은 README를 참고할 것 —
코드가 결론을 내리지 않는다.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# 한글 라벨이 깨지지 않도록 시스템에 있는 한글 지원 폰트를 우선 사용한다.
_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score

EFFECTS = ["reverb", "distortion", "highshelf"]

# dataviz 스킬 참조 팔레트의 categorical slot 1/2/3 (blue/orange/aqua) + muted gray.
# 세 이펙트에 고정 순서로 배정하고, 통제/기준선은 무채색으로 구분한다.
COLORS = {
    "reverb": "#2a78d6",
    "distortion": "#eb6834",
    "highshelf": "#1baf7a",
    "baseline": "#898781",
}
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


def load_embeddings(path: Path):
    data = np.load(path, allow_pickle=False)
    return {
        "embeddings": data["embeddings"],
        "src_id": data["src_id"],
        "instrument": data["instrument"],
        "pitch": data["pitch"],
        "effect": data["effect"],
        "level_idx": data["level_idx"],
        "param_value": data["param_value"],
    }


def held_out_r2(X, y, groups, seed, n_splits=5, test_size=0.3):
    """GroupShuffleSplit으로 src_id 단위 분리 후 Ridge의 held-out R²를 반환.

    같은 소스의 dry/wet이 train/test에 걸치면 모델이 이펙트가 아니라 소스를
    외우게 되어 R²가 허위로 높아진다 — 그래서 무작위 split이 아니라 그룹 단위로 나눈다.
    """
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    scores = []
    for train_idx, test_idx in gss.split(X, y, groups):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(r2_score(y[test_idx], pred))
    return float(np.mean(scores)), float(np.std(scores))


def mean_pairwise_cosine(vectors: np.ndarray):
    """벡터 집합의 평균 쌍별 코사인 유사도 (대각선 제외 상삼각 평균)."""
    if len(vectors) < 2:
        return None, None
    sim = cosine_similarity(vectors)
    iu = np.triu_indices_from(sim, k=1)
    pair_sims = sim[iu]
    return float(pair_sims.mean()), float(pair_sims.std())


# 크기가 사실상 0인 차이 벡터는 방향이 정의되지 않으므로 정규화에서 제외한다.
MIN_DIFF_NORM = 1e-6


def analyze_effect(effect_name, d, dry_by_src, seed):
    mask = d["effect"] == effect_name
    X = d["embeddings"][mask]
    y = d["param_value"][mask]
    groups = d["src_id"][mask]
    levels = d["level_idx"][mask]

    # (a) Ridge 선형 프로브 — held-out R²
    r2_mean, r2_std = held_out_r2(X, y, groups, seed)

    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y)
    r2_shuffled_mean, r2_shuffled_std = held_out_r2(X, y_shuffled, groups, seed)

    # (b) 차이 벡터 — 방향과 크기 분리
    # 정규화 없이 코사인을 재면 소스마다 다른 벡터 크기가 방향 불일치로 오인되어
    # "방향은 고정, 크기만 파라미터 의존"(H2)인 경우를 "정보 없음"(H0)으로 잘못
    # 판정하게 된다. 그래서 ① 방향(정규화 후 코사인)과 ② 크기(‖v‖-파라미터 상관)를
    # 반드시 나눠서 본다.
    n_levels = int(levels.max()) + 1
    diffs_by_level = {}
    direction_cos_by_level = {}
    norms_by_level = {}
    for lvl in range(n_levels):
        lvl_mask = mask & (d["level_idx"] == lvl)
        srcs = d["src_id"][lvl_mask]
        embs = d["embeddings"][lvl_mask]
        diffs = np.stack([embs[i] - dry_by_src[srcs[i]] for i in range(len(srcs))])
        diffs_by_level[lvl] = diffs

        norms = np.linalg.norm(diffs, axis=1)
        norms_by_level[lvl] = norms

        valid = norms >= MIN_DIFF_NORM
        unit_diffs = diffs[valid] / norms[valid, None]
        cos_mean, cos_std = mean_pairwise_cosine(unit_diffs)
        direction_cos_by_level[lvl] = {
            "cosine_mean": cos_mean,
            "cosine_std": cos_std,
            "n_sources": int(valid.sum()),
            "n_dropped_near_zero_norm": int((~valid).sum()),
        }

    valid_cos = [v["cosine_mean"] for v in direction_cos_by_level.values() if v["cosine_mean"] is not None]
    direction_cosine_mean = float(np.mean(valid_cos)) if valid_cos else None

    all_norms = np.concatenate([norms_by_level[lvl] for lvl in range(n_levels)])
    param_values_for_norms = np.concatenate([y[levels == lvl] for lvl in range(n_levels)])
    magnitude_rho, magnitude_pvalue = spearmanr(param_values_for_norms, all_norms)
    magnitude_rho_abs, magnitude_pvalue_abs = spearmanr(np.abs(param_values_for_norms), all_norms)

    # (c) 단조성 — 평균 방향 벡터에 투영 후 파라미터 값과 Spearman
    all_diffs = np.concatenate([diffs_by_level[lvl] for lvl in range(n_levels)], axis=0)
    mean_direction = all_diffs.mean(axis=0)
    norm = np.linalg.norm(mean_direction)
    mean_direction_unit = mean_direction / norm if norm > 0 else mean_direction

    projections = all_diffs @ mean_direction_unit
    param_values_ordered = np.concatenate(
        [y[levels == lvl] for lvl in range(n_levels)]
    )
    rho, pvalue = spearmanr(param_values_ordered, projections)
    # highshelf처럼 파라미터 범위가 0을 사이에 두고 대칭이면(-15~+15), 방향 벡터
    # 투영값은 dry(0dB) 근방에서 최소인 V자 형태가 되어 signed rho가 대칭성 때문에
    # 낮게(혹은 음수로) 나올 수 있다. |파라미터|와의 상관도 함께 보고해 "세기" 해석을 보완한다.
    rho_abs, pvalue_abs = spearmanr(np.abs(param_values_ordered), projections)

    return {
        "probe_r2": r2_mean,
        "probe_r2_std": r2_std,
        "probe_r2_shuffled": r2_shuffled_mean,
        "probe_r2_shuffled_std": r2_shuffled_std,
        "direction_cosine_mean": direction_cosine_mean,
        "direction_cosine_by_level": {str(k): v for k, v in direction_cos_by_level.items()},
        "magnitude_spearman_rho": float(magnitude_rho),
        "magnitude_spearman_pvalue": float(magnitude_pvalue),
        "magnitude_spearman_rho_abs_param": float(magnitude_rho_abs),
        "magnitude_spearman_pvalue_abs_param": float(magnitude_pvalue_abs),
        "monotonicity_spearman_rho": float(rho),
        "monotonicity_spearman_pvalue": float(pvalue),
        "monotonicity_spearman_rho_abs_param": float(rho_abs),
        "monotonicity_spearman_pvalue_abs_param": float(pvalue_abs),
        "_projections": projections,
        "_param_values": param_values_ordered,
    }


def random_vector_baseline(dim, n_vectors, seed):
    """512차원 무작위 벡터 쌍의 코사인 유사도 — 차이 벡터 일관성의 기준선."""
    rng = np.random.RandomState(seed)
    vectors = rng.normal(size=(n_vectors, dim))
    return mean_pairwise_cosine(vectors)


def instrument_classification_upper_bound(d, seed):
    """악기 분류 상한 — dry 임베딩만으로 악기 라벨을 얼마나 잘 맞히는지.

    논문은 953개 악기, 훨씬 큰 데이터로 90.4%를 보고했다. 여기서는 샘플링된
    소스 수가 훨씬 적어 절대 정확도를 논문과 직접 비교할 수 없다 — 목적은
    "이펙트 프로브보다 악기 식별이 훨씬 쉬운가"를 보는 상대적 상한선이다.
    """
    dry_mask = d["effect"] == "dry"
    X = d["embeddings"][dry_mask]
    labels = d["instrument"][dry_mask]

    unique, counts = np.unique(labels, return_counts=True)
    usable = set(unique[counts >= 2])
    keep = np.array([lbl in usable for lbl in labels])
    n_dropped = int((~keep).sum())
    X, labels = X[keep], labels[keep]

    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        return {
            "accuracy": None,
            "n_classes_used": int(len(unique)),
            "n_sources_dropped_singleton": n_dropped,
            "note": "클래스가 2개 미만이라 분류 상한을 계산할 수 없음 (n-sources를 늘릴 것)",
        }

    n_splits = min(5, int(counts.min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000)
    scores = cross_val_score(clf, X, labels, cv=skf, scoring="accuracy")

    return {
        "accuracy": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "n_classes_used": int(len(unique)),
        "n_sources_used": int(len(labels)),
        "n_sources_dropped_singleton": n_dropped,
        "paper_reference_accuracy": 0.904,
        "note": "샘플 수가 논문(953개 악기)보다 훨씬 적어 절대치는 참고용. "
        "이펙트 프로브 R²와의 상대적 격차가 해석의 핵심.",
    }


def plot_probe_r2(results, instrument_ctrl, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    x = np.arange(len(EFFECTS))
    width = 0.35

    real_r2 = [max(results[e]["probe_r2"], 0.0) for e in EFFECTS]
    shuffled_r2 = [max(results[e]["probe_r2_shuffled"], 0.0) for e in EFFECTS]

    ax.bar(x - width / 2, real_r2, width, label="실제 라벨", color=[COLORS[e] for e in EFFECTS], zorder=3)
    ax.bar(x + width / 2, shuffled_r2, width, label="레이블 셔플 (통제)", color=COLORS["baseline"], zorder=3)

    if instrument_ctrl.get("accuracy") is not None:
        ax.axhline(
            instrument_ctrl["accuracy"],
            color=INK_SECONDARY,
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )
        ax.text(
            len(EFFECTS) - 0.5,
            instrument_ctrl["accuracy"] + 0.02,
            f"악기 분류 상한 {instrument_ctrl['accuracy']:.2f}",
            color=INK_SECONDARY,
            fontsize=9,
            ha="right",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(EFFECTS)
    ax.set_ylabel("Held-out R² (0 미만은 0으로 표시) / 분류 정확도")
    ax.set_ylim(0, 1.05)
    ax.set_title("이펙트별 선형 프로브 R² vs 셔플 기준선 vs 악기 분류 상한")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_direction_cos(results, random_baseline, out_path):
    """① 방향(정규화 후 코사인)과 ② 크기(‖v‖-파라미터 Spearman ρ)를 나란히 그린다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    labels = EFFECTS + ["무작위 벡터 (통제)"]
    direction_means = [results[e]["direction_cosine_mean"] or 0.0 for e in EFFECTS] + [random_baseline[0] or 0.0]
    colors = [COLORS[e] for e in EFFECTS] + [COLORS["baseline"]]

    x1 = np.arange(len(labels))
    ax1.bar(x1, direction_means, color=colors, zorder=3)
    ax1.axhline(0, color=GRID_COLOR, linewidth=1)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("평균 쌍별 코사인 유사도 (정규화된 차이 벡터)")
    ax1.set_title("① 방향 일관성")
    style_axis(ax1)

    x2 = np.arange(len(EFFECTS))
    magnitude_rhos = [results[e]["magnitude_spearman_rho"] for e in EFFECTS]
    ax2.bar(x2, magnitude_rhos, color=[COLORS[e] for e in EFFECTS], zorder=3)
    # highshelf처럼 파라미터 범위가 0을 사이에 둔 대칭이면 signed rho가 왜곡될 수 있어
    # |파라미터| 기준 rho를 막대 위에 함께 표기한다 (모노토닉성 플롯과 동일한 보완).
    for xi, e in zip(x2, EFFECTS):
        rho_abs = results[e]["magnitude_spearman_rho_abs_param"]
        y = magnitude_rhos[EFFECTS.index(e)]
        ax2.text(xi, y + (0.05 if y >= 0 else -0.05), f"|p| ρ={rho_abs:.2f}", ha="center",
                 va="bottom" if y >= 0 else "top", fontsize=8, color=INK_SECONDARY)
    ax2.axhline(0, color=GRID_COLOR, linewidth=1)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(EFFECTS)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_ylabel("Spearman ρ (‖차이 벡터‖ vs 파라미터 값)")
    ax2.set_title("② 크기-파라미터 상관")
    style_axis(ax2)

    fig.suptitle("이펙트 방향 벡터의 소스 간 일관성 (방향/크기 분리)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_monotonicity(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150, sharey=False)
    for ax, effect_name in zip(axes, EFFECTS):
        r = results[effect_name]
        ax.scatter(
            r["_param_values"],
            r["_projections"],
            s=14,
            alpha=0.5,
            color=COLORS[effect_name],
            edgecolors="none",
        )
        ax.set_title(
            f"{effect_name}\n"
            f"Spearman ρ={r['monotonicity_spearman_rho']:.2f} "
            f"(|파라미터| 기준 ρ={r['monotonicity_spearman_rho_abs_param']:.2f})"
        )
        ax.set_xlabel("파라미터 값")
        style_axis(ax)
    axes[0].set_ylabel("평균 방향 벡터로의 투영값")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CLAP 임베딩의 이펙트 정보 함유량 분석")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path}가 없습니다. 먼저 01_embed.py를 실행하세요.")

    d = load_embeddings(emb_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dry_mask = d["effect"] == "dry"
    dry_by_src = dict(zip(d["src_id"][dry_mask], d["embeddings"][dry_mask]))

    print("이펙트별 프로브/일관성/단조성 계산 중...")
    results = {e: analyze_effect(e, d, dry_by_src, args.seed) for e in EFFECTS}

    print("통제 계산 중 (무작위 벡터, 악기 분류 상한)...")
    dim = d["embeddings"].shape[1]
    n_sources = len(set(d["src_id"].tolist()))
    random_baseline = random_vector_baseline(dim, n_sources, args.seed)
    instrument_ctrl = instrument_classification_upper_bound(d, args.seed)

    print("그림 저장 중...")
    plot_probe_r2(results, instrument_ctrl, out_dir / "probe_r2.png")
    plot_direction_cos(results, random_baseline, out_dir / "direction_cos.png")
    plot_monotonicity(results, out_dir / "monotonicity.png")

    results_json = {
        "effects": {
            e: {k: v for k, v in r.items() if not k.startswith("_")} for e, r in results.items()
        },
        "controls": {
            "random_vector_cosine_mean": random_baseline[0],
            "random_vector_cosine_std": random_baseline[1],
            "instrument_classification": instrument_ctrl,
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"완료: {out_dir / 'results.json'}, {out_dir}/*.png")


if __name__ == "__main__":
    main()
