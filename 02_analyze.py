"""CLAP FX Probe — 02_analyze.py

01_embed.py가 만든 임베딩으로 (a) 선형/분류 프로브, (b) 차이 벡터 방향/크기,
(c) 단조성, 그리고 통제(레이블 셔플, 무작위 벡터, 악기 패밀리 분류)를 계산한다.

1차 실험에서 발견된 방법론적 결함을 수정한 개정판:
  1) 악기 통제를 개별 악기(표본 부족) 대신 NSynth 패밀리 11종으로 바꾸고,
     R²(회귀)와 accuracy(분류)가 단위가 달라 비교 불가능했던 문제를 NMI로 해결.
  2) distortion > reverb > highshelf 순서가 스윕 강도 차이 때문일 수 있다는 문제는,
     오디오 도메인 거리(D_audio)로 강도를 "기계적으로" 맞추는 대신 각 이펙트의 스윕
     범위 자체를 실무에서 흔히 쓰는 세기로 다시 잡는 것으로 해결한다 (01_embed.py의
     EFFECT_SPECS 참고 — reverb room_size 0→0.5, distortion drive_db 0→15,
     highshelf gain_db ±9dB). "단위 음향 변화당 인코딩 효율"이라는 기계적 지표는
     실무적 타당성이 없어 뺐다.
  3) highshelf처럼 파라미터가 부호를 가지면(-9~+9) 부스트(+)와 컷(-)이 반대
     방향이라 전역 평균 방향 벡터가 상쇄돼 무의미해지는 문제를 부호별 분리로 수정.

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
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit

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

# 크기가 사실상 0인 차이 벡터는 방향이 정의되지 않으므로 정규화에서 제외한다.
MIN_DIFF_NORM = 1e-6


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
        "instrument_family": data["instrument_family"],
        "pitch": data["pitch"],
        "effect": data["effect"],
        "level_idx": data["level_idx"],
        "param_value": data["param_value"],
    }


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """행 단위 코사인 유사도."""
    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return (a_n * b_n).sum(axis=1)


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


def bootstrap_r2_ci(X, y, groups, seed, n_boot=1000, ci=0.95):
    """소스 단위 부트스트랩(복원추출) + out-of-bag 평가로 held-out R²의 신뢰구간을 낸다.

    src_id를 복원추출로 재표집해 학습셋을 만들고, 한 번도 뽑히지 않은 소스를
    검증셋(OOB)으로 쓴다. 그룹 구조(같은 소스의 dry/wet을 같이 움직이는 것)를
    유지한 채로 하는 부트스트랩이라 GroupShuffleSplit의 취지와 일치한다.
    """
    unique_srcs = np.unique(groups)
    n = len(unique_srcs)
    src_to_rows = {s: np.where(groups == s)[0] for s in unique_srcs}
    rng = np.random.RandomState(seed)

    scores = []
    for _ in range(n_boot):
        boot_srcs = rng.choice(unique_srcs, size=n, replace=True)
        oob_srcs = np.setdiff1d(unique_srcs, boot_srcs)
        if len(oob_srcs) < 3:
            continue
        train_idx = np.concatenate([src_to_rows[s] for s in boot_srcs])
        test_idx = np.concatenate([src_to_rows[s] for s in oob_srcs])
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(r2_score(y[test_idx], pred))

    scores = np.array(scores)
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "mean": float(scores.mean()),
        "ci_low": float(np.percentile(scores, lo_pct)),
        "ci_high": float(np.percentile(scores, hi_pct)),
        "n_boot_used": int(len(scores)),
        "method": "source-level bootstrap (resample src_id with replacement, OOB test)",
    }


def classification_metrics(X, y, groups, seed, n_splits=5, test_size=0.3):
    """held-out accuracy / chance-normalized accuracy / NMI (src_id 기준 GroupShuffleSplit).

    클래스 수가 다른 두 분류 문제(예: 악기 패밀리 11종 vs 이펙트 레벨 7종)를 비교하려면
    accuracy 하나만으론 안 된다 — 우연 수준 자체가 다르기 때문이다. NMI는 클래스 수와
    무관하게 비교 가능해 주 지표로 쓴다.
    """
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


def mean_pairwise_cosine(vectors: np.ndarray):
    """벡터 집합의 평균 쌍별 코사인 유사도 (대각선 제외 상삼각 평균)."""
    if len(vectors) < 2:
        return None, None
    sim = cosine_similarity(vectors)
    iu = np.triu_indices_from(sim, k=1)
    pair_sims = sim[iu]
    return float(pair_sims.mean()), float(pair_sims.std())


def _direction_and_monotonicity(diffs: np.ndarray, params_abs: np.ndarray):
    """diff 벡터 집합에서 (정규화 후) 방향 일관성과 |param| 기준 단조성을 계산.

    호출부에서 이미 원하는 레벨/부호 그룹으로 필터링된 diffs/params_abs를 넘겨받는다.
    signed 이펙트의 부호 그룹별 계산과 unsigned 이펙트의 전체-레벨 계산이 이 함수 하나를
    공유한다 — 부호가 섞인 채로 평균 내면 방향이 상쇄된다는 문제(문제 3)를 부호 그룹
    분리로 해결하되, 계산 자체는 그룹 내에서 동일한 로직을 쓴다.
    """
    norms = np.linalg.norm(diffs, axis=1)
    valid = norms >= MIN_DIFF_NORM
    unit_diffs = diffs[valid] / norms[valid, None]
    cos_mean, cos_std = mean_pairwise_cosine(unit_diffs)

    mean_direction = diffs.mean(axis=0)
    norm = np.linalg.norm(mean_direction)
    mean_direction_unit = mean_direction / norm if norm > 0 else mean_direction
    projections = diffs @ mean_direction_unit
    rho, pvalue = spearmanr(params_abs, projections)

    return {
        "direction_cosine_mean": cos_mean,
        "direction_cosine_std": cos_std,
        "n_valid_direction": int(valid.sum()),
        "n_dropped_near_zero_norm": int((~valid).sum()),
        "monotonicity_spearman_rho_abs_param": float(rho),
        "monotonicity_spearman_pvalue_abs_param": float(pvalue),
        "n_rows": int(len(diffs)),
    }, mean_direction_unit, projections


def analyze_effect(effect_name, d, dry_by_src, seed, n_boot):
    mask = d["effect"] == effect_name
    X = d["embeddings"][mask]
    y = d["param_value"][mask]
    groups = d["src_id"][mask]
    levels = d["level_idx"][mask]

    # (a) Ridge 선형 프로브 — held-out R² + 부트스트랩 95% CI
    r2_mean, r2_std = held_out_r2(X, y, groups, seed)
    r2_ci = bootstrap_r2_ci(X, y, groups, seed, n_boot=n_boot)

    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y)
    r2_shuffled_mean, r2_shuffled_std = held_out_r2(X, y_shuffled, groups, seed)

    # (a-2) 7-way 분류 프로브 — 파라미터가 이미 이산 레벨(7단계)이므로 분류로도 돌린다.
    # R²(회귀)와 악기 통제의 accuracy(분류)는 단위가 달라 직접 비교가 안 됐던 문제를
    # 여기서부터 해결한다 — 이 분류 프로브와 악기 패밀리 분류를 NMI로 나란히 비교한다.
    clf = classification_metrics(X, levels, groups, seed)

    # 레벨별 diff 벡터 (항상 계산 — 근접-0 노름 필터는 여기서 처리)
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

    # (b)② 크기 — |param| 기준 (부호와 무관하게 항상 pooled: 노름은 부호가 없다)
    all_norms = np.concatenate([norms_by_level[lvl] for lvl in range(n_levels)])
    all_diffs = np.concatenate([diffs_by_level[lvl] for lvl in range(n_levels)], axis=0)
    all_params = np.concatenate([y[levels == lvl] for lvl in range(n_levels)])
    magnitude_rho, magnitude_pvalue = spearmanr(all_params, all_norms)
    magnitude_rho_abs, magnitude_pvalue_abs = spearmanr(np.abs(all_params), all_norms)

    # 추가 점검 (A) — 무효과 레벨(param==0)이 진짜 dry와 같은가.
    # reverb는 room_size=0이어도 wet_level=0.4가 항상 섞이므로 완전한 dry가 아닐 수 있다.
    zero_mask = mask & np.isclose(d["param_value"], 0.0)
    if zero_mask.sum() > 0:
        zero_srcs = d["src_id"][zero_mask]
        zero_embs = d["embeddings"][zero_mask]
        zero_dry = np.stack([dry_by_src[s] for s in zero_srcs])
        neutral_level_cos_check = float(np.mean(cosine_rows(zero_embs, zero_dry)))
    else:
        neutral_level_cos_check = None

    # 문제 3 수정 — 파라미터 범위가 0을 걸치면(signed) 부스트(+)/컷(-)이 반대 방향이라
    # 전역 평균 방향 벡터가 상쇄된다. 자동 판정 후 부호별로 따로 계산한다.
    is_signed = bool(float(all_params.min()) < 0 < float(all_params.max()))

    direction_positive = direction_negative = None
    cos_pos_neg = None
    n_neutral_excluded = 0
    direction_cosine_mean = None
    monotonicity_spearman_rho = monotonicity_spearman_pvalue = None
    monotonicity_spearman_rho_abs_param = monotonicity_spearman_pvalue_abs_param = None
    plot_projections = plot_params = None

    if is_signed:
        neutral = np.isclose(all_params, 0.0)
        pos = (all_params > 0) & ~neutral
        neg = (all_params < 0) & ~neutral
        n_neutral_excluded = int(neutral.sum())

        direction_positive, v_pos_unit, _ = _direction_and_monotonicity(all_diffs[pos], np.abs(all_params[pos]))
        direction_negative, v_neg_unit, _ = _direction_and_monotonicity(all_diffs[neg], np.abs(all_params[neg]))
        cos_pos_neg = float(np.dot(v_pos_unit, v_neg_unit))
        # 전역 pooled 방향/단조성은 부호가 섞여 의미가 없으므로 signed 이펙트는 None으로 둔다
        # (부호 그룹별 값은 direction_positive/direction_negative에 있다).
    else:
        valid_cos = [v["cosine_mean"] for v in direction_cos_by_level.values() if v["cosine_mean"] is not None]
        direction_cosine_mean = float(np.mean(valid_cos)) if valid_cos else None

        mono_result, _mean_dir_unit, projections = _direction_and_monotonicity(all_diffs, np.abs(all_params))
        monotonicity_spearman_rho_abs_param = mono_result["monotonicity_spearman_rho_abs_param"]
        monotonicity_spearman_pvalue_abs_param = mono_result["monotonicity_spearman_pvalue_abs_param"]
        # unsigned는 signed 값과 abs(param) 값이 사실상 대응되므로 signed rho도 함께 보고
        rho, pvalue = spearmanr(all_params, projections)
        monotonicity_spearman_rho = float(rho)
        monotonicity_spearman_pvalue = float(pvalue)
        plot_projections = projections
        plot_params = all_params

    return {
        "is_signed": is_signed,
        "probe_r2": r2_mean,
        "probe_r2_std": r2_std,
        "probe_r2_ci_low": r2_ci["ci_low"],
        "probe_r2_ci_high": r2_ci["ci_high"],
        "probe_r2_ci_method": r2_ci["method"],
        "probe_r2_shuffled": r2_shuffled_mean,
        "probe_r2_shuffled_std": r2_shuffled_std,
        "probe_accuracy_7way": clf["accuracy"],
        "probe_accuracy_7way_std": clf["accuracy_std"],
        "probe_acc_chance_normalized": clf["acc_chance_normalized"],
        "probe_nmi": clf["nmi"],
        "probe_nmi_std": clf["nmi_std"],
        "probe_chance_level_7way": clf["chance_level"],
        "direction_cosine_mean": direction_cosine_mean,
        "direction_cosine_by_level": {str(k): v for k, v in direction_cos_by_level.items()},
        "direction_positive": direction_positive,
        "direction_negative": direction_negative,
        "cos_pos_neg": cos_pos_neg,
        "n_neutral_excluded_rows": n_neutral_excluded,
        "neutral_level_cos_check": neutral_level_cos_check,
        "magnitude_spearman_rho": float(magnitude_rho),
        "magnitude_spearman_pvalue": float(magnitude_pvalue),
        "magnitude_spearman_rho_abs_param": float(magnitude_rho_abs),
        "magnitude_spearman_pvalue_abs_param": float(magnitude_pvalue_abs),
        "monotonicity_spearman_rho": monotonicity_spearman_rho,
        "monotonicity_spearman_pvalue": monotonicity_spearman_pvalue,
        "monotonicity_spearman_rho_abs_param": monotonicity_spearman_rho_abs_param,
        "monotonicity_spearman_pvalue_abs_param": monotonicity_spearman_pvalue_abs_param,
        "_projections": plot_projections,
        "_param_values": plot_params,
    }


def random_vector_baseline(dim, n_vectors, seed):
    """512차원 무작위 벡터 쌍의 코사인 유사도 — 차이 벡터 일관성의 기준선."""
    rng = np.random.RandomState(seed)
    vectors = rng.normal(size=(n_vectors, dim))
    return mean_pairwise_cosine(vectors)


def instrument_family_control(d, seed):
    """악기 패밀리(11종) 분류 상한 — 개별 악기 대신 패밀리로 바꿔 표본 부족 문제를 해결.

    1차 실험은 개별 악기 47클래스/294샘플 ≈ 클래스당 6개로 표본이 너무 적었다.
    패밀리 단위(11종)로 바꾸면 300소스 기준 클래스당 약 27개로 안정적이다.
    7종으로 무작위 서브샘플링한 버전도 함께 내 이펙트 프로브(7-way)와 클래스 수를
    완전히 맞춘 직접 비교를 제공한다.
    """
    dry_mask = d["effect"] == "dry"
    X = d["embeddings"][dry_mask]
    families = d["instrument_family"][dry_mask]
    groups = d["src_id"][dry_mask]  # dry는 소스당 1행이라 그룹 크기가 항상 1

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


def plot_probe_r2(results, family_ctrl, out_path):
    """① R²(회귀, 셔플 통제 포함) ② NMI(분류, 이펙트 vs 악기 패밀리 — 단위 통일 비교)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
    x = np.arange(len(EFFECTS))
    width = 0.35

    real_r2 = [max(results[e]["probe_r2"], 0.0) for e in EFFECTS]
    shuffled_r2 = [max(results[e]["probe_r2_shuffled"], 0.0) for e in EFFECTS]
    ci_low = [max(results[e]["probe_r2_ci_low"], 0.0) for e in EFFECTS]
    ci_high = [results[e]["probe_r2_ci_high"] for e in EFFECTS]
    yerr = np.array([[r - lo, hi - r] for r, lo, hi in zip(real_r2, ci_low, ci_high)]).T
    yerr = np.clip(yerr, 0, None)

    ax1.bar(
        x - width / 2, real_r2, width, yerr=yerr, capsize=3,
        label="실제 라벨 (95% CI)", color=[COLORS[e] for e in EFFECTS], zorder=3,
    )
    ax1.bar(x + width / 2, shuffled_r2, width, label="레이블 셔플 (통제)", color=COLORS["baseline"], zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(EFFECTS)
    ax1.set_ylabel("Held-out R² (0 미만은 0으로 표시)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("① 선형 프로브 R² (부트스트랩 95% CI)")
    ax1.legend(frameon=False, fontsize=8)
    style_axis(ax1)

    # R²(회귀)와 accuracy(분류)는 단위가 달라 직접 비교할 수 없었다 — NMI는 클래스 수와
    # 무관하게 비교 가능하므로 이펙트 7-way 분류와 악기 패밀리 분류를 여기서 나란히 본다.
    labels2 = EFFECTS + ["악기 패밀리\n(11종)", "악기 패밀리\n(7종 서브샘플)"]
    nmi_vals = [results[e]["probe_nmi"] for e in EFFECTS] + [family_ctrl["full"]["nmi"], family_ctrl["subsampled"].get("nmi", 0.0)]
    colors2 = [COLORS[e] for e in EFFECTS] + [COLORS["baseline"], COLORS["baseline"]]
    x2 = np.arange(len(labels2))
    ax2.bar(x2, nmi_vals, color=colors2, zorder=3)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=8)
    ax2.set_ylabel("Normalized Mutual Information")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("② 분류 NMI — 이펙트 vs 악기 패밀리 (단위 통일 비교)")
    style_axis(ax2)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_direction_cos(results, random_baseline, out_path):
    """① 방향(정규화 후 코사인) — signed 이펙트는 (+)/(-) 그룹으로 나눠 표시. ② 크기-파라미터 상관."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    labels, means, colors = [], [], []
    for e in EFFECTS:
        r = results[e]
        if r["is_signed"]:
            labels += [f"{e}(+)", f"{e}(-)"]
            means += [r["direction_positive"]["direction_cosine_mean"] or 0.0, r["direction_negative"]["direction_cosine_mean"] or 0.0]
            colors += [COLORS[e], COLORS[e]]
        else:
            labels.append(e)
            means.append(r["direction_cosine_mean"] or 0.0)
            colors.append(COLORS[e])
    labels.append("무작위 벡터 (통제)")
    means.append(random_baseline[0] or 0.0)
    colors.append(COLORS["baseline"])

    x1 = np.arange(len(labels))
    ax1.bar(x1, means, color=colors, zorder=3)
    ax1.axhline(0, color=GRID_COLOR, linewidth=1)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("평균 쌍별 코사인 유사도 (정규화된 차이 벡터)")
    ax1.set_title("① 방향 일관성 (signed 이펙트는 부호별로 분리)")
    style_axis(ax1)

    x2 = np.arange(len(EFFECTS))
    magnitude_rhos = [results[e]["magnitude_spearman_rho"] for e in EFFECTS]
    ax2.bar(x2, magnitude_rhos, color=[COLORS[e] for e in EFFECTS], zorder=3)
    for xi, e in zip(x2, EFFECTS):
        rho_abs = results[e]["magnitude_spearman_rho_abs_param"]
        yv = magnitude_rhos[EFFECTS.index(e)]
        ax2.text(xi, yv + (0.05 if yv >= 0 else -0.05), f"|p| ρ={rho_abs:.2f}", ha="center",
                  va="bottom" if yv >= 0 else "top", fontsize=8, color=INK_SECONDARY)
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
    """unsigned 이펙트만 그린다 — signed는 부호가 섞이면 무의미해 signed_direction.png로 분리."""
    unsigned = [e for e in EFFECTS if not results[e]["is_signed"]]
    if not unsigned:
        print("unsigned 이펙트가 없어 monotonicity.png를 생성하지 않습니다.")
        return

    fig, axes = plt.subplots(1, len(unsigned), figsize=(4.3 * len(unsigned), 4.2), dpi=150, sharey=False)
    if len(unsigned) == 1:
        axes = [axes]
    for ax, effect_name in zip(axes, unsigned):
        r = results[effect_name]
        ax.scatter(r["_param_values"], r["_projections"], s=14, alpha=0.5, color=COLORS[effect_name], edgecolors="none")
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


def plot_signed_direction(results, d, dry_by_src, out_path):
    """signed 이펙트(예: highshelf)의 부호별 방향 분석 — (+)/(-) 각각 |param| vs 투영값."""
    signed_effects = [e for e in EFFECTS if results[e]["is_signed"]]
    if not signed_effects:
        print("signed 이펙트가 없어 signed_direction.png를 생성하지 않습니다.")
        return

    fig, axes = plt.subplots(1, len(signed_effects), figsize=(6 * len(signed_effects), 4.5), dpi=150)
    if len(signed_effects) == 1:
        axes = [axes]

    for ax, effect_name in zip(axes, signed_effects):
        mask = d["effect"] == effect_name
        y = d["param_value"][mask]
        levels = d["level_idx"][mask]
        srcs = d["src_id"][mask]
        embs = d["embeddings"][mask]
        dry = np.stack([dry_by_src[s] for s in srcs])
        diffs = embs - dry

        n_levels = int(levels.max()) + 1
        all_params = np.concatenate([y[levels == lvl] for lvl in range(n_levels)])
        all_diffs = np.concatenate([diffs[levels == lvl] for lvl in range(n_levels)], axis=0)

        neutral = np.isclose(all_params, 0.0)
        pos = (all_params > 0) & ~neutral
        neg = (all_params < 0) & ~neutral

        for sub_mask, sign_label, color in [(pos, "부스트 (+)", COLORS[effect_name]), (neg, "컷 (-)", INK_SECONDARY)]:
            sub_diffs = all_diffs[sub_mask]
            sub_params_abs = np.abs(all_params[sub_mask])
            norm = np.linalg.norm(sub_diffs.mean(axis=0))
            direction_unit = sub_diffs.mean(axis=0) / norm if norm > 0 else sub_diffs.mean(axis=0)
            proj = sub_diffs @ direction_unit
            ax.scatter(sub_params_abs, proj, s=14, alpha=0.5, color=color, edgecolors="none", label=sign_label)

        r = results[effect_name]
        ax.set_title(f"{effect_name}\ncos(v_+, v_-) = {r['cos_pos_neg']:.2f}")
        ax.set_xlabel("|파라미터 값|")
        ax.set_ylabel("자기 부호 그룹 방향 벡터로의 투영값")
        ax.legend(frameon=False, fontsize=8)
        style_axis(ax)

    fig.suptitle("부호 있는 파라미터의 방향 분석 (부스트/컷을 분리)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CLAP 임베딩의 이펙트 정보 함유량 분석")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000, help="Ridge R² 부트스트랩 CI 반복 횟수")
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

    print("이펙트별 프로브/방향/크기/단조성 계산 중 (부트스트랩 CI 포함, 시간이 걸릴 수 있음)...")
    results = {e: analyze_effect(e, d, dry_by_src, args.seed, args.n_boot) for e in EFFECTS}

    print("통제 계산 중 (무작위 벡터, 악기 패밀리 분류)...")
    dim = d["embeddings"].shape[1]
    n_sources = len(set(d["src_id"].tolist()))
    random_baseline = random_vector_baseline(dim, n_sources, args.seed)
    family_full, family_subsampled = instrument_family_control(d, args.seed)

    print("그림 저장 중...")
    plot_probe_r2(results, {"full": family_full, "subsampled": family_subsampled}, out_dir / "probe_r2.png")
    plot_direction_cos(results, random_baseline, out_dir / "direction_cos.png")
    plot_monotonicity(results, out_dir / "monotonicity.png")
    plot_signed_direction(results, d, dry_by_src, out_dir / "signed_direction.png")

    results_json = {
        "effects": {
            e: {k: v for k, v in r.items() if not k.startswith("_")} for e, r in results.items()
        },
        "controls": {
            "random_vector_cosine_mean": random_baseline[0],
            "random_vector_cosine_std": random_baseline[1],
            "instrument_family": family_full,
            "instrument_family_7class_subsampled": family_subsampled,
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"완료: {out_dir / 'results.json'}, {out_dir}/*.png")


if __name__ == "__main__":
    main()
