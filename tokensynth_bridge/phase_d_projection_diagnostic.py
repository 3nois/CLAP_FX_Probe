"""9차 후속 D — TokenSynth가 이펙트 성분을 어디서 잃는지 특정한다.

9차에서 확인된 격차: 임베딩 단계 손잡이 예측 cos 0.71~0.86 vs 오디오 단계 방향
일치도 cos 0.03~0.06(F-4). 유력 용의자는 논문 III-A의 projection layer
(e_clap[512] -> 2-layer MLP -> ê[1024] -> transformer)다 — 악기 판별에 유용한
방향만 보존하고 이펙트 성분을 버렸다는 가설을 가중치 forward만으로 검증한다.

★ 재학습·재렌더링 없음. `out/caches/oat_emb_ts.npz`(TokenSynth 공간, 1,200소스)와
TokenSynth 체크포인트의 clap_projection 서브모듈 forward만 쓴다.

  D-1  변위 감쇠율 — 투영 전후 wet/dry 상대 변위·코사인 비교 (보조)
  D-2  ★핵심 — 같은 프로브를 투영 전/후에 걸어 이펙트 R²와 악기 NMI를 대조.
       "이펙트만 떨어지고 악기는 유지"가 확인되면 선택적 폐기가 특정된다.
  D-3  부분공간 재확인 — 대리모델 없이 평균 차이 벡터로 LDA 부분공간 투영 비율을
       투영 전/후 양쪽에서 냄(4차 z≈-4 재현 여부).

결과 해석은 이 스크립트가 단정하지 않는다. README 판정 기준표를 따를 것.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from tokensynth import TokenSynth

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False
INK_SECONDARY = "#52514e"; GRID_COLOR = "#e1e0d9"
COLORS = {"reverb": "#2a78d6", "distortion": "#eb6834", "highshelf": "#1baf7a", "null": "#e34948", "baseline": "#898781",
          "before": "#898781", "after": "#2a78d6"}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


EFFECT_NAMES = ["reverb", "distortion", "highshelf"]
LEVEL_RAW_RANGE = {"reverb": (0.0, 0.5), "distortion": (0.0, 15.0), "highshelf": (-9.0, 9.0)}
LEVEL_RAW_VALUES = {"reverb": [0.0, 0.25, 0.5], "distortion": [0.0, 7.5, 15.0], "highshelf": [-9.0, 0.0, 9.0]}


# ---------------------------------------------------------------------------
# projection layer 단독 forward
# ---------------------------------------------------------------------------
def load_projection(device):
    synth = TokenSynth.from_pretrained(aug=True, device=device)
    proj = synth.clap_projection
    proj.eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj, synth.hparams.embed_dim


def proj_forward(proj, X, device, batch_size=256):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            t = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(proj(t).cpu().numpy())
    return np.concatenate(outs, axis=0)


# ---------------------------------------------------------------------------
# 프로브 방법론 (1~9차와 동일 — Ridge, GroupShuffleSplit, source-level 부트스트랩)
# ---------------------------------------------------------------------------
def held_out_r2(X, Y, groups, seed, n_splits=5, test_size=0.3):
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    scores = []
    for train_idx, test_idx in gss.split(X, Y, groups):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], Y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(r2_score(Y[test_idx], pred))
    return float(np.mean(scores)), float(np.std(scores))


def bootstrap_r2_ci(X, Y, groups, seed, n_boot=500):
    unique_srcs = np.unique(groups)
    src_to_rows = {s: np.where(groups == s)[0] for s in unique_srcs}
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_boot):
        boot_srcs = rng.choice(unique_srcs, size=len(unique_srcs), replace=True)
        oob_srcs = np.setdiff1d(unique_srcs, boot_srcs)
        if len(oob_srcs) < 3:
            continue
        train_idx = np.concatenate([src_to_rows[s] for s in boot_srcs])
        test_idx = np.concatenate([src_to_rows[s] for s in oob_srcs])
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], Y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(r2_score(Y[test_idx], pred))
    scores = np.array(scores)
    return float(scores.mean()), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)), int(len(scores))


def effect_probe(X, effect, emb_all, src_id_arr, effect_idx, seed):
    lo, hi = LEVEL_RAW_RANGE[effect]
    levels_norm = [(v - lo) / (hi - lo) for v in LEVEL_RAW_VALUES[effect]]
    Y = np.tile(np.array(levels_norm), emb_all.shape[0])
    groups = np.repeat(src_id_arr, 3)
    r2_mean, r2_std = held_out_r2(X, Y, groups, seed)
    boot_mean, lo_ci, hi_ci, n_boot = bootstrap_r2_ci(X, Y, groups, seed)
    return {"probe_r2": r2_mean, "probe_r2_std": r2_std, "probe_r2_ci_low": lo_ci, "probe_r2_ci_high": hi_ci,
            "n_rows": int(len(Y)), "n_sources": int(len(np.unique(groups))), "n_boot": n_boot}


def family_probe(X, family_arr, seed, n_reps=20, test_size=0.3):
    accs, nmis = [], []
    sss = StratifiedShuffleSplit(n_splits=n_reps, test_size=test_size, random_state=seed)
    for train_idx, test_idx in sss.split(X, family_arr):
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(X[train_idx], family_arr[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(family_arr[test_idx], pred))
        nmis.append(normalized_mutual_info_score(family_arr[test_idx], pred))
    accs, nmis = np.array(accs), np.array(nmis)
    return {
        "accuracy_mean": float(accs.mean()), "accuracy_ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "nmi_mean": float(nmis.mean()), "nmi_ci": [float(np.percentile(nmis, 2.5)), float(np.percentile(nmis, 97.5))],
        "n_reps": n_reps,
    }


# ---------------------------------------------------------------------------
# LDA 부분공간 (07_subspace.py와 동일 방법론)
# ---------------------------------------------------------------------------
def fit_lda_basis(embeddings, family_labels):
    unique = np.unique(family_labels)
    n_components = min(len(unique) - 1, embeddings.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(embeddings, family_labels)
    basis, _ = np.linalg.qr(lda.scalings_[:, :n_components])
    return basis


def projection_ratio(vec, basis):
    proj = basis @ (basis.T @ vec)
    return float(np.linalg.norm(proj) / (np.linalg.norm(vec) + 1e-12))


def random_unit_vectors(n, dim, seed):
    rng = np.random.RandomState(seed)
    v = rng.normal(size=(n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def bootstrap_cos_ci(cos_per_row, row_to_source, seed, n_boot=1000):
    sources = np.unique(row_to_source)
    rng = np.random.RandomState(seed)
    src_to_rows = {s: np.where(row_to_source == s)[0] for s in sources}
    means = []
    for _ in range(n_boot):
        boot = rng.choice(sources, size=len(sources), replace=True)
        rows = np.concatenate([src_to_rows[s] for s in boot])
        means.append(cos_per_row[rows].mean())
    means = np.array(means)
    return float(cos_per_row.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cos_rows(A, B):
    num = np.sum(A * B, axis=-1)
    den = np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-12
    return num / den


def main():
    parser = argparse.ArgumentParser(description="9차 후속 D — projection layer 진단")
    parser.add_argument("--oat-emb-ts", type=str, default="out/caches/oat_emb_ts.npz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    out_dir = Path(args.out)
    results_dir = out_dir / "results"; figures_dir = out_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"로딩 중: {args.oat_emb_ts}")
    d = np.load(args.oat_emb_ts, allow_pickle=False)
    emb = d["emb"]  # (1200,3,3,512)
    src_id_arr = d["src_id"]
    family_arr = d["instrument_family"]
    n_sources = emb.shape[0]

    print("projection layer 로딩 중 (forward 전용)...")
    proj, proj_dim = load_projection(device)
    print(f"proj_dim = {proj_dim}")

    # =========================================================================
    # D-1 — 변위 감쇠율 (보조)
    # =========================================================================
    print("\n=== D-1: 변위 감쇠율 ===")
    attenuation = {}
    cosine_before_after = {}
    for ei, effect in enumerate(EFFECT_NAMES):
        e_dry = emb[:, ei, 0, :]
        e_wet = emb[:, ei, 2, :]
        proj_dry = proj_forward(proj, e_dry, device)
        proj_wet = proj_forward(proj, e_wet, device)

        r_clap = np.linalg.norm(e_wet - e_dry, axis=-1) / (np.linalg.norm(e_dry, axis=-1) + 1e-12)
        r_proj = np.linalg.norm(proj_wet - proj_dry, axis=-1) / (np.linalg.norm(proj_dry, axis=-1) + 1e-12)
        ratio = r_proj / (r_clap + 1e-12)

        ratio_mean, ratio_lo, ratio_hi = bootstrap_cos_ci(ratio, src_id_arr, args.seed)
        r_clap_mean, _, _ = bootstrap_cos_ci(r_clap, src_id_arr, args.seed)
        r_proj_mean, _, _ = bootstrap_cos_ci(r_proj, src_id_arr, args.seed)

        cos_before = cos_rows(e_wet, e_dry)
        cos_after = cos_rows(proj_wet, proj_dry)
        cb_mean, cb_lo, cb_hi = bootstrap_cos_ci(cos_before, src_id_arr, args.seed)
        ca_mean, ca_lo, ca_hi = bootstrap_cos_ci(cos_after, src_id_arr, args.seed)

        attenuation[effect] = {
            "r_clap_mean": r_clap_mean, "r_proj_mean": r_proj_mean,
            "attenuation_ratio_mean": ratio_mean, "attenuation_ratio_ci": [ratio_lo, ratio_hi],
        }
        cosine_before_after[effect] = {
            "cos_before_mean": cb_mean, "cos_before_ci": [cb_lo, cb_hi],
            "cos_after_mean": ca_mean, "cos_after_ci": [ca_lo, ca_hi],
        }
        print(f"  {effect:<12} r_clap={r_clap_mean:.4f} r_proj={r_proj_mean:.4f} "
              f"감쇠율={ratio_mean:.4f} CI={[round(ratio_lo,4),round(ratio_hi,4)]}  "
              f"cos before={cb_mean:.4f} after={ca_mean:.4f}")

    # 패밀리별 감쇠율(보조)
    attenuation_by_family = {}
    families_sorted = sorted(set(family_arr.tolist()))
    for ei, effect in enumerate(EFFECT_NAMES):
        e_dry = emb[:, ei, 0, :]; e_wet = emb[:, ei, 2, :]
        proj_dry = proj_forward(proj, e_dry, device); proj_wet = proj_forward(proj, e_wet, device)
        r_clap = np.linalg.norm(e_wet - e_dry, axis=-1) / (np.linalg.norm(e_dry, axis=-1) + 1e-12)
        r_proj = np.linalg.norm(proj_wet - proj_dry, axis=-1) / (np.linalg.norm(proj_dry, axis=-1) + 1e-12)
        ratio = r_proj / (r_clap + 1e-12)
        for fam in families_sorted:
            mask = family_arr == fam
            attenuation_by_family.setdefault(effect, {})[fam] = float(ratio[mask].mean())

    # =========================================================================
    # D-2 — ★핵심: 프로브 정보량 전/후 대조
    # =========================================================================
    print("\n=== D-2: 프로브 정보량 전/후 대조 (핵심 검정) ===")
    probe_before_after = {"effect_r2": {}, "family": {}}

    for ei, effect in enumerate(EFFECT_NAMES):
        X_clap = emb[:, ei, :, :].reshape(-1, 512)
        X_proj = proj_forward(proj, X_clap, device)
        before = effect_probe(X_clap, effect, emb, src_id_arr, ei, args.seed)
        after = effect_probe(X_proj, effect, emb, src_id_arr, ei, args.seed)
        probe_before_after["effect_r2"][effect] = {"before": before, "after": after,
                                                     "drop": before["probe_r2"] - after["probe_r2"]}
        print(f"  [이펙트 R²] {effect:<12} before={before['probe_r2']:.4f} CI={[round(before['probe_r2_ci_low'],4),round(before['probe_r2_ci_high'],4)]}  "
              f"after={after['probe_r2']:.4f} CI={[round(after['probe_r2_ci_low'],4),round(after['probe_r2_ci_high'],4)]}  "
              f"하락={before['probe_r2']-after['probe_r2']:+.4f}")

    # 악기 패밀리 — distortion level0(=drive_db 0, 실질적으로 순수 dry)
    dist_idx = EFFECT_NAMES.index("distortion")
    X_clap_fam = emb[:, dist_idx, 0, :]
    X_proj_fam = proj_forward(proj, X_clap_fam, device)
    fam_before = family_probe(X_clap_fam, family_arr, args.seed)
    fam_after = family_probe(X_proj_fam, family_arr, args.seed)
    probe_before_after["family"] = {"before": fam_before, "after": fam_after,
                                     "acc_drop": fam_before["accuracy_mean"] - fam_after["accuracy_mean"],
                                     "nmi_drop": fam_before["nmi_mean"] - fam_after["nmi_mean"]}
    print(f"\n  [악기 패밀리] before acc={fam_before['accuracy_mean']:.4f} NMI={fam_before['nmi_mean']:.4f}  "
          f"after acc={fam_after['accuracy_mean']:.4f} NMI={fam_after['nmi_mean']:.4f}")

    # 판정
    effect_r2_before_mean = np.mean([probe_before_after["effect_r2"][e]["before"]["probe_r2"] for e in EFFECT_NAMES])
    effect_r2_after_mean = np.mean([probe_before_after["effect_r2"][e]["after"]["probe_r2"] for e in EFFECT_NAMES])
    effect_relative_drop = (effect_r2_before_mean - effect_r2_after_mean) / (abs(effect_r2_before_mean) + 1e-12)
    family_relative_drop_nmi = (fam_before["nmi_mean"] - fam_after["nmi_mean"]) / (abs(fam_before["nmi_mean"]) + 1e-12)

    if effect_relative_drop > 0.3 and family_relative_drop_nmi < 0.15:
        d2_verdict = "선택적 폐기 확정 — projection이 이펙트 성분만 골라 버린다"
    elif effect_relative_drop > 0.3 and family_relative_drop_nmi >= 0.15:
        d2_verdict = "일반적 압축 — 이펙트뿐 아니라 악기 정보도 함께 줄어든다(선택적 폐기 아님)"
    else:
        d2_verdict = "projection이 원인이 아님 — 이펙트 R²가 크게 안 떨어짐. transformer 쪽을 의심해야 함(범위 밖)"

    probe_before_after["verdict"] = {
        "effect_r2_before_mean": float(effect_r2_before_mean), "effect_r2_after_mean": float(effect_r2_after_mean),
        "effect_relative_drop": float(effect_relative_drop), "family_nmi_relative_drop": float(family_relative_drop_nmi),
        "verdict": d2_verdict,
    }
    print(f"\n판정: {d2_verdict}")
    print(f"(이펙트 R² 상대하락={effect_relative_drop:.3f}, 악기 NMI 상대하락={family_relative_drop_nmi:.3f})")

    # =========================================================================
    # D-3 — 부분공간 재확인 (J 없이, 평균 차이 벡터)
    # =========================================================================
    print("\n=== D-3: 부분공간 투영 재확인 ===")
    subspace_before_after = {}
    basis_clap = fit_lda_basis(X_clap_fam, family_arr)
    basis_proj = fit_lda_basis(X_proj_fam, family_arr)

    rand_clap = random_unit_vectors(1000, X_clap_fam.shape[1], args.seed)
    rand_proj_vecs = random_unit_vectors(1000, proj_dim, args.seed)
    rand_ratios_clap = np.array([projection_ratio(v, basis_clap) for v in rand_clap])
    rand_ratios_proj = np.array([projection_ratio(v, basis_proj) for v in rand_proj_vecs])

    for ei, effect in enumerate(EFFECT_NAMES):
        e_dry = emb[:, ei, 0, :]; e_wet = emb[:, ei, 2, :]
        v_clap = (e_wet - e_dry).mean(axis=0)
        proj_dry = proj_forward(proj, e_dry, device); proj_wet = proj_forward(proj, e_wet, device)
        v_proj = (proj_wet - proj_dry).mean(axis=0)

        ratio_clap = projection_ratio(v_clap, basis_clap)
        ratio_proj = projection_ratio(v_proj, basis_proj)
        z_clap = (ratio_clap - rand_ratios_clap.mean()) / rand_ratios_clap.std()
        z_proj = (ratio_proj - rand_ratios_proj.mean()) / rand_ratios_proj.std()

        subspace_before_after[effect] = {
            "projection_ratio_before": ratio_clap, "z_before": float(z_clap),
            "projection_ratio_after": ratio_proj, "z_after": float(z_proj),
            "random_baseline_before": {"mean": float(rand_ratios_clap.mean()), "std": float(rand_ratios_clap.std())},
            "random_baseline_after": {"mean": float(rand_ratios_proj.mean()), "std": float(rand_ratios_proj.std())},
        }
        print(f"  {effect:<12} before: ratio={ratio_clap:.4f} z={z_clap:.2f}  |  after: ratio={ratio_proj:.4f} z={z_proj:.2f}")

    # =========================================================================
    # 저장 + 그림
    # =========================================================================
    results = {
        "meta": {"proj_dim": proj_dim, "seed": args.seed, "source_npz": args.oat_emb_ts},
        "depends_on_surrogate": "none",
        "attenuation": attenuation,
        "attenuation_by_family": attenuation_by_family,
        "cosine_before_after": cosine_before_after,
        "probe_before_after": probe_before_after,
        "subspace_before_after": subspace_before_after,
    }
    with open(results_dir / "results_10_projection.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    x = np.arange(len(EFFECT_NAMES))
    vals = [attenuation[e]["attenuation_ratio_mean"] for e in EFFECT_NAMES]
    los = [attenuation[e]["attenuation_ratio_ci"][0] for e in EFFECT_NAMES]
    his = [attenuation[e]["attenuation_ratio_ci"][1] for e in EFFECT_NAMES]
    yerr = np.array([np.array(vals) - np.array(los), np.array(his) - np.array(vals)])
    ax.bar(x, vals, yerr=np.clip(yerr, 0, None), capsize=3, color=[COLORS[e] for e in EFFECT_NAMES], zorder=3)
    ax.axhline(1.0, color=COLORS["null"], linestyle="--", linewidth=1, label="1.0 (변위 보존)")
    ax.set_xticks(x); ax.set_xticklabels(EFFECT_NAMES)
    ax.set_ylabel("감쇠율 = r_proj / r_clap")
    ax.set_title("D-1 — 투영 전후 변위 감쇠율")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "proj_attenuation.png")
    plt.close(fig)

    # ★ 핵심 그림
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    ax = axes[0]
    x = np.arange(len(EFFECT_NAMES))
    width = 0.35
    before_r2 = [probe_before_after["effect_r2"][e]["before"]["probe_r2"] for e in EFFECT_NAMES]
    after_r2 = [probe_before_after["effect_r2"][e]["after"]["probe_r2"] for e in EFFECT_NAMES]
    before_lo = [probe_before_after["effect_r2"][e]["before"]["probe_r2_ci_low"] for e in EFFECT_NAMES]
    before_hi = [probe_before_after["effect_r2"][e]["before"]["probe_r2_ci_high"] for e in EFFECT_NAMES]
    after_lo = [probe_before_after["effect_r2"][e]["after"]["probe_r2_ci_low"] for e in EFFECT_NAMES]
    after_hi = [probe_before_after["effect_r2"][e]["after"]["probe_r2_ci_high"] for e in EFFECT_NAMES]
    yerr_b = np.array([np.array(before_r2) - np.array(before_lo), np.array(before_hi) - np.array(before_r2)])
    yerr_a = np.array([np.array(after_r2) - np.array(after_lo), np.array(after_hi) - np.array(after_r2)])
    ax.bar(x - width / 2, before_r2, width, yerr=np.clip(yerr_b, 0, None), capsize=3, label="투영 전 (e_clap, 512d)", color=COLORS["before"], zorder=3)
    ax.bar(x + width / 2, after_r2, width, yerr=np.clip(yerr_a, 0, None), capsize=3, label="투영 후 (ê, 1024d)", color=COLORS["after"], zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(EFFECT_NAMES)
    ax.set_ylabel("이펙트 파라미터 프로브 R²")
    ax.set_title("이펙트 정보")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    ax = axes[1]
    labels = ["accuracy", "NMI"]
    before_vals = [fam_before["accuracy_mean"], fam_before["nmi_mean"]]
    after_vals = [fam_after["accuracy_mean"], fam_after["nmi_mean"]]
    x2 = np.arange(2)
    ax.bar(x2 - width / 2, before_vals, width, label="투영 전 (512d)", color=COLORS["before"], zorder=3)
    ax.bar(x2 + width / 2, after_vals, width, label="투영 후 (1024d)", color=COLORS["after"], zorder=3)
    ax.set_xticks(x2); ax.set_xticklabels(labels)
    ax.set_ylabel("악기 패밀리 분류 성능")
    ax.set_title("악기 정보")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    fig.suptitle(f"D-2 — 이펙트 R² vs 악기 정보, 투영 전/후  [{d2_verdict}]", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures_dir / "proj_information.png")
    plt.close(fig)

    print(f"\n저장: {results_dir/'results_10_projection.json'}, {figures_dir/'proj_attenuation.png'}, {figures_dir/'proj_information.png'}")
    print("★ D-2를 완료했습니다. 판정을 확인하세요.")


if __name__ == "__main__":
    main()
