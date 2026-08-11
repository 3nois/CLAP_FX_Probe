"""CLAP FX Probe — 21_handle_predict_phase1.py (6차 후속 과제 8, Phase 1: 정방향/역방향 손잡이 예측)

7차에서 "손잡이 방향이 소스마다 다르다"(within > between)를 확인했다. 이번 질문은
"그 소스의 손잡이를 예측할 수 있는가"다. 최종 도구("이 소리에서 리버브 뺀 음색을 줘")는
방향(어느 쪽이 dry인가 — 기계가 예측)과 크기(얼마나 갈까 — 사용자가 슬라이더로 조절)를
분리한다. 이 스크립트는 방향 예측이 가능한지를 검정한다.

재렌더링 0회 — `out/caches/oat_emb.npz`(과제 7, 1,200소스×3레벨×3이펙트, 조건A)만 읽는다.

  과제 A   정방향: e_dry(레벨0) -> v = e(레벨2) - e(레벨0)
  과제 B1  역방향(파라미터 known, 상한): e(레벨2) -> v_to_dry = e(레벨0) - e(레벨2)
  과제 B2  역방향(파라미터 unknown, ★진짜 질문): e(레벨1 또는 2, 라벨 없이 섞음)
           -> v_to_dry = e(레벨0) - e(현재 레벨)

모델 4종(①전역평균 ②패밀리평균오라클 — 둘 다 학습 아닌 기준선 — ③선형 ④MLP)을 방향
(단위벡터, 1-cos 손실)과 크기(스칼라, 상대오차 손실)로 분리된 두 헤드로 낸다. ③④는
동일 시드·동일 epoch 예산으로 비교한다. 결과 해석은 이 스크립트가 단정하지 않는다.
README 6차 후속 절의 판정 기준표를 따를 것.
"""
import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
COLORS = {"reverb": "#2a78d6", "distortion": "#eb6834", "highshelf": "#1baf7a", "baseline": "#898781", "null": "#e34948"}
MODEL_NAMES = ["global_mean", "family_mean_oracle", "linear", "mlp"]
EFFECT_NAMES = ["reverb", "distortion", "highshelf"]

BASELINE_GLOBAL_REF = 0.24  # 7차 참고치 (본 스크립트가 독립적으로 재계산해 대조)
BASELINE_FAMILY_REF = 0.34


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 분할 — 패밀리별 층화 80/10/10, 소스 단위
# ---------------------------------------------------------------------------
def stratified_split(family_arr, seed, train_frac=0.8, val_frac=0.1):
    rng = np.random.RandomState(seed)
    families = sorted(set(family_arr.tolist()))
    train_idx, val_idx, test_idx = [], [], []
    per_family = {}
    for fam in families:
        idx = np.where(family_arr == fam)[0]
        perm = rng.permutation(idx)
        n = len(perm)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        tr, va, te = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
        train_idx.append(tr); val_idx.append(va); test_idx.append(te)
        per_family[fam] = {"n_total": int(n), "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))}
    return (np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx), per_family)


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
class LinearDualHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dir_out = nn.Linear(dim, dim)
        self.mag_out = nn.Linear(dim, 1)

    def forward(self, x):
        raw = self.dir_out(x)
        direction = raw / (raw.norm(dim=-1, keepdim=True) + 1e-8)
        magnitude = F.softplus(self.mag_out(x)).squeeze(-1)
        return direction, magnitude


class MLPDualHead(nn.Module):
    def __init__(self, dim=512, hidden=1024, dropout=0.1):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.dir_out = nn.Linear(hidden, dim)
        self.mag_out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        raw = self.dir_out(h)
        direction = raw / (raw.norm(dim=-1, keepdim=True) + 1e-8)
        magnitude = F.softplus(self.mag_out(h)).squeeze(-1)
        return direction, magnitude


def unit_np(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def dual_head_loss(dir_pred, mag_pred, dir_true, mag_true, lambda_mag=0.3):
    cos = (dir_pred * dir_true).sum(-1)
    dir_loss = (1 - cos).mean()
    mag_loss = (torch.abs(mag_pred - mag_true) / (mag_true + 1e-8)).mean()
    return dir_loss + lambda_mag * mag_loss, dir_loss, mag_loss


def train_dual_head(model, X_train, Y_train, X_val, Y_val, seed, max_epochs=300, patience=30, lr=1e-3,
                     weight_decay=1e-4, lambda_mag=0.3):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Yt_dir = F.normalize(Y_train, dim=-1)
    Yt_mag = Y_train.norm(dim=-1)
    Yv_dir = F.normalize(Y_val, dim=-1)
    Yv_mag = Y_val.norm(dim=-1)

    best_val_cos = -2.0
    best_state = None
    patience_ctr = 0
    for epoch in range(max_epochs):
        model.train()
        dp, mp = model(X_train)
        loss, _, _ = dual_head_loss(dp, mp, Yt_dir, Yt_mag, lambda_mag)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            dpv, mpv = model(X_val)
            val_cos = (dpv * Yv_dir).sum(-1).mean().item()
        if val_cos > best_val_cos:
            best_val_cos = val_cos
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_val_cos, epoch + 1


def predict_global_mean(Y_train, X_eval_n):
    v_const = Y_train.mean(axis=0)
    direction = unit_np(v_const)
    magnitude = np.linalg.norm(v_const)
    return np.tile(direction, (X_eval_n, 1)), np.full(X_eval_n, magnitude)


def predict_family_mean_oracle(Y_train, family_train, family_eval):
    families = sorted(set(family_train.tolist()))
    fam_vec = {}
    global_fallback = Y_train.mean(axis=0)
    for fam in families:
        mask = family_train == fam
        fam_vec[fam] = Y_train[mask].mean(axis=0) if mask.sum() > 0 else global_fallback
    dirs, mags = [], []
    for fam in family_eval:
        v = fam_vec.get(fam, global_fallback)
        dirs.append(unit_np(v))
        mags.append(np.linalg.norm(v))
    return np.array(dirs), np.array(mags)


def bootstrap_cos_ci(cos_per_row, row_to_source, seed, n_boot=1000):
    sources = np.unique(row_to_source)
    rng = np.random.RandomState(seed)
    means = []
    src_to_rows = {s: np.where(row_to_source == s)[0] for s in sources}
    for _ in range(n_boot):
        boot_srcs = rng.choice(sources, size=len(sources), replace=True)
        rows = np.concatenate([src_to_rows[s] for s in boot_srcs])
        means.append(float(cos_per_row[rows].mean()))
    means = np.array(means)
    return float(means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_all_models(X_train, Y_train, fam_train, X_val, Y_val, X_test, Y_test, fam_test, src_test, seed,
                    max_epochs=300, patience=30):
    """반환: {model_name: {"cos_per_row":.., "mag_pred":.., "mag_true":.., "val_cos":.., "epochs_used":..}}"""
    results = {}
    n_test = X_test.shape[0]
    Yt_dir_test = unit_np(Y_test)
    Yt_mag_test = np.linalg.norm(Y_test, axis=-1)

    # ① 전역 평균
    dir_pred, mag_pred = predict_global_mean(Y_train, n_test)
    cos = np.sum(dir_pred * Yt_dir_test, axis=-1)
    results["global_mean"] = {"cos_per_row": cos, "mag_pred": mag_pred, "mag_true": Yt_mag_test, "val_cos": None, "epochs_used": None}

    # ② 패밀리 평균(오라클)
    dir_pred, mag_pred = predict_family_mean_oracle(Y_train, fam_train, fam_test)
    cos = np.sum(dir_pred * Yt_dir_test, axis=-1)
    results["family_mean_oracle"] = {"cos_per_row": cos, "mag_pred": mag_pred, "mag_true": Yt_mag_test, "val_cos": None, "epochs_used": None}

    # ③④ 학습 모델
    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    Ytr_t = torch.tensor(Y_train, dtype=torch.float32)
    Xval_t = torch.tensor(X_val, dtype=torch.float32)
    Yval_t = torch.tensor(Y_val, dtype=torch.float32)
    Xte_t = torch.tensor(X_test, dtype=torch.float32)

    for name, model_fn, wd in [
        ("linear", lambda: LinearDualHead(512), 1e-2),
        ("mlp", lambda: MLPDualHead(512, 1024, 0.1), 1e-4),
    ]:
        model = model_fn()
        model, best_val_cos, epochs_used = train_dual_head(
            model, Xtr_t, Ytr_t, Xval_t, Yval_t, seed, max_epochs=max_epochs, patience=patience, weight_decay=wd)
        model.eval()
        with torch.no_grad():
            dp, mp = model(Xte_t)
        dp = dp.numpy(); mp = mp.numpy()
        cos = np.sum(dp * Yt_dir_test, axis=-1)
        results[name] = {"cos_per_row": cos, "mag_pred": mp, "mag_true": Yt_mag_test, "val_cos": best_val_cos, "epochs_used": epochs_used}

    for name in results:
        r = results[name]
        mean_cos, ci_lo, ci_hi = bootstrap_cos_ci(r["cos_per_row"], src_test, seed)
        r["cos_mean"] = mean_cos
        r["cos_median"] = float(np.median(r["cos_per_row"]))
        r["cos_ci"] = [ci_lo, ci_hi]
        rel_err = np.abs(r["mag_pred"] - r["mag_true"]) / (r["mag_true"] + 1e-8)
        r["mag_rel_err_median"] = float(np.median(rel_err))
        r["mag_rel_err_mean"] = float(np.mean(rel_err))
        ss_res = float(np.sum((r["mag_true"] - r["mag_pred"]) ** 2))
        ss_tot = float(np.sum((r["mag_true"] - r["mag_true"].mean()) ** 2))
        r["mag_r2"] = 1 - ss_res / ss_tot if ss_tot > 1e-12 else None
        del r["cos_per_row"], r["mag_pred"], r["mag_true"]
    return results


def main():
    parser = argparse.ArgumentParser(description="6차 후속 과제 8 Phase 1 — 정방향/역방향(B1,B2) 손잡이 예측")
    parser.add_argument("--oat-emb", type=str, default="out/caches/oat_emb.npz")
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lambda-mag", type=float, default=0.3)
    args = parser.parse_args()

    out_dir = Path(args.out)
    results_dir = out_dir / "results"
    figures_dir = out_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"로딩 중: {args.oat_emb}")
    d = np.load(args.oat_emb, allow_pickle=False)
    emb = d["emb"]  # (1200,3,3,512)
    src_id = d["src_id"]
    family = d["instrument_family"]
    n_sources = emb.shape[0]

    train_idx, val_idx, test_idx, per_family = stratified_split(family, args.seed)
    print(f"분할: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} (소스 단위, 패밀리 층화 80/10/10)")

    split_meta = {
        "train_frac": 0.8, "val_frac": 0.1, "test_frac": 0.1, "seed": args.seed,
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "per_family": per_family,
    }

    # =========================================================================
    # 과제 A — 정방향
    # =========================================================================
    print("\n=== 과제 A: 정방향 (e_dry -> v) ===")
    forward_results = {}
    for ei, effect in enumerate(EFFECT_NAMES):
        X = emb[:, ei, 0, :]
        Y = emb[:, ei, 2, :] - emb[:, ei, 0, :]
        res = run_all_models(
            X[train_idx], Y[train_idx], family[train_idx],
            X[val_idx], Y[val_idx],
            X[test_idx], Y[test_idx], family[test_idx], src_id[test_idx],
            args.seed, args.max_epochs, args.patience,
        )
        forward_results[effect] = res
        print(f"  {effect}: " + "  ".join(f"{m}={res[m]['cos_mean']:.4f}[{res[m]['cos_ci'][0]:.3f},{res[m]['cos_ci'][1]:.3f}]" for m in MODEL_NAMES))

    # P26 검정
    p26 = {}
    for effect in EFFECT_NAMES:
        best_learned = max(forward_results[effect]["linear"]["cos_mean"], forward_results[effect]["mlp"]["cos_mean"])
        p26[effect] = {
            "best_learned_cos": best_learned, "threshold": BASELINE_FAMILY_REF,
            "exceeds_family_baseline": best_learned > BASELINE_FAMILY_REF,
        }
    print("P26 (학습 모델 cos > 0.34?):", {e: p26[e]["exceeds_family_baseline"] for e in EFFECT_NAMES})

    # =========================================================================
    # 과제 B1 — 역방향, 파라미터 known (상한)
    # =========================================================================
    print("\n=== 과제 B1: 역방향(파라미터 known, 상한) ===")
    b1_results = {}
    for ei, effect in enumerate(EFFECT_NAMES):
        X = emb[:, ei, 2, :]
        Y = emb[:, ei, 0, :] - emb[:, ei, 2, :]
        res = run_all_models(
            X[train_idx], Y[train_idx], family[train_idx],
            X[val_idx], Y[val_idx],
            X[test_idx], Y[test_idx], family[test_idx], src_id[test_idx],
            args.seed, args.max_epochs, args.patience,
        )
        b1_results[effect] = res
        print(f"  {effect}: " + "  ".join(f"{m}={res[m]['cos_mean']:.4f}" for m in MODEL_NAMES))

    # =========================================================================
    # 과제 B2 — 역방향, 파라미터 unknown (★ 진짜 질문). 레벨1/2 혼합, 라벨 없이.
    # =========================================================================
    print("\n=== 과제 B2: 역방향(파라미터 unknown, ★진짜 질문) ===")
    b2_results = {}
    b2_by_level_cos = {}  # 사후 분석용 — 모델 입력엔 레벨 정보 없음, 평가 시에만 분리
    for ei, effect in enumerate(EFFECT_NAMES):
        X1 = emb[:, ei, 1, :]; Y1 = emb[:, ei, 0, :] - emb[:, ei, 1, :]
        X2 = emb[:, ei, 2, :]; Y2 = emb[:, ei, 0, :] - emb[:, ei, 2, :]
        X_all = np.concatenate([X1, X2], axis=0)
        Y_all = np.concatenate([Y1, Y2], axis=0)
        fam_all = np.concatenate([family, family], axis=0)
        src_all = np.concatenate([src_id, src_id], axis=0)
        level_all = np.concatenate([np.ones(n_sources, dtype=int), np.full(n_sources, 2, dtype=int)])
        pos_all = np.concatenate([np.arange(n_sources), np.arange(n_sources)])  # 원본 소스 위치(분할 인덱스와 매칭용)

        def sub(idx_src):
            mask = np.isin(pos_all, idx_src)
            return X_all[mask], Y_all[mask], fam_all[mask], src_all[mask], level_all[mask]

        Xtr, Ytr, Ftr, _, _ = sub(train_idx)
        Xva, Yva, _, _, _ = sub(val_idx)
        Xte, Yte, Fte, Ste, Lte = sub(test_idx)

        res = run_all_models(Xtr, Ytr, Ftr, Xva, Yva, Xte, Yte, Fte, Ste, args.seed, args.max_epochs, args.patience)
        b2_results[effect] = res
        print(f"  {effect}: " + "  ".join(f"{m}={res[m]['cos_mean']:.4f}" for m in MODEL_NAMES))

        # 레벨별 사후 분리(방향 cos) — MLP만, 진단용
        Xte_t = torch.tensor(Xte, dtype=torch.float32)
        model = MLPDualHead(512, 1024, 0.1)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32); Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
        Xva_t = torch.tensor(Xva, dtype=torch.float32); Yva_t = torch.tensor(Yva, dtype=torch.float32)
        model, _, _ = train_dual_head(model, Xtr_t, Ytr_t, Xva_t, Yva_t, args.seed, args.max_epochs, args.patience)
        model.eval()
        with torch.no_grad():
            dp, mp = model(Xte_t)
        dp = dp.numpy(); mp = mp.numpy()
        Yte_dir = unit_np(Yte)
        cos_all = np.sum(dp * Yte_dir, axis=-1)
        mag_true = np.linalg.norm(Yte, axis=-1)
        b2_by_level_cos[effect] = {
            "level1": {"cos_mean": float(cos_all[Lte == 1].mean()), "mag_mean": float(mag_true[Lte == 1].mean()), "n": int((Lte == 1).sum())},
            "level2": {"cos_mean": float(cos_all[Lte == 2].mean()), "mag_mean": float(mag_true[Lte == 2].mean()), "n": int((Lte == 2).sum())},
        }

    # P27 검정 (B1 vs B2, MLP 기준)
    p27 = {}
    for effect in EFFECT_NAMES:
        b1_cos = b1_results[effect]["mlp"]["cos_mean"]
        b2_cos = b2_results[effect]["mlp"]["cos_mean"]
        p27[effect] = {"b1_cos": b1_cos, "b2_cos": b2_cos, "gap": b1_cos - b2_cos, "b1_greater": b1_cos > b2_cos}
    print("P27 (B1 > B2, MLP 기준):", {e: (round(p27[e]["b1_cos"], 4), round(p27[e]["b2_cos"], 4), p27[e]["b1_greater"]) for e in EFFECT_NAMES})

    # =========================================================================
    # 그림
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
    for ax, effect in zip(axes, EFFECT_NAMES):
        vals = [forward_results[effect][m]["cos_mean"] for m in MODEL_NAMES]
        los = [forward_results[effect][m]["cos_ci"][0] for m in MODEL_NAMES]
        his = [forward_results[effect][m]["cos_ci"][1] for m in MODEL_NAMES]
        x = np.arange(len(MODEL_NAMES))
        yerr = np.array([np.clip(np.array(vals) - np.array(los), 0, None), np.clip(np.array(his) - np.array(vals), 0, None)])
        ax.bar(x, vals, yerr=yerr, capsize=3, color=COLORS[effect], zorder=3)
        ax.axhline(BASELINE_GLOBAL_REF, color=COLORS["baseline"], linestyle=":", linewidth=1, label=f"전역평균 참고값={BASELINE_GLOBAL_REF}")
        ax.axhline(BASELINE_FAMILY_REF, color=COLORS["null"], linestyle="--", linewidth=1, label=f"패밀리평균 참고값={BASELINE_FAMILY_REF}")
        ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES, rotation=25, ha="right", fontsize=8)
        ax.set_title(effect, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=7)
        style_axis(ax)
    axes[0].set_ylabel("held-out cos(v_pred, v_true)")
    fig.suptitle("과제 A — 정방향 예측 (모델별 + 기준선 2종)")
    fig.tight_layout()
    fig.savefig(figures_dir / "predict_forward.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
    for ax, effect in zip(axes, EFFECT_NAMES):
        x = np.arange(len(MODEL_NAMES))
        width = 0.35
        b1_vals = [b1_results[effect][m]["cos_mean"] for m in MODEL_NAMES]
        b2_vals = [b2_results[effect][m]["cos_mean"] for m in MODEL_NAMES]
        ax.bar(x - width / 2, b1_vals, width, label="B1 (known, 상한)", color=COLORS[effect], alpha=0.9, zorder=3)
        ax.bar(x + width / 2, b2_vals, width, label="B2 (unknown, 진짜 질문)", color=COLORS[effect], alpha=0.45, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES, rotation=25, ha="right", fontsize=8)
        ax.set_title(effect, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=7)
        style_axis(ax)
    axes[0].set_ylabel("held-out cos(v_to_dry_pred, v_to_dry_true)")
    fig.suptitle("과제 B1 vs B2 — 파라미터 known(상한) vs unknown(진짜 질문)")
    fig.tight_layout()
    fig.savefig(figures_dir / "predict_reverse.png")
    plt.close(fig)

    # =========================================================================
    # 저장
    # =========================================================================
    def tag_none(d):
        if isinstance(d, dict):
            return {k: tag_none(v) for k, v in d.items()}
        return d

    results8 = {
        "meta": {
            "split": split_meta,
            "model_config": {
                "max_epochs": args.max_epochs, "patience": args.patience, "lambda_mag": args.lambda_mag,
                "linear_weight_decay": 1e-2, "mlp_weight_decay": 1e-4, "mlp_hidden": 1024, "mlp_dropout": 0.1,
                "seed": args.seed,
            },
            "source_npz": args.oat_emb,
        },
        "depends_on_surrogate": "none",
        "baselines": {
            "global_mean_reference_from_7th": BASELINE_GLOBAL_REF,
            "family_mean_reference_from_7th": BASELINE_FAMILY_REF,
            "note": "이 스크립트가 독립적으로 재계산한 값은 forward[effect][global_mean|family_mean_oracle].cos_mean",
        },
        "forward": tag_none(forward_results),
        "reverse_b1": tag_none(b1_results),
        "reverse_b2": tag_none(b2_results),
        "reverse_b2_by_level": b2_by_level_cos,
        "prereg_checks": {"P26": p26, "P27": p27},
    }
    results8_path = results_dir / "results_8.json"
    with open(results8_path, "w") as f:
        json.dump(results8, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 1 요약 ===")
    print("\n과제 A (정방향) — MLP cos:")
    for e in EFFECT_NAMES:
        print(f"  {e:<12} {forward_results[e]['mlp']['cos_mean']:.4f}  (0.34 초과: {p26[e]['exceeds_family_baseline']})")
    print("\n과제 B1 vs B2 (역방향) — MLP cos:")
    for e in EFFECT_NAMES:
        print(f"  {e:<12} B1={b1_results[e]['mlp']['cos_mean']:.4f}  B2={b2_results[e]['mlp']['cos_mean']:.4f}  격차={p27[e]['gap']:.4f}")
    print("\nB2 레벨별 분리(MLP, 진단):")
    for e in EFFECT_NAMES:
        print(f"  {e}: {b2_by_level_cos[e]}")
    print(f"\n저장: {results8_path}, {figures_dir/'predict_forward.png'}, {figures_dir/'predict_reverse.png'}")
    print("★ 과제 A와 B2를 완료했습니다. 여기서 멈춥니다 (B3·C·D는 다음 단계).")


if __name__ == "__main__":
    main()
