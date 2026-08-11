"""CLAP FX Probe — 22_handle_predict_phase2.py (6차 후속 과제 8, Phase 2: 크기/복원/LOFO)

Phase 1(21_handle_predict_phase1.py)에서 과제 A(정방향)와 B2(역방향·파라미터 unknown,
★진짜 질문)를 완료했다 — 세 이펙트 전부 B2 방향 cos > 0.6로 "손잡이 구현 가능" 판정.
이 스크립트는 나머지 셋을 마저 낸다.

  과제 B3  크기 예측 — B2에서 이미 학습한 듀얼헤드 모델의 크기 헤드를 그대로 평가한다
           (새 모델 아님 — B2 입력·데이터가 동일하므로 재사용). 실패해도 프로젝트
           결론에 영향 없음(크기는 사용자 슬라이더 몫).
  과제 C   복원 검증 — e_dry_hat = e_wet + v_to_dry_pred, cos(e_dry_hat, e_dry_true)를
           기준선 cos(e_wet, e_dry_true)(아무것도 안 했을 때)와 비교. B2(실전) 기준,
           B1(상한) 버전도 병기.
  과제 D   LOFO 진단(부차) — 과제 A(정방향) MLP로 10-fold, 안 본 패밀리 테스트.
           층화 결과와의 격차로 "패밀리 템플릿 암기 여부"를 본다.

재렌더링 0회, `out/caches/oat_emb.npz`만 읽는다. Phase 1과 동일한 시드로 동일한
분할·모델 설정을 재현한다(모델 가중치는 저장하지 않았으므로 필요한 것만 재학습 —
소규모 데이터라 수 초 내로 끝난다). 결과 해석은 이 스크립트가 단정하지 않는다.
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
EFFECT_NAMES = ["reverb", "distortion", "highshelf"]


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 21_handle_predict_phase1.py와 동일 — 파일명이 숫자로 시작해 import 불가하므로 복제
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
    return dir_loss + lambda_mag * mag_loss


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
    epoch = 0
    for epoch in range(max_epochs):
        model.train()
        dp, mp = model(X_train)
        loss = dual_head_loss(dp, mp, Yt_dir, Yt_mag, lambda_mag)
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


def bootstrap_mean_ci(vals_per_row, row_to_source, seed, n_boot=1000):
    sources = np.unique(row_to_source)
    rng = np.random.RandomState(seed)
    src_to_rows = {s: np.where(row_to_source == s)[0] for s in sources}
    means = []
    for _ in range(n_boot):
        boot_srcs = rng.choice(sources, size=len(sources), replace=True)
        rows = np.concatenate([src_to_rows[s] for s in boot_srcs])
        means.append(float(vals_per_row[rows].mean()))
    means = np.array(means)
    return float(means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fit_mlp(Xtr, Ytr, Xva, Yva, seed, max_epochs, patience):
    model = MLPDualHead(512, 1024, 0.1)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32); Yva_t = torch.tensor(Yva, dtype=torch.float32)
    model, val_cos, epochs_used = train_dual_head(model, Xtr_t, Ytr_t, Xva_t, Yva_t, seed, max_epochs, patience)
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        dp, mp = model(torch.tensor(X, dtype=torch.float32))
    return dp.numpy(), mp.numpy()


def main():
    parser = argparse.ArgumentParser(description="6차 후속 과제 8 Phase 2 — 크기(B3)/복원(C)/LOFO(D)")
    parser.add_argument("--oat-emb", type=str, default="out/caches/oat_emb.npz")
    parser.add_argument("--results8", type=str, default="out/results/results_8.json")
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out)
    results_dir = out_dir / "results"
    figures_dir = out_dir / "figures"

    print(f"로딩 중: {args.oat_emb}")
    d = np.load(args.oat_emb, allow_pickle=False)
    emb = d["emb"]
    src_id = d["src_id"]
    family = d["instrument_family"]
    n_sources = emb.shape[0]

    with open(args.results8) as f:
        r8 = json.load(f)

    train_idx, val_idx, test_idx, per_family = stratified_split(family, args.seed)
    print(f"분할(Phase 1과 동일 시드로 재현): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # =========================================================================
    # 과제 B3 — 크기 예측 (B1/B2 모델 재학습 후 크기 헤드 평가 — 새 태스크 아님)
    #           + 과제 C — 복원 검증 (같은 모델의 예측을 그대로 재사용)
    # =========================================================================
    print("\n=== 과제 B3(크기) + C(복원 검증) 계산 중 ===")
    b3_results = {}
    reconstruction_results = {}
    for ei, effect in enumerate(EFFECT_NAMES):
        # ---- B1 데이터/모델 (상한 참고용) ----
        X_b1 = emb[:, ei, 2, :]
        Y_b1 = emb[:, ei, 0, :] - emb[:, ei, 2, :]
        model_b1 = fit_mlp(X_b1[train_idx], Y_b1[train_idx], X_b1[val_idx], Y_b1[val_idx], args.seed, args.max_epochs, args.patience)
        dp_b1, mp_b1 = predict(model_b1, X_b1[test_idx])

        # ---- B2 데이터/모델 (레벨1+2 혼합, 진짜 질문) ----
        X1 = emb[:, ei, 1, :]; Y1 = emb[:, ei, 0, :] - emb[:, ei, 1, :]
        X2 = emb[:, ei, 2, :]; Y2 = emb[:, ei, 0, :] - emb[:, ei, 2, :]
        X_all = np.concatenate([X1, X2], axis=0)
        Y_all = np.concatenate([Y1, Y2], axis=0)
        pos_all = np.concatenate([np.arange(n_sources), np.arange(n_sources)])
        level_all = np.concatenate([np.ones(n_sources, dtype=int), np.full(n_sources, 2, dtype=int)])
        src_all_dup = np.concatenate([src_id, src_id])

        def sub(idx_src):
            mask = np.isin(pos_all, idx_src)
            return X_all[mask], Y_all[mask], src_all_dup[mask], level_all[mask], pos_all[mask]

        Xtr2, Ytr2, _, _, _ = sub(train_idx)
        Xva2, Yva2, _, _, _ = sub(val_idx)
        Xte2, Yte2, Ste2, Lte2, Pte2 = sub(test_idx)

        model_b2 = fit_mlp(Xtr2, Ytr2, Xva2, Yva2, args.seed, args.max_epochs, args.patience)
        dp_b2, mp_b2 = predict(model_b2, Xte2)

        # ---- B3: 크기 예측 성능 (B2 모델의 크기 헤드) ----
        mag_true2 = np.linalg.norm(Yte2, axis=-1)
        rel_err = np.abs(mp_b2 - mag_true2) / (mag_true2 + 1e-8)
        ss_res = float(np.sum((mag_true2 - mp_b2) ** 2))
        ss_tot = float(np.sum((mag_true2 - mag_true2.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else None
        mag_mean, mag_lo, mag_hi = bootstrap_mean_ci(1 - rel_err, Ste2, args.seed)  # "1-상대오차"를 cos류와 같은 방향(높을수록 좋음)으로 부트스트랩
        b3_results[effect] = {
            "mag_r2": r2, "mag_rel_err_median": float(np.median(rel_err)), "mag_rel_err_mean": float(np.mean(rel_err)),
            "one_minus_rel_err_boot_mean": mag_mean, "one_minus_rel_err_ci": [mag_lo, mag_hi],
            "n": int(len(mag_true2)), "note": "B2(레벨1+2 혼합) 모델의 크기 헤드를 그대로 평가 — 별도 모델 아님",
        }

        # ---- 과제 C: 복원 검증 ----
        # B2(실전): e_wet = X_te2 (레벨1 또는 2), e_dry_true = emb[level0][해당 소스]
        e_dry_true_b2 = emb[:, ei, 0, :][Pte2]
        v_pred_b2 = dp_b2 * mp_b2[:, None]
        e_dry_hat_b2 = Xte2 + v_pred_b2
        cos_recon_b2 = np.sum(unit_np(e_dry_hat_b2) * unit_np(e_dry_true_b2), axis=-1)
        cos_base_b2 = np.sum(unit_np(Xte2) * unit_np(e_dry_true_b2), axis=-1)

        # B1(상한): e_wet = level2, e_dry_true = level0
        e_dry_true_b1 = emb[:, ei, 0, :][test_idx]
        v_pred_b1 = dp_b1 * mp_b1[:, None]
        e_dry_hat_b1 = X_b1[test_idx] + v_pred_b1
        cos_recon_b1 = np.sum(unit_np(e_dry_hat_b1) * unit_np(e_dry_true_b1), axis=-1)
        cos_base_b1 = np.sum(unit_np(X_b1[test_idx]) * unit_np(e_dry_true_b1), axis=-1)

        recon_b2_mean, recon_b2_lo, recon_b2_hi = bootstrap_mean_ci(cos_recon_b2, Ste2, args.seed)
        base_b2_mean, base_b2_lo, base_b2_hi = bootstrap_mean_ci(cos_base_b2, Ste2, args.seed)
        recon_b1_mean, recon_b1_lo, recon_b1_hi = bootstrap_mean_ci(cos_recon_b1, src_id[test_idx], args.seed)
        base_b1_mean, base_b1_lo, base_b1_hi = bootstrap_mean_ci(cos_base_b1, src_id[test_idx], args.seed)

        reconstruction_results[effect] = {
            "b2_practical": {
                "reconstruction_cos_mean": recon_b2_mean, "reconstruction_cos_ci": [recon_b2_lo, recon_b2_hi],
                "baseline_do_nothing_cos_mean": base_b2_mean, "baseline_do_nothing_cos_ci": [base_b2_lo, base_b2_hi],
                "improvement": recon_b2_mean - base_b2_mean,
                "exceeds_baseline": recon_b2_lo > base_b2_hi,
            },
            "b1_upper_bound": {
                "reconstruction_cos_mean": recon_b1_mean, "reconstruction_cos_ci": [recon_b1_lo, recon_b1_hi],
                "baseline_do_nothing_cos_mean": base_b1_mean, "baseline_do_nothing_cos_ci": [base_b1_lo, base_b1_hi],
                "improvement": recon_b1_mean - base_b1_mean,
                "exceeds_baseline": recon_b1_lo > base_b1_hi,
            },
        }
        print(f"  {effect}: B3 mag_r2={r2:.4f} rel_err_median={b3_results[effect]['mag_rel_err_median']:.4f}  |  "
              f"C(B2) recon={recon_b2_mean:.4f} vs baseline={base_b2_mean:.4f} (개선 {recon_b2_mean-base_b2_mean:+.4f})  |  "
              f"C(B1) recon={recon_b1_mean:.4f} vs baseline={base_b1_mean:.4f} (개선 {recon_b1_mean-base_b1_mean:+.4f})")

    # P28: B2 방향(cos) > B3 크기(R²)?
    p28 = {}
    for effect in EFFECT_NAMES:
        b2_dir_cos = r8["reverse_b2"][effect]["mlp"]["cos_mean"]
        b3_mag_r2 = b3_results[effect]["mag_r2"]
        p28[effect] = {"b2_direction_cos": b2_dir_cos, "b3_magnitude_r2": b3_mag_r2, "direction_greater": b2_dir_cos > b3_mag_r2}
    print("\nP28 (B2 방향 > B3 크기?):", {e: (round(p28[e]["b2_direction_cos"], 3), round(p28[e]["b3_magnitude_r2"], 3), p28[e]["direction_greater"]) for e in EFFECT_NAMES})

    # =========================================================================
    # 과제 D — LOFO 진단 (부차, 과제 A 정방향 MLP로 10-fold)
    # =========================================================================
    print("\n=== 과제 D: LOFO 진단 (정방향, MLP, 10-fold) ===")
    families_sorted = sorted(set(family.tolist()))
    lofo_results = {effect: {} for effect in EFFECT_NAMES}
    rng_lofo = np.random.RandomState(args.seed)
    for ei, effect in enumerate(EFFECT_NAMES):
        X = emb[:, ei, 0, :]
        Y = emb[:, ei, 2, :] - emb[:, ei, 0, :]
        fold_cos = []
        for held_out_fam in families_sorted:
            train_pool = np.where(family != held_out_fam)[0]
            test_fam_idx = np.where(family == held_out_fam)[0]
            # 훈련 풀에서 10%를 조기종료용 val로 분리 (소스 단위, 시드 고정)
            perm = rng_lofo.permutation(train_pool)
            n_val = max(1, int(round(len(perm) * 0.1)))
            val_fold_idx = perm[:n_val]
            train_fold_idx = perm[n_val:]

            model = fit_mlp(X[train_fold_idx], Y[train_fold_idx], X[val_fold_idx], Y[val_fold_idx], args.seed, args.max_epochs, args.patience)
            dp, _ = predict(model, X[test_fam_idx])
            yt_dir = unit_np(Y[test_fam_idx])
            cos = np.sum(dp * yt_dir, axis=-1)
            lofo_results[effect][held_out_fam] = {"cos_mean": float(cos.mean()), "n": int(len(cos))}
            fold_cos.append(cos)
        all_cos = np.concatenate(fold_cos)
        stratified_cos = r8["forward"][effect]["mlp"]["cos_mean"]
        lofo_results[effect]["_overall"] = {
            "lofo_mean_cos": float(all_cos.mean()),
            "stratified_test_cos_ref": stratified_cos,
            "gap_stratified_minus_lofo": stratified_cos - float(all_cos.mean()),
        }
        print(f"  {effect}: LOFO 평균={all_cos.mean():.4f}  층화 test={stratified_cos:.4f}  격차={stratified_cos-all_cos.mean():+.4f}")

    p29 = {
        effect: {
            "lofo_mean": lofo_results[effect]["_overall"]["lofo_mean_cos"],
            "stratified_mean": lofo_results[effect]["_overall"]["stratified_test_cos_ref"],
            "lofo_less_than_stratified": lofo_results[effect]["_overall"]["lofo_mean_cos"] < lofo_results[effect]["_overall"]["stratified_test_cos_ref"],
        }
        for effect in EFFECT_NAMES
    }
    print("P29 (LOFO < 층화?):", {e: p29[e]["lofo_less_than_stratified"] for e in EFFECT_NAMES})

    # =========================================================================
    # 그림
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
    for ax, effect in zip(axes, EFFECT_NAMES):
        r = reconstruction_results[effect]
        labels = ["기준선\n(아무것도 안함)", "B1 복원\n(상한)", "B2 복원\n(실전)"]
        vals = [r["b2_practical"]["baseline_do_nothing_cos_mean"], r["b1_upper_bound"]["reconstruction_cos_mean"], r["b2_practical"]["reconstruction_cos_mean"]]
        los = [r["b2_practical"]["baseline_do_nothing_cos_ci"][0], r["b1_upper_bound"]["reconstruction_cos_ci"][0], r["b2_practical"]["reconstruction_cos_ci"][0]]
        his = [r["b2_practical"]["baseline_do_nothing_cos_ci"][1], r["b1_upper_bound"]["reconstruction_cos_ci"][1], r["b2_practical"]["reconstruction_cos_ci"][1]]
        x = np.arange(3)
        yerr = np.array([np.clip(np.array(vals) - np.array(los), 0, None), np.clip(np.array(his) - np.array(vals), 0, None)])
        ax.bar(x, vals, yerr=yerr, capsize=3, color=[COLORS["baseline"], "#bcd3f0", COLORS[effect]], zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(effect, fontsize=10)
        ax.set_ylim(0, 1.0)
        style_axis(ax)
    axes[0].set_ylabel("cos(e_dry_hat, e_dry_true)")
    fig.suptitle("과제 C — 복원 검증 (기준선 대비)")
    fig.tight_layout()
    fig.savefig(figures_dir / "reconstruction.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    for ax, effect in zip(axes, EFFECT_NAMES):
        fams = [f for f in families_sorted]
        vals = [lofo_results[effect][f]["cos_mean"] for f in fams]
        x = np.arange(len(fams))
        ax.bar(x, vals, color=COLORS[effect], zorder=3)
        strat = lofo_results[effect]["_overall"]["stratified_test_cos_ref"]
        ax.axhline(strat, color="black", linestyle="--", linewidth=1, label=f"층화 test={strat:.3f}")
        lofo_mean = lofo_results[effect]["_overall"]["lofo_mean_cos"]
        ax.axhline(lofo_mean, color=COLORS["null"], linestyle=":", linewidth=1.2, label=f"LOFO 평균={lofo_mean:.3f}")
        ax.set_xticks(x); ax.set_xticklabels(fams, rotation=60, ha="right", fontsize=7)
        ax.set_title(effect, fontsize=10)
        ax.set_ylim(-0.2, 1.0)
        ax.legend(frameon=False, fontsize=7)
        style_axis(ax)
    axes[0].set_ylabel("held-out cos (안 본 패밀리)")
    fig.suptitle("과제 D — LOFO 진단 (정방향 MLP, 10-fold) vs 층화 결과")
    fig.tight_layout()
    fig.savefig(figures_dir / "lofo_diagnostic.png")
    plt.close(fig)

    # =========================================================================
    # 저장
    # =========================================================================
    def tag_none(x):
        if isinstance(x, dict):
            return {k: tag_none(v) for k, v in x.items()}
        return x

    r8["reverse_b3"] = tag_none(b3_results)
    r8["reconstruction"] = tag_none(reconstruction_results)
    r8["lofo"] = tag_none(lofo_results)
    r8["prereg_checks"]["P28"] = p28
    r8["prereg_checks"]["P29"] = p29
    r8["depends_on_surrogate"] = "none"

    with open(args.results8, "w") as f:
        json.dump(r8, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 2 요약 ===")
    print("\n과제 B3 (크기, B2 모델 재사용):")
    for e in EFFECT_NAMES:
        print(f"  {e:<12} R²={b3_results[e]['mag_r2']:.4f}  상대오차 중앙값={b3_results[e]['mag_rel_err_median']:.4f}")
    print("\n과제 C (복원, B2=실전):")
    for e in EFFECT_NAMES:
        rr = reconstruction_results[e]["b2_practical"]
        print(f"  {e:<12} 복원={rr['reconstruction_cos_mean']:.4f}  기준선={rr['baseline_do_nothing_cos_mean']:.4f}  "
              f"개선={rr['improvement']:+.4f}  기준선 초과(CI 비중첩)={rr['exceeds_baseline']}")
    print("\n과제 D (LOFO vs 층화):")
    for e in EFFECT_NAMES:
        o = lofo_results[e]["_overall"]
        print(f"  {e:<12} LOFO={o['lofo_mean_cos']:.4f}  층화={o['stratified_test_cos_ref']:.4f}  격차={o['gap_stratified_minus_lofo']:+.4f}")
    print(f"\n저장: {args.results8}, {figures_dir/'reconstruction.png'}, {figures_dir/'lofo_diagnostic.png'}")
    print("★ 과제 8(B3·C·D) 완료.")


if __name__ == "__main__":
    main()
