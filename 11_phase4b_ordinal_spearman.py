# -*- coding: utf-8 -*-
"""Phase 4b — 이펙트 축의 순서형 지표 병기 (AMI 의 명목형 편향 보정).

AMI 는 25구간을 서로 무관한 25개 범주로 취급하므로, 인접 구간을 맞혀도 점수를
주지 않는다. 이펙트 레벨은 순서형이므로 이 처리는 체계적으로 불리하다.
악기 패밀리는 명목형이라 잃을 순서 구조가 없으므로, "악기 대 이펙트" 배수는
이펙트 쪽에 불리하게 기울어 있다.

Phase 4 와 **완전히 같은 분할·같은 예측기**를 쓰고 지표만 추가한다.
  (1) 분류기(LogisticRegression) 의 예측 구간 vs 실제 구간  → AMI 와 Spearman
      같은 예측을 두 지표로 읽으면 얼마나 달라지는지
  (2) 회귀기(Ridge) 의 연속 예측 vs 실제 θ                   → Spearman 과 R²
      순서 구조를 온전히 쓰는 경우

출력: out/results/11_phase4b_ordinal.md
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_mutual_info_score, r2_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

p4 = import_module("11_phase4_instrument_contrast")

CACHE_DIR, RESULTS_DIR = p4.CACHE_DIR, p4.RESULTS_DIR
SEED, N_SPLITS, TEST_SIZE = p4.SEED, p4.N_SPLITS, p4.TEST_SIZE
B = 25

# Phase 4 §4 의 배수 계산에 쓰인 세 축
LEGACY3 = ["distortion_drive_db", "reverb_room_size", "highshelf_gain"]
INSTRUMENT_AMI = 0.7732   # Phase 4 확정값 (10 패밀리, 1,200 소스)


def load_axis_1200(axis_name):
    """phase2(400) + phase2ext(800) 를 src_id 순으로 합친다."""
    base = np.load(CACHE_DIR / f"11_phase2_{axis_name}.npz")
    ext_p = CACHE_DIR / f"11_phase2ext_{axis_name}.npz"
    emb, sid = base["embeddings"], base["src_id"]
    if ext_p.exists():
        ext = np.load(ext_p)
        emb = np.concatenate([emb, ext["embeddings"]], axis=0)
        sid = np.concatenate([sid, ext["src_id"]])
    order = np.argsort(sid)
    return emb[order].astype(np.float64), sid[order], base["theta_raw"]


def evaluate(axis_name):
    emb, sid, theta_raw = load_axis_1200(axis_name)
    n_s, n_l, dim = emb.shape
    X = emb.reshape(n_s * n_l, dim)
    groups = np.repeat(sid, n_l)

    # Phase 4 와 동일한 정규화·비닝
    theta_norm = (theta_raw - theta_raw.min()) / (theta_raw.max() - theta_raw.min() + 1e-12)
    y_cont = np.tile(theta_norm, n_s)
    bin_per_level = np.minimum((np.arange(n_l) * B) // n_l, B - 1)
    y_bin = np.tile(bin_per_level, n_s)

    gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)
    ami_l, rho_clf_l, rho_reg_l, r2_l = [], [], [], []

    for tr, te in gss.split(X, y_bin, groups):
        if len(np.unique(y_bin[tr])) < 2:
            continue
        # (1) 분류기 — Phase 4 의 AMI 와 동일 예측
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y_bin[tr])
        pred_bin = clf.predict(X[te])
        ami_l.append(adjusted_mutual_info_score(y_bin[te], pred_bin))
        rho_clf_l.append(spearmanr(y_bin[te], pred_bin).statistic)

        # (2) 회귀기 — 순서 구조를 온전히 사용
        reg = Ridge(alpha=1.0).fit(X[tr], y_cont[tr])
        pred_cont = reg.predict(X[te])
        rho_reg_l.append(spearmanr(y_cont[te], pred_cont).statistic)
        r2_l.append(r2_score(y_cont[te], pred_cont))

    return {
        "axis": axis_name, "n_sources": int(n_s), "n_levels": int(n_l),
        "ami": float(np.mean(ami_l)),
        "rho_clf": float(np.mean(rho_clf_l)),
        "rho_reg": float(np.mean(rho_reg_l)),
        "r2": float(np.mean(r2_l)),
        "n_folds": len(ami_l),
    }


def main():
    t0 = time.time()
    rows = []
    for ax in LEGACY3:
        print(f"  {ax} ...", flush=True)
        r = evaluate(ax)
        rows.append(r)
        print(f"    AMI={r['ami']:.4f}  ρ(분류)={r['rho_clf']:.4f}  "
              f"ρ(회귀)={r['rho_reg']:.4f}  R²={r['r2']:.4f}")

    L = ["# Phase 4b — 이펙트 축의 순서형 지표 병기\n",
         "AMI 는 25구간을 무관한 25범주로 취급해 인접 구간 적중에 점수를 주지 않는다.",
         "이펙트 레벨은 순서형이므로 이 처리가 체계적으로 불리하며, 명목형인 악기",
         "패밀리와의 배수 비교를 이펙트 쪽에 불리하게 기울인다.\n",
         f"Phase 4 와 동일한 분할(GroupShuffleSplit {N_SPLITS}겹, test {TEST_SIZE}, seed {SEED})",
         "과 동일한 예측기를 쓰고 지표만 추가했다.\n",
         "## 1. 같은 예측을 두 지표로 읽기\n",
         "분류기(LogisticRegression)의 **동일한 예측 구간**에 대해 AMI 와 Spearman ρ 를 함께 낸다.\n",
         "| 축 | n | AMI | Spearman ρ | 비고 |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['axis']} | {r['n_sources']} | {r['ami']:.4f} | **{r['rho_clf']:.4f}** | "
                 f"같은 예측, 다른 지표 |")

    L += ["", "## 2. 순서 구조를 온전히 쓰는 경우\n",
          "회귀기(Ridge)의 연속 예측에 대한 Spearman ρ 와 R².\n",
          "| 축 | Spearman ρ | R² |", "|---|---|---|"]
    for r in rows:
        L.append(f"| {r['axis']} | **{r['rho_reg']:.4f}** | {r['r2']:.4f} |")

    ami_v = [r["ami"] for r in rows]
    rho_v = [r["rho_clf"] for r in rows]
    L += ["", "## 3. 배수 비교에 미치는 영향\n", "```",
          f"악기 패밀리 AMI            {INSTRUMENT_AMI:.4f}   (명목형 — 잃을 순서 구조 없음)",
          f"이펙트 AMI                 {min(ami_v):.4f} ~ {max(ami_v):.4f}   (순서형 — 구조 폐기됨)",
          f"  → 배수                   {INSTRUMENT_AMI/max(ami_v):.2f} ~ {INSTRUMENT_AMI/min(ami_v):.2f}배",
          "",
          f"이펙트 Spearman ρ (동일 예측)  {min(rho_v):.4f} ~ {max(rho_v):.4f}",
          "  → 같은 예측기가 순서형 지표로는 훨씬 높게 읽힌다",
          "```\n",
          "**판정** — 배수는 AMI 기준값이며, 순서형 편향이 포함된 **상한**으로 보고해야 한다.",
          "악기가 더 강하다는 방향 자체는 유지되나, 배수를 그대로 인용해서는 안 된다.\n",
          "⚠️ Spearman ρ 와 AMI 는 척도가 달라 직접 비교할 수 없다. 위 표는 *같은 예측이",
          "지표에 따라 얼마나 다르게 읽히는지* 를 보이는 것이지 두 값을 등치하는 것이 아니다.\n"]

    out = RESULTS_DIR / "11_phase4b_ordinal.md"
    out.write_text("\n".join(L), encoding="utf-8")
    with open(RESULTS_DIR / "11_phase4b_ordinal.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "n_splits": N_SPLITS, "test_size": TEST_SIZE,
                   "bins": B, "instrument_ami": INSTRUMENT_AMI, "rows": rows}, f,
                  indent=2, ensure_ascii=False)
    print(f"\n저장: {out}   ({time.time()-t0:.1f}초)")


if __name__ == "__main__":
    main()
