# -*- coding: utf-8 -*-
"""Phase 4 — 악기 패밀리 대조 재구축 (결함 B 대응, 사용자 지시 §D).

렌더링 불필요 — out/caches/11_phase2_bypass.npz(악기 판별용, dry)와
out/caches/11_phase2_<axis>.npz(이펙트 레벨 판별용) 캐시만 읽는다.

주 지표: AMI(adjusted_mutual_info_score, 우연 수준 보정) + 불확실성 계수
U = I(X;Y)/H(Y). NMI는 연속성 유지용으로만 병기한다. 무작위 7종 서브샘플링은
1회가 아니라 500회 부트스트랩으로 분포를 낸다. 이펙트 레벨은 5/7/10/25 비닝
민감도를 함께 본다. R²(연속 세타 회귀)를 모든 MI계열 비교에 병기한다.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, adjusted_mutual_info_score, mutual_info_score,
    normalized_mutual_info_score, r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "out" / "caches"
RESULTS_DIR = ROOT / "out" / "results"

SEED = 0
N_SPLITS = 5
TEST_SIZE = 0.3
N_BOOT_7CLASS = 500
BIN_COUNTS = [5, 7, 10, 25]
OLD_REPORTED_NMI_7CLASS = 0.844

MAIN_AXES = [
    "distortion_drive_db", "reverb_wet_level", "reverb_room_size", "reverb_damping", "reverb_width",
    "highshelf_gain", "lowshelf_gain", "peak_gain", "eq_cascade_intensity",
]


def uncertainty_coefficient(y_true, y_pred):
    mi = mutual_info_score(y_true, y_pred)
    _, counts = np.unique(y_true, return_counts=True)
    h_y = scipy_entropy(counts)  # base e, mutual_info_score도 자연로그
    return float(mi / h_y) if h_y > 0 else float("nan")


def classification_metrics(X, y, groups, seed, n_splits=N_SPLITS, test_size=TEST_SIZE):
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes if n_classes > 0 else None
    accs, amis, nmis, us = [], [], [], []
    for train_idx, test_idx in gss.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        amis.append(adjusted_mutual_info_score(y[test_idx], pred))
        nmis.append(normalized_mutual_info_score(y[test_idx], pred))
        us.append(uncertainty_coefficient(y[test_idx], pred))
    return {
        "accuracy": float(np.mean(accs)), "ami": float(np.mean(amis)),
        "nmi": float(np.mean(nmis)), "uncertainty_coef": float(np.mean(us)),
        "chance_level": chance, "n_classes": n_classes, "n_folds_used": len(accs),
    }


def held_out_r2(X, y_continuous, groups, seed, n_splits=N_SPLITS, test_size=TEST_SIZE):
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    r2s = []
    for train_idx, test_idx in gss.split(X, y_continuous, groups):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], y_continuous[train_idx])
        pred = model.predict(X[test_idx])
        r2s.append(r2_score(y_continuous[test_idx], pred))
    return float(np.mean(r2s))


def load_bypass_and_families():
    bd = np.load(CACHE_DIR / "11_phase2_bypass.npz")
    X = bd["embeddings"]
    src_id = bd["src_id"]
    with open(RESULTS_DIR / "11_phase2_sources.json", encoding="utf-8") as f:
        src_meta = json.load(f)
    src_by_id = {s["src_id"]: s for s in src_meta["sources"]}
    families = np.array([src_by_id[i]["family"] for i in src_id])
    return X, families, src_id


def main():
    lines = ["# Phase 4 — 악기 패밀리 대조 재구축 (결함 B)\n"]
    lines.append("데이터: `out/caches/11_phase2_bypass.npz`(dry, 400소스) — 렌더링 없음.\n")

    X, families, src_id = load_bypass_and_families()
    unique_fams = sorted(np.unique(families).tolist())
    n_fam = len(unique_fams)
    lines.append(f"전체 패밀리 {n_fam}개: {unique_fams}\n")

    # ------------------------------------------------------------
    # 1. 전 패밀리 분류 (금지: 무작위 서브샘플 1회)
    # ------------------------------------------------------------
    lines.append("## 1. 전 패밀리(10종) 분류 — 클래스 수 보정 지표\n")
    full = classification_metrics(X, families, src_id, SEED)
    lines.append(f"- accuracy = {full['accuracy']:.4f} (우연수준 {full['chance_level']:.4f})")
    lines.append(f"- **AMI = {full['ami']:.4f}** (주 지표, 우연 수준 보정)")
    lines.append(f"- 불확실성 계수 U = {full['uncertainty_coef']:.4f} (레이블 엔트로피의 {full['uncertainty_coef']*100:.1f}% 회복)")
    lines.append(f"- NMI = {full['nmi']:.4f} (연속성 유지용 병기)\n")

    # ------------------------------------------------------------
    # 2. 무작위 7종 서브샘플 500회 부트스트랩
    # ------------------------------------------------------------
    lines.append("## 2. 무작위 7종 서브샘플 — 500회 부트스트랩 분포\n")
    rng = np.random.RandomState(SEED)
    boot_nmi, boot_ami, boot_u = [], [], []
    for b in range(N_BOOT_7CLASS):
        chosen = rng.choice(unique_fams, size=min(7, n_fam), replace=False)
        mask = np.isin(families, chosen)
        try:
            m = classification_metrics(X[mask], families[mask], src_id[mask], seed=SEED + b)
        except ValueError:
            continue
        boot_nmi.append(m["nmi"]); boot_ami.append(m["ami"]); boot_u.append(m["uncertainty_coef"])
    boot_nmi = np.array(boot_nmi)
    nmi_lo, nmi_hi = np.percentile(boot_nmi, [2.5, 97.5])
    pct_rank = float((boot_nmi < OLD_REPORTED_NMI_7CLASS).mean() * 100)
    lines.append(f"- n_boot 유효 = {len(boot_nmi)}/{N_BOOT_7CLASS}")
    lines.append(f"- NMI 분포: mean={boot_nmi.mean():.4f}, 95% CI=[{nmi_lo:.4f}, {nmi_hi:.4f}]")
    lines.append(f"- AMI 분포: mean={np.mean(boot_ami):.4f}, 95% CI=[{np.percentile(boot_ami,2.5):.4f}, {np.percentile(boot_ami,97.5):.4f}]")
    lines.append(f"- **과거 보고값 NMI=0.844의 분포 내 위치: 백분위 {pct_rank:.1f}%** "
                 f"({'분포 중심부' if 25 < pct_rank < 75 else '분포 가장자리 — 표본 운이었을 가능성'})\n")

    # ------------------------------------------------------------
    # 3. 이펙트 레벨 분류 — 비닝 민감도
    # ------------------------------------------------------------
    lines.append("## 3. 이펙트 레벨 분류 — 축별, 비닝 해상도별\n")
    lines.append("| 축 | 비닝 수 | AMI | U | NMI | R²(연속) |\n|---|---|---|---|---|---|")
    effect_ami_all = []
    for axis_name in MAIN_AXES:
        p = CACHE_DIR / f"11_phase2_{axis_name}.npz"
        if not p.exists():
            lines.append(f"| {axis_name} | — | (캐시 없음, 건너뜀) | | | |")
            continue
        d = np.load(p)
        emb = d["embeddings"]  # (400, 25, 512)
        theta_raw = d["theta_raw"]
        axis_src_id = d["src_id"]
        n_s, n_l, _ = emb.shape
        Xe = emb.reshape(n_s * n_l, 512)
        groups_e = np.repeat(axis_src_id, n_l)
        theta_cont = np.tile((theta_raw - theta_raw.min()) / (theta_raw.max() - theta_raw.min() + 1e-12), n_s)
        r2 = held_out_r2(Xe, theta_cont, groups_e, SEED)
        for B in BIN_COUNTS:
            bin_idx_per_level = np.minimum((np.arange(n_l) * B) // n_l, B - 1)
            y_bin = np.tile(bin_idx_per_level, n_s)
            m = classification_metrics(Xe, y_bin, groups_e, SEED)
            lines.append(f"| {axis_name} | {B} | {m['ami']:.4f} | {m['uncertainty_coef']:.4f} | {m['nmi']:.4f} | {r2:.4f} |")
            if B == 25:
                effect_ami_all.append((axis_name, m["ami"]))

    lines.append("")
    if effect_ami_all:
        lines.append(f"## 4. 악기 vs 이펙트 정보량 비 (구 \"3.4~9.2배\"의 갱신판)\n")
        at_or_below_chance = [(n, a) for n, a in effect_ami_all if a <= 0]
        above_chance = [(n, a) for n, a in effect_ami_all if a > 0]
        if at_or_below_chance:
            lines.append(f"- **AMI≤0(우연 수준 이하, 비율 정의 불가)**: "
                         + ", ".join(f"{n}={a:.4f}" for n, a in at_or_below_chance))
        legacy3 = {n: a for n, a in effect_ami_all if n in ("distortion_drive_db", "reverb_room_size", "highshelf_gain")}
        if legacy3:
            vals = list(legacy3.values())
            lines.append(f"- **핵심 비교(구 3이펙트 대응축 distortion/room_size/highshelf만, B=25)**: "
                         f"악기 AMI={full['ami']:.4f} / 대응축 범위 [{min(vals):.4f}, {max(vals):.4f}] "
                         f"→ **비 [{full['ami']/max(vals):.2f}배, {full['ami']/min(vals):.2f}배]** — 구 3.4~9.2배와 직접 비교 가능")
        near_zero_positive = [(n, a) for n, a in above_chance if a < 0.02 and n not in legacy3]
        if near_zero_positive:
            lines.append(f"- AMI가 0보다 크지만 매우 작은 축(우연 수준 근접, 비율에 넣으면 분모 불안정으로 "
                         f"수백 배까지 치솟아 단일 대표값으로 부적절): "
                         + ", ".join(f"{n}={a:.4f}(비={full['ami']/a:.0f}배)" for n, a in near_zero_positive))
        lines.append("")

    # ------------------------------------------------------------
    # 한계
    # ------------------------------------------------------------
    lines.append("## 한계\n")
    lines.append("이펙트 레벨은 순서형(ordinal), 악기 패밀리는 명목형(nominal)이다. "
                 "AMI/NMI/U는 모두 순서 정보를 버리고 분할표(contingency table)만 보므로 "
                 "이펙트 쪽에 불리하게 작용할 수 있다 — 위 R²(연속 세타 회귀, 순서 보존)를 "
                 "항상 같은 행에 병기했다. MI계열과 R²의 결론이 갈리는 축이 있으면 그 자체를 "
                 "발견으로 보고한다(위 표에서 AMI는 낮은데 R²는 높은 축, 또는 그 반대인 축을 확인할 것).\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "11_phase4_instrument_contrast.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
