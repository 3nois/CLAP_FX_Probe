# -*- coding: utf-8 -*-
"""Phase 4 — 1,200소스 재실행 (사용자 지시 §추가).

11_phase2_bypass.npz(400) + 11_phase2ext_bypass.npz(800)를 concat해 최종 수치로
삼는다. 400소스판(out/results/11_phase4_instrument_contrast.md)은 예비값으로
남기고 링크만 건다.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

p4 = import_module("11_phase4_instrument_contrast")

CACHE_DIR = p4.CACHE_DIR
RESULTS_DIR = p4.RESULTS_DIR
SEED = p4.SEED
N_BOOT_7CLASS = p4.N_BOOT_7CLASS
BIN_COUNTS = p4.BIN_COUNTS
OLD_REPORTED_NMI_7CLASS = p4.OLD_REPORTED_NMI_7CLASS
MAIN_AXES = p4.MAIN_AXES


def load_bypass_and_families_1200():
    base = np.load(CACHE_DIR / "11_phase2_bypass.npz")
    ext = np.load(CACHE_DIR / "11_phase2ext_bypass.npz")
    X = np.concatenate([base["embeddings"], ext["embeddings"]], axis=0)
    src_id = np.concatenate([base["src_id"], ext["src_id"]])
    order = np.argsort(src_id)
    X, src_id = X[order], src_id[order]

    with open(RESULTS_DIR / "11_phase2_sources.json", encoding="utf-8") as f:
        base_meta = json.load(f)["sources"]
    with open(RESULTS_DIR / "11_phase2_sources_ext.json", encoding="utf-8") as f:
        ext_meta = json.load(f)["sources"]
    src_by_id = {s["src_id"]: s for s in base_meta + ext_meta}
    families = np.array([src_by_id[i]["family"] for i in src_id])
    return X, families, src_id


def main():
    lines = ["# Phase 4 — 악기 패밀리 대조 (1,200소스, 최종)\n"]
    lines.append("데이터: `out/caches/11_phase2_bypass.npz`(400) + `11_phase2ext_bypass.npz`(800) "
                 "concat — 렌더링 없음. 400소스 예비값은 "
                 "[`11_phase4_instrument_contrast.md`](11_phase4_instrument_contrast.md) 참고.\n")

    X, families, src_id = load_bypass_and_families_1200()
    unique_fams = sorted(np.unique(families).tolist())
    n_fam = len(unique_fams)
    lines.append(f"전체 패밀리 {n_fam}개, 소스 {len(src_id)}개: {unique_fams}\n")

    lines.append("## 1. 전 패밀리(10종) 분류 — 클래스 수 보정 지표\n")
    full = p4.classification_metrics(X, families, src_id, SEED)
    lines.append(f"- accuracy = {full['accuracy']:.4f} (우연수준 {full['chance_level']:.4f})")
    lines.append(f"- **AMI = {full['ami']:.4f}** (주 지표)")
    lines.append(f"- 불확실성 계수 U = {full['uncertainty_coef']:.4f} (레이블 엔트로피의 {full['uncertainty_coef']*100:.1f}% 회복)")
    lines.append(f"- NMI = {full['nmi']:.4f}\n")

    lines.append("## 2. 무작위 7종 서브샘플 — 500회 부트스트랩 분포\n")
    rng = np.random.RandomState(SEED)
    boot_nmi, boot_ami = [], []
    for b in range(N_BOOT_7CLASS):
        chosen = rng.choice(unique_fams, size=min(7, n_fam), replace=False)
        mask = np.isin(families, chosen)
        try:
            m = p4.classification_metrics(X[mask], families[mask], src_id[mask], seed=SEED + b)
        except ValueError:
            continue
        boot_nmi.append(m["nmi"]); boot_ami.append(m["ami"])
    boot_nmi = np.array(boot_nmi)
    nmi_lo, nmi_hi = np.percentile(boot_nmi, [2.5, 97.5])
    pct_rank = float((boot_nmi < OLD_REPORTED_NMI_7CLASS).mean() * 100)
    lines.append(f"- n_boot 유효 = {len(boot_nmi)}/{N_BOOT_7CLASS}")
    lines.append(f"- NMI 분포: mean={boot_nmi.mean():.4f}, 95% CI=[{nmi_lo:.4f}, {nmi_hi:.4f}]")
    lines.append(f"- AMI 분포: mean={np.mean(boot_ami):.4f}, 95% CI=[{np.percentile(boot_ami,2.5):.4f}, {np.percentile(boot_ami,97.5):.4f}]")
    lines.append(f"- **과거 보고값 NMI=0.844의 분포 내 위치: 백분위 {pct_rank:.1f}%**\n")

    lines.append("## 3. 이펙트 레벨 분류 — 축별, 비닝 해상도별 (1,200소스, gp6/reverb/gain/cascade)\n")
    lines.append("| 축 | 비닝 수 | AMI | U | NMI | R²(연속) |\n|---|---|---|---|---|---|")
    effect_ami_all = []
    for axis_name in MAIN_AXES:
        base_p = CACHE_DIR / f"11_phase2_{axis_name}.npz"
        ext_p = CACHE_DIR / f"11_phase2ext_{axis_name}.npz"
        if not base_p.exists():
            continue
        bd = np.load(base_p)
        if ext_p.exists():
            ed = np.load(ext_p)
            emb = np.concatenate([bd["embeddings"], ed["embeddings"]], axis=0)
            axis_src_id = np.concatenate([bd["src_id"], ed["src_id"]])
        else:
            emb, axis_src_id = bd["embeddings"], bd["src_id"]
        theta_raw = bd["theta_raw"]
        n_s, n_l, _ = emb.shape
        Xe = emb.reshape(n_s * n_l, 512)
        groups_e = np.repeat(axis_src_id, n_l)
        theta_cont = np.tile((theta_raw - theta_raw.min()) / (theta_raw.max() - theta_raw.min() + 1e-12), n_s)
        r2 = p4.held_out_r2(Xe, theta_cont, groups_e, SEED)
        for B in BIN_COUNTS:
            bin_idx_per_level = np.minimum((np.arange(n_l) * B) // n_l, B - 1)
            y_bin = np.tile(bin_idx_per_level, n_s)
            m = p4.classification_metrics(Xe, y_bin, groups_e, SEED)
            lines.append(f"| {axis_name} | {B} | {m['ami']:.4f} | {m['uncertainty_coef']:.4f} | {m['nmi']:.4f} | {r2:.4f} |")
            if B == 25:
                effect_ami_all.append((axis_name, m["ami"]))
        print(f"완료: {axis_name} (N={n_s})")

    lines.append("")
    legacy3 = {n: a for n, a in effect_ami_all if n in ("distortion_drive_db", "reverb_room_size", "highshelf_gain")}
    if legacy3:
        vals = list(legacy3.values())
        lines.append(f"## 4. 악기 vs 이펙트 정보량 비 (핵심 비교, B=25)\n")
        lines.append(f"악기 AMI={full['ami']:.4f} / 대응축 범위 [{min(vals):.4f}, {max(vals):.4f}] "
                     f"→ **비 [{full['ami']/max(vals):.2f}배, {full['ami']/min(vals):.2f}배]**\n")

    lines.append("## 한계\n")
    lines.append("400소스판과 동일 — 이펙트 레벨은 순서형, 악기 패밀리는 명목형. R² 병기로 대응.\n")

    out_path = RESULTS_DIR / "11_phase4_1200.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {out_path}")
    print(f"AMI(1200)={full['ami']:.4f} vs AMI(400, 예비)=0.6915")


if __name__ == "__main__":
    main()
