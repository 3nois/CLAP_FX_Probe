# -*- coding: utf-8 -*-
"""Phase 5-B(Q3) + 5-C(Q4) — out/prereg/11_phase5.md 확정 방법론.

25레벨 Phase 2 격자에서 handle을 구간별로 정의하고(전범위/3분할/인접표본),
20_family_cosine_oat.py·21_handle_predict_phase1.py의 함수를 재사용해
within/between/gap(Q3)과 forward/B1/B2 예측(Q4)을 낸다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

dr = import_module("11_phase2_doseresponse")
fam_mod = import_module("20_family_cosine_oat")
pred_mod = import_module("21_handle_predict_phase1")
p4 = import_module("11_phase4_1200")

RESULTS_DIR = dr.RESULTS_DIR
SEED = 0

AXES = ["distortion_drive_db", "reverb_room_size", "highshelf_gain", "lowshelf_gain", "peak_gain"]
INTERVALS = [
    ("전범위", 0, 24),
    ("하위1/3", 0, 8), ("중위1/3", 8, 16), ("상위1/3", 16, 24),
    ("인접-하", 2, 3), ("인접-중", 12, 13), ("인접-상", 21, 22),
]


def get_family_array_1200():
    with open(RESULTS_DIR / "11_phase2_sources.json", encoding="utf-8") as f:
        base_sources = json.load(f)["sources"]
    with open(RESULTS_DIR / "11_phase2_sources_ext.json", encoding="utf-8") as f:
        ext_sources = json.load(f)["sources"]
    all_sources = sorted(base_sources + ext_sources, key=lambda s: s["src_id"])
    return np.array([s["family"] for s in all_sources])


def run_q3(v, family):
    U = fam_mod.unit(v)
    n = U.shape[0]
    w_dist, b_dist, gap_dist = fam_mod.bootstrap_within_between(U, family, n_boot=300, seed=SEED)
    return {
        "within_mean": float(w_dist.mean()), "within_ci": list(fam_mod.ci95(w_dist)),
        "between_mean": float(b_dist.mean()), "between_ci": list(fam_mod.ci95(b_dist)),
        "gap_mean": float(gap_dist.mean()), "gap_ci": list(fam_mod.ci95(gap_dist)),
        "verdict": "within > between" if fam_mod.ci95(gap_dist)[0] > 0 else "within ≈ between",
    }


def run_q4(emb, idx_a, idx_b, family, src_id, skip_b2=False):
    train_idx, val_idx, test_idx, per_family = pred_mod.stratified_split(family, SEED)

    def eval_condition(X, Y, fam_arr, src_arr):
        return pred_mod.run_all_models(
            X[train_idx], Y[train_idx], fam_arr[train_idx],
            X[val_idx], Y[val_idx],
            X[test_idx], Y[test_idx], fam_arr[test_idx], src_arr[test_idx],
            SEED, max_epochs=300, patience=30,
        )

    e_a, e_b = emb[:, idx_a, :], emb[:, idx_b, :]
    forward = eval_condition(e_a, e_b - e_a, family, src_id)
    b1 = eval_condition(e_b, e_a - e_b, family, src_id)

    b2 = None
    if not skip_b2:
        pool_idx = list(range(idx_a + 1, idx_b + 1))
        X_parts, Y_parts, fam_parts, src_parts, pos_parts = [], [], [], [], []
        n_src = emb.shape[0]
        for pi in pool_idx:
            X_parts.append(emb[:, pi, :])
            Y_parts.append(e_a - emb[:, pi, :])
            fam_parts.append(family)
            src_parts.append(src_id)
            pos_parts.append(np.arange(n_src))
        X_all = np.concatenate(X_parts, axis=0)
        Y_all = np.concatenate(Y_parts, axis=0)
        fam_all = np.concatenate(fam_parts, axis=0)
        src_all = np.concatenate(src_parts, axis=0)
        pos_all = np.concatenate(pos_parts, axis=0)

        def sub(idx):
            mask = np.isin(pos_all, idx)
            return X_all[mask], Y_all[mask], fam_all[mask], src_all[mask]

        Xtr, Ytr, Ftr, _ = sub(train_idx)
        Xva, Yva, _, _ = sub(val_idx)
        Xte, Yte, Fte, Ste = sub(test_idx)
        b2 = pred_mod.run_all_models(Xtr, Ytr, Ftr, Xva, Yva, Xte, Yte, Fte, Ste, SEED, 300, 30)

    return forward, b1, b2


def main():
    family = get_family_array_1200()
    lines_q3 = ["# Phase 5-B (Q3) — 손잡이가 소스마다 다른가\n"]
    lines_q3.append("예측(사전 등록): 구간이 좁아질수록 within이 낮아질 것.\n")
    lines_q3.append("| 축 | 구간 | within | between | gap (95% CI) | 판정 |")
    lines_q3.append("|---|---|---|---|---|---|")

    lines_q4 = ["# Phase 5-C (Q4) — 방향을 예측할 수 있는가\n"]
    lines_q4.append("기준선 = 5-B의 between(같은 구간). B2(mlp)가 이를 넘는지가 핵심.\n")
    lines_q4.append("| 축 | 구간 | 정방향(mlp) | B1(mlp) | B2(mlp) | between 기준선 | B2>기준선? |")
    lines_q4.append("|---|---|---|---|---|---|---|")

    raw = {"q3": {}, "q4": {}}
    for axis_name in AXES:
        emb, theta_raw, src_id = dr.load_concat(axis_name)
        order = np.argsort(src_id)
        emb, src_id = emb[order], src_id[order]
        assert np.array_equal(src_id, np.arange(1200))

        for label, idx_a, idx_b in INTERVALS:
            key = f"{axis_name}::{label}"
            v = emb[:, idx_b, :] - emb[:, idx_a, :]
            q3 = run_q3(v, family)
            raw["q3"][key] = q3
            lines_q3.append(f"| {axis_name} | {label} | {q3['within_mean']:.4f} | {q3['between_mean']:.4f} | "
                            f"{q3['gap_mean']:.4f} [{q3['gap_ci'][0]:.4f},{q3['gap_ci'][1]:.4f}] | {q3['verdict']} |")
            print(f"Q3 완료: {key} -> {q3['verdict']}")

            skip_b2 = (idx_b - idx_a) == 1
            forward, b1, b2 = run_q4(emb, idx_a, idx_b, family, src_id, skip_b2=skip_b2)
            b2_mlp = b2["mlp"]["cos_mean"] if b2 is not None else None
            exceeds = (b2_mlp is not None) and (b2_mlp > q3["between_mean"])
            raw["q4"][key] = {"forward": forward, "b1": b1, "b2": b2, "between_baseline": q3["between_mean"]}
            b2_str = f"{b2_mlp:.4f}" if b2_mlp is not None else "생략(구간폭1)"
            lines_q4.append(f"| {axis_name} | {label} | {forward['mlp']['cos_mean']:.4f} | "
                            f"{b1['mlp']['cos_mean']:.4f} | {b2_str} | {q3['between_mean']:.4f} | "
                            f"{'예' if exceeds else ('생략' if b2_mlp is None else '아니오')} |")
            print(f"Q4 완료: {key}  forward={forward['mlp']['cos_mean']:.4f} b1={b1['mlp']['cos_mean']:.4f} "
                  f"b2={b2_str}  초과={exceeds}")

    with open(RESULTS_DIR / "11_phase5_q3.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines_q3))
    with open(RESULTS_DIR / "11_phase5_q4.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines_q4))
    print(f"\n저장: 11_phase5_q3.md, 11_phase5_q4.md")

    def strip_tensors(o):
        return o

    with open(RESULTS_DIR / "11_phase5_q3q4_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
