"""CLAP FX Probe — 11_phase9_m2m3.py (Phase 9 §6-5: M2 / M3)

캐시 전용, CLAP 재계산 없음.

M3(패밀리 보존) — M1(R0/R1) 각각의 top-1이 질의와 같은 악기 패밀리인 비율.
M2(실사용, 정답 없음) — 라이브러리 = bypass(1200,dry) + wet@18(1200) + wet@24(1200)
  = 3600, dry/wet 라벨. 질의 소스의 leave-source-out은 결함 22 그룹 전체를 제외
  (사용자 지적 반영, 그룹 크기>1인 20개 소스에서만 실질적 차이). R0(query=e_wet)와
  R1(query=e_wet+alpha*v_hat, alpha는 §6-4에서 val로 이미 선택한 값 재사용 —
  M2는 정답이 없어 alpha를 자체적으로 고를 수 없다) 두 팔만 비교한다.
"""
import json
from importlib import import_module

import numpy as np

r0mod = import_module("11_phase9_retrieval")
r1mod = import_module("11_phase9_r1")

AXES, QUERY_LEVELS, SEED = r0mod.AXES, r0mod.QUERY_LEVELS, r0mod.SEED
unit = r0mod.unit
load_axis, load_bypass = r0mod.load_axis, r0mod.load_bypass
load_family_array, load_dup_groups = r0mod.load_family_array, r0mod.load_dup_groups
stratified_split, bootstrap_ci = r0mod.stratified_split, r0mod.bootstrap_ci
paired_bootstrap_diff = r0mod.paired_bootstrap_diff
train_b2, predict_direction = r1mod.train_b2, r1mod.predict_direction

M2_LIB_LEVELS = [18, 24]


def top1_family_match(query_src, q, library, lib_family, query_family):
    sims = unit(q) @ unit(library).T
    top1 = np.argmax(sims, axis=1)
    return lib_family[top1] == query_family[query_src]


def m3_for_m1(axis, emb, bypass, family_arr, test_idx, r0_alpha_v):
    """M1 라이브러리(bypass 1200) 기준 top-1 패밀리 보존율, R0 vs R1."""
    out = {}
    for lvl, e_wet, v_hat, alpha in r0_alpha_v:
        m_r0 = top1_family_match(test_idx, e_wet[test_idx], bypass, family_arr, family_arr)
        q_r1 = unit(e_wet[test_idx] + alpha * v_hat[test_idx])
        m_r1 = top1_family_match(test_idx, q_r1, bypass, family_arr, family_arr)
        lo0, hi0 = bootstrap_ci(m_r0)
        lo1, hi1 = bootstrap_ci(m_r1)
        dlo, dhi = paired_bootstrap_diff(m_r1, m_r0)
        out[lvl] = {
            "R0_family_preserve": {"mean": float(m_r0.mean()), "ci": [lo0, hi0]},
            "R1_family_preserve": {"mean": float(m_r1.mean()), "ci": [lo1, hi1]},
            "R1_minus_R0_ci": [dlo, dhi],
        }
        print(f"  [M3/M1] {axis} lvl={lvl}: R0={m_r0.mean():.4f} {[round(lo0,4),round(hi0,4)]}  "
              f"R1={m_r1.mean():.4f} {[round(lo1,4),round(hi1,4)]}  diff_CI={[round(dlo,4),round(dhi,4)]}")
    return out


def build_m2_library(axis, emb, bypass, family_arr):
    lib = np.concatenate([bypass] + [emb[:, lvl, :] for lvl in M2_LIB_LEVELS], axis=0)
    n = bypass.shape[0]
    lib_src = np.concatenate([np.arange(n) for _ in range(1 + len(M2_LIB_LEVELS))])
    lib_is_dry = np.concatenate([np.ones(n, bool)] + [np.zeros(n, bool) for _ in M2_LIB_LEVELS])
    lib_family = family_arr[lib_src]
    return lib, lib_src, lib_is_dry, lib_family


def m2_recall(query_src, q, lib, lib_src, lib_is_dry, lib_family, family_arr, group_of, ks=(1, 5, 10)):
    sims = unit(q) @ unit(lib).T
    for i, s in enumerate(query_src):
        excl = group_of[s]
        mask = np.isin(lib_src, list(excl))
        sims[i, mask] = -np.inf
    order = np.argsort(-sims, axis=1)
    out = {}
    for k in ks:
        topk = order[:, :k]
        dry_ratio = lib_is_dry[topk].mean(axis=1)
        fam_match_top1 = lib_family[topk[:, 0]] == family_arr[query_src]
        lo, hi = bootstrap_ci(dry_ratio)
        out[k] = {"dry_ratio_mean": float(dry_ratio.mean()), "dry_ratio_ci": [lo, hi],
                   "dry_ratio_per_query": dry_ratio}
        if k == 1:
            out["family_preserve_top1"] = float(fam_match_top1.mean())
            out["family_preserve_top1_arr"] = fam_match_top1
    return out


def main():
    bypass = load_bypass()
    family_arr = load_family_array()
    group_of = load_dup_groups()
    train_idx, val_idx, test_idx = stratified_split(family_arr, seed=SEED)
    r1_prior = json.load(open("out/results/11_phase9_r1.json"))

    results = {"M3_on_M1": {}, "M2": {}}

    for axis in AXES:
        emb, theta = load_axis(axis)
        model = train_b2(axis, emb, bypass, train_idx, val_idx)

        r0_alpha_v = []
        for lvl in QUERY_LEVELS:
            e_wet = emb[:, lvl, :]
            v_hat = unit(predict_direction(model, e_wet))
            alpha = r1_prior[axis][str(lvl)]["R1"]["alpha"]
            r0_alpha_v.append((lvl, e_wet, v_hat, alpha))

        print(f"\n=== {axis}: M3 on M1 (bypass 라이브러리, top-1 패밀리 보존) ===")
        results["M3_on_M1"][axis] = m3_for_m1(axis, emb, bypass, family_arr, test_idx, r0_alpha_v)

        print(f"\n=== {axis}: M2 (실사용, 정답 없음, group-aware leave-source-out) ===")
        lib, lib_src, lib_is_dry, lib_family = build_m2_library(axis, emb, bypass, family_arr)
        results["M2"][axis] = {}
        for lvl, e_wet, v_hat, alpha in r0_alpha_v:
            r0 = m2_recall(test_idx, e_wet[test_idx], lib, lib_src, lib_is_dry, lib_family, family_arr, group_of)
            q_r1 = unit(e_wet[test_idx] + alpha * v_hat[test_idx])
            r1 = m2_recall(test_idx, q_r1, lib, lib_src, lib_is_dry, lib_family, family_arr, group_of)
            dlo, dhi = paired_bootstrap_diff(r1[10]["dry_ratio_per_query"], r0[10]["dry_ratio_per_query"])
            print(f"  lvl={lvl} (alpha={alpha}): R0 dry@10={r0[10]['dry_ratio_mean']:.4f} {r0[10]['dry_ratio_ci']}  "
                  f"R1 dry@10={r1[10]['dry_ratio_mean']:.4f} {r1[10]['dry_ratio_ci']}  diff_CI={[round(dlo,4),round(dhi,4)]}")
            print(f"       family_preserve@1: R0={r0['family_preserve_top1']:.4f}  R1={r1['family_preserve_top1']:.4f}")
            results["M2"][axis][lvl] = {
                "alpha": alpha,
                "R0": {k: {"dry_ratio_mean": v["dry_ratio_mean"], "dry_ratio_ci": v["dry_ratio_ci"]}
                       for k, v in r0.items() if isinstance(k, int)},
                "R1": {k: {"dry_ratio_mean": v["dry_ratio_mean"], "dry_ratio_ci": v["dry_ratio_ci"]}
                       for k, v in r1.items() if isinstance(k, int)},
                "R0_family_preserve_top1": r0["family_preserve_top1"],
                "R1_family_preserve_top1": r1["family_preserve_top1"],
                "dry_ratio_at10_diff_ci": [dlo, dhi],
            }

    with open("out/results/11_phase9_m2m3.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n저장: out/results/11_phase9_m2m3.json")


if __name__ == "__main__":
    main()
