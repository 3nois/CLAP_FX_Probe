# -*- coding: utf-8 -*-
"""3-1 v3 — signed context(EQ gain) 부호 분기 분리 + Bonferroni 보정.

out/prereg/11_phase3.md v3 addendum 확정 방법론.
unsigned 9개는 v2 결과 재사용(재계산 없음). signed 3쌍(focus=cutoff,
context=gain)만 부호 분기별로 재계산 + 부호 대칭 별도 보고.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

base = import_module("11_phase2_render")
CACHE_DIR = base.CACHE_DIR
RESULTS_DIR = base.RESULTS_DIR

SEED = 0
N_BOOT = 2000
N_SOURCE_PAIRS = 5000
N_RANDOM_NULL_PAIRS = 1000
DIM = 512
N_TOTAL_TESTS = 15  # 9 unsigned(v2) + 6 signed 분기(v3)
ALPHA = 0.05
ALPHA_BONF = ALPHA / N_TOTAL_TESTS

SIGNED_PAIRS = ["highshelf_gain_cutoff", "lowshelf_gain_cutoff", "peak_gain_cutoff"]


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def angle_between(a, b):
    c = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def bootstrap_mean_ci(x, pct_lo, pct_hi, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(x)
    boots = np.array([np.mean(x[rng.randint(0, n, n)]) for _ in range(n_boot)])
    return float(np.mean(x)), float(np.percentile(boots, pct_lo)), float(np.percentile(boots, pct_hi))


def random_null_angles(seed=SEED, n_pairs=N_RANDOM_NULL_PAIRS, dim=DIM):
    rng = np.random.RandomState(seed)
    a = rng.normal(size=(n_pairs, dim))
    b = rng.normal(size=(n_pairs, dim))
    a, b = normalize(a), normalize(b)
    return angle_between(a, b)


def cos_rows(a, b):
    return np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def run_signed_branch(pair_name, branch_sign, pct_lo, pct_hi, null_lo_c, null_hi_c):
    """context=gain(axis1), focus=cutoff(axis2). branch_sign: '+' or '-'."""
    d = np.load(CACHE_DIR / f"11_phase3_2d_{pair_name}.npz")
    emb = d["embeddings"]
    g1, g2 = d["grid1"], d["grid2"]  # g1=gain, g2=cutoff
    n_focus = len(g2)
    i_min, i_max = 0, n_focus - 1

    if branch_sign == "+":
        branch_idx = [b for b in range(len(g1)) if g1[b] > 0]
    else:
        branch_idx = [b for b in range(len(g1)) if g1[b] < 0]

    d_A = {}
    for b in branch_idx:
        e_max = emb[:, b, i_max, :]
        e_min = emb[:, b, i_min, :]
        c = cos_rows(e_max, e_min)
        d_A[b] = float((1.0 - c).mean())

    b0 = max(branch_idx, key=lambda b: d_A[b])
    e_max0, e_min0 = emb[:, b0, i_max, :], emb[:, b0, i_min, :]
    v_b0 = normalize(e_max0) - normalize(e_min0)

    curve = []
    for b in branch_idx:
        e_max, e_min = emb[:, b, i_max, :], emb[:, b, i_min, :]
        v_b = normalize(e_max) - normalize(e_min)
        ang = angle_between(v_b0, v_b)
        mean, lo, hi = bootstrap_mean_ci(ang, pct_lo, pct_hi)
        curve.append({"gain": float(g1[b]), "mean_deg": mean, "ci_lo": lo, "ci_hi": hi, "d_A": d_A[b]})

    n_src = v_b0.shape[0]
    rng = np.random.RandomState(SEED)
    idx_i = rng.randint(0, n_src, N_SOURCE_PAIRS)
    idx_j = rng.randint(0, n_src, N_SOURCE_PAIRS)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    ang_source = angle_between(v_b0[idx_i], v_b0[idx_j])
    src_mean, src_lo, src_hi = bootstrap_mean_ci(ang_source, pct_lo, pct_hi)

    non_b0 = [r for r in curve if r["gain"] != float(g1[b0])]
    max_entry = max(non_b0, key=lambda r: r["mean_deg"]) if non_b0 else None

    def distinguishable(lo, hi):
        return hi < null_lo_c or lo > null_hi_c

    context_vs_null = bool(distinguishable(max_entry["ci_lo"], max_entry["ci_hi"])) if max_entry else False
    if max_entry is None:
        verdict = "판정불가"
    elif not context_vs_null:
        verdict = "context 무관(보정 후 널과 구분 안 됨)"
    elif max_entry["ci_hi"] < src_lo:
        verdict = "context 부차적"
    elif max_entry["ci_lo"] > src_hi:
        verdict = "context 우세"
    else:
        verdict = "대등"

    return {
        "pair": pair_name, "branch": branch_sign, "b0_gain": float(g1[b0]),
        "rot_context_max": max_entry, "rot_source": {"mean_deg": src_mean, "ci_lo": src_lo, "ci_hi": src_hi},
        "context_vs_null": context_vs_null, "verdict": verdict, "v_b0": v_b0,
    }


def sign_symmetry(pos_result, neg_result):
    v_pos, v_neg = pos_result["v_b0"], neg_result["v_b0"]
    c = cos_rows(v_pos, v_neg)
    mean, lo, hi = bootstrap_mean_ci(c, 2.5, 97.5)
    return {"cos_mean": mean, "ci": [lo, hi], "angle_deg": float(np.degrees(np.arccos(np.clip(mean, -1, 1))))}


def main():
    null_angles = random_null_angles()
    pct_lo, pct_hi = 100 * ALPHA_BONF / 2, 100 * (1 - ALPHA_BONF / 2)
    null_lo_c, null_hi_c = np.percentile(null_angles, [pct_lo, pct_hi])
    print(f"Bonferroni 보정: alpha={ALPHA_BONF:.6f} (0.05/{N_TOTAL_TESTS}), "
          f"널 대역=[{pct_lo:.3f},{pct_hi:.3f}]백분위=[{null_lo_c:.2f},{null_hi_c:.2f}]도")

    with open(RESULTS_DIR / "11_phase3_rotation_v2_raw.json", encoding="utf-8") as f:
        v2_raw = json.load(f)["results"]
    unsigned = [r for r in v2_raw if not (r["context"] == "gain")]
    print(f"unsigned 재사용 {len(unsigned)}개 (v2 그대로, Bonferroni 재판정만)")

    lines = ["# 3-1 v3 — signed context 부호 분기 분리 + Bonferroni 보정 (2026-08-19)\n"]
    lines.append(f"Bonferroni 보정: 15개 검정(unsigned 9 + signed 분기 6) 기준 alpha={ALPHA_BONF:.6f}, "
                 f"널 대역=[{null_lo_c:.2f}°,{null_hi_c:.2f}°]\n")

    lines.append("## unsigned 9개 — v2 재사용, Bonferroni 재판정\n")
    lines.append("| 쌍 | focus | context | rot_context 최대(95%CI) | rot_source | 보정 후 널과 구분 | 판정(보정) |")
    lines.append("|---|---|---|---|---|---|---|")
    unsigned_final = []
    for r in unsigned:
        if r.get("rot_context_max") is None:
            continue
        me, rs = r["rot_context_max"], r["rot_source"]
        distinguishable = bool(me["ci_hi"] < null_lo_c or me["ci_lo"] > null_hi_c)
        if not distinguishable:
            verdict = "context 무관(보정 후 널과 구분 안 됨)"
        elif me["ci_hi"] < rs["ci_lo"]:
            verdict = "context 부차적"
        elif me["ci_lo"] > rs["ci_hi"]:
            verdict = "context 우세"
        else:
            verdict = "대등"
        unsigned_final.append({**r, "verdict_bonf": verdict, "distinguishable_bonf": distinguishable})
        lines.append(f"| {r['pair']} | {r['focus']} | {r['context']} | "
                     f"{me['mean_deg']:.1f}° [{me['ci_lo']:.1f},{me['ci_hi']:.1f}] | "
                     f"{rs['mean_deg']:.1f}° [{rs['ci_lo']:.1f},{rs['ci_hi']:.1f}] | "
                     f"{'구분됨' if distinguishable else '**구분 안 됨**'} | **{verdict}** |")

    lines.append("\n## signed 3쌍 — 부호 분기 분리 (focus=cutoff, context=gain)\n")
    lines.append("| 쌍 | 분기 | b0(gain) | rot_context 최대(95%CI) | rot_source | 보정 후 널과 구분 | 판정(보정) |")
    lines.append("|---|---|---|---|---|---|---|")
    signed_results = {}
    for pair_name in SIGNED_PAIRS:
        pos = run_signed_branch(pair_name, "+", pct_lo, pct_hi, null_lo_c, null_hi_c)
        neg = run_signed_branch(pair_name, "-", pct_lo, pct_hi, null_lo_c, null_hi_c)
        signed_results[pair_name] = {"pos": pos, "neg": neg}
        for br in [pos, neg]:
            me, rs = br["rot_context_max"], br["rot_source"]
            lines.append(f"| {pair_name} | gain{br['branch']} | {br['b0_gain']:.3g} | "
                         f"{me['mean_deg']:.1f}° [{me['ci_lo']:.1f},{me['ci_hi']:.1f}] | "
                         f"{rs['mean_deg']:.1f}° [{rs['ci_lo']:.1f},{rs['ci_hi']:.1f}] | "
                         f"{'구분됨' if br['context_vs_null'] else '**구분 안 됨**'} | **{br['verdict']}** |")
        print(f"완료: {pair_name} gain+ -> {pos['verdict']}, gain- -> {neg['verdict']}")

    lines.append("\n## 부호 대칭 (별도 지표 — 회전으로 해석하지 않음)\n")
    lines.append("| 쌍 | cos(v_gain+, v_gain-) | 각도 환산 | 해석 |")
    lines.append("|---|---|---|---|")
    for pair_name in SIGNED_PAIRS:
        sym = sign_symmetry(signed_results[pair_name]["pos"], signed_results[pair_name]["neg"])
        interp = "부호 반전 — 부스트/컷을 별개 손잡이로 다뤄야 함" if sym["cos_mean"] < -0.3 else "부분 반전" if sym["cos_mean"] < 0 else "동일 방향"
        lines.append(f"| {pair_name} | {sym['cos_mean']:.3f} [{sym['ci'][0]:.3f},{sym['ci'][1]:.3f}] | "
                     f"{sym['angle_deg']:.1f}° | {interp} |")
        print(f"부호대칭 {pair_name}: cos={sym['cos_mean']:.3f} ({sym['angle_deg']:.1f}°)")

    lines.append("\n## 다중비교 명시\n")
    lines.append(f"15개 검정을 95% 널 대역(보정 전)으로 개별 판정하면 우연 기대 오탐 수 = "
                 f"15×0.05 = 0.75건. v2에서 근소 초과(peak_gain_cutoff cutoff-focus, "
                 f"95.3° vs 널 상한 94.77°)는 이 우연 기대값 범위 안에 있어 보정 전 기준으로도 "
                 f"단정하기 어려웠다. Bonferroni 보정(alpha={ALPHA_BONF:.6f}) 적용 후 재판정한 "
                 f"결과는 위 표에 반영했다.\n")

    lines.append("## 결함 17\n")
    lines.append("> signed context 축(EQ gain)에서 부호 분기를 섞으면 반대 방향 벡터끼리 "
                 "상쇄돼 회전이 ~90°(무작위 널과 구분 안 됨)로 나와 '무관'으로 오판된다 — "
                 "1차 실험 결함 2(부호별 방향 처리 오류)의 재발이다. v2에서 이 문제로 "
                 "오판됐던 3개 EQ 쌍의 cutoff-focus 방향을 부호 분기별로 재계산해 바로잡았다.\n")

    out_path = RESULTS_DIR / "11_phase3_rotation_v3.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n저장: {out_path}")

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "v_b0"}
        return o

    with open(RESULTS_DIR / "11_phase3_rotation_v3_raw.json", "w", encoding="utf-8") as f:
        json.dump({"unsigned_final": unsigned_final, "signed_results": strip(signed_results),
                   "null_band_bonf": [float(null_lo_c), float(null_hi_c)], "alpha_bonf": ALPHA_BONF},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
