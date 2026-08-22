# -*- coding: utf-8 -*-
"""3-1 v5 — 결함 19 수정: unsigned 9건도 v4 벡터 정의로 재산출.

사용자 지시(2026-08-22): "unsigned은 v3 유지"라는 v4의 면제는 결함 18의 원인을
"부호 분기 경계"로 오진한 데서 나온 것이었다. 실제 원인(정규화 순서)은 부호와
무관하므로 unsigned 9건에도 똑같이 적용된다. rot_source도 같은 v_b0 정의를
쓰므로 함께 재산출한다.

방법: v2.run_pair_direction()과 동일한 구조(퇴화 판정 d_A<=null_disp_p95 cos
기반, b0=argmax d_A)를 쓰되, 회전 벡터만
    v_b = normalize(e_max) - normalize(e_min)   (v2/v3, 결함18)
        ->
    v_b = normalize(e_max - e_min)              (v4 수정)
로 바꾼다. signed 6건은 v4 결과를 그대로 재사용(이미 이 정의로 계산됨).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

base = import_module("11_phase2_render")
dr = import_module("11_phase2_doseresponse")
CACHE_DIR = base.CACHE_DIR
RESULTS_DIR = base.RESULTS_DIR

SEED = 0
N_BOOT = 2000
N_SOURCE_PAIRS = 5000
N_RANDOM_NULL_PAIRS = 1000
DIM = 512
N_TOTAL_TESTS = 15  # 9 unsigned(v5 재산출) + 6 signed(v4 재사용) — 개수 불변
ALPHA = 0.05
ALPHA_BONF = ALPHA / N_TOTAL_TESTS

PAIRS_2D = ["reverb_wet_room", "reverb_wet_damping", "reverb_room_damping",
            "highshelf_gain_cutoff", "lowshelf_gain_cutoff", "peak_gain_cutoff"]


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def angle_between(a, b):
    c = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def cos_rows(a, b):
    return np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


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


def get_v_A(emb, focus_is_axis1, i_max, i_min, b_idx):
    if focus_is_axis1:
        e_max = emb[:, i_max, b_idx, :]
        e_min = emb[:, i_min, b_idx, :]
    else:
        e_max = emb[:, b_idx, i_max, :]
        e_min = emb[:, b_idx, i_min, :]
    return e_max, e_min


def run_pair_direction_v5(pair_name, focus_is_axis1, null_disp_p95, pct_lo, pct_hi, null_lo_c, null_hi_c):
    """결함 19 수정: v_b = normalize(e_max - e_min) (v2 get_v_A/degenerate 판정은 그대로)."""
    d = np.load(CACHE_DIR / f"11_phase3_2d_{pair_name}.npz")
    emb = d["embeddings"]
    g1, g2 = d["grid1"], d["grid2"]
    axis1, axis2 = str(d["axis1"]), str(d["axis2"])
    focus_axis = axis1 if focus_is_axis1 else axis2
    context_axis = axis2 if focus_is_axis1 else axis1
    context_grid = g2 if focus_is_axis1 else g1
    n_context = len(context_grid)
    n_focus = len(g1) if focus_is_axis1 else len(g2)
    i_min, i_max = 0, n_focus - 1

    # 퇴화 context 판정 — v2와 동일(cos 기반 d_A, 변경 없음)
    d_A = []
    for b in range(n_context):
        e_max, e_min = get_v_A(emb, focus_is_axis1, i_max, i_min, b)
        d_A.append(float((1.0 - cos_rows(e_max, e_min)).mean()))
    d_A = np.array(d_A)
    degenerate_mask = d_A <= null_disp_p95
    excluded = [{"context_val": float(context_grid[b]), "d_A": float(d_A[b])} for b in range(n_context) if degenerate_mask[b]]
    valid_bs = [b for b in range(n_context) if not degenerate_mask[b]]

    if len(valid_bs) == 0:
        return {"pair": pair_name, "focus": focus_axis, "context": context_axis,
                "excluded": excluded, "verdict": "전 구간 퇴화 — 판정 불가", "b0": None}

    b0 = max(valid_bs, key=lambda b: d_A[b])

    e_max0, e_min0 = get_v_A(emb, focus_is_axis1, i_max, i_min, b0)
    v_b0 = normalize(e_max0 - e_min0)  # 결함 18/19 수정

    rot_context_curve = []
    for b in valid_bs:
        e_max, e_min = get_v_A(emb, focus_is_axis1, i_max, i_min, b)
        v_b = normalize(e_max - e_min)  # 결함 18/19 수정
        ang = angle_between(v_b0, v_b)
        mean, lo, hi = bootstrap_mean_ci(ang, pct_lo, pct_hi)
        rot_context_curve.append({"context_val": float(context_grid[b]), "mean_deg": mean, "ci_lo": lo, "ci_hi": hi, "d_A": float(d_A[b])})

    n_src = v_b0.shape[0]
    rng = np.random.RandomState(SEED)
    idx_i = rng.randint(0, n_src, N_SOURCE_PAIRS)
    idx_j = rng.randint(0, n_src, N_SOURCE_PAIRS)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    ang_source = angle_between(v_b0[idx_i], v_b0[idx_j])
    src_mean, src_lo, src_hi = bootstrap_mean_ci(ang_source, pct_lo, pct_hi)

    non_b0 = [r for r in rot_context_curve if r["context_val"] != float(context_grid[b0])]
    max_entry = max(non_b0, key=lambda r: r["mean_deg"]) if non_b0 else None

    def distinguishable(lo, hi):
        return hi < null_lo_c or lo > null_hi_c

    context_vs_null = bool(distinguishable(max_entry["ci_lo"], max_entry["ci_hi"])) if max_entry else False
    if max_entry is None:
        verdict = "판정불가(유효 context 없음)"
    elif not context_vs_null:
        verdict = "context 무관(보정 후 널과 구분 안 됨)"
    elif max_entry["ci_hi"] < src_lo:
        verdict = "context 부차적"
    elif max_entry["ci_lo"] > src_hi:
        verdict = "context 우세"
    else:
        verdict = "대등"

    return {
        "pair": pair_name, "focus": focus_axis, "context": context_axis, "b0": float(context_grid[b0]),
        "excluded": excluded, "rot_context_max": max_entry,
        "rot_source": {"mean_deg": src_mean, "ci_lo": src_lo, "ci_hi": src_hi},
        "context_vs_null": context_vs_null, "verdict": verdict,
    }


def main():
    null_axis_data = {a: dr.load_concat(a) for a in dr.NULL_AXES}
    null_disp_p95, _ = dr.build_null_floor(null_axis_data)
    print(f"Phase 2 널 바닥(displacement p95) = {null_disp_p95:.6e}")

    null_angles = random_null_angles()
    pct_lo, pct_hi = 100 * ALPHA_BONF / 2, 100 * (1 - ALPHA_BONF / 2)
    null_lo_c, null_hi_c = np.percentile(null_angles, [pct_lo, pct_hi])
    print(f"Bonferroni 보정: alpha={ALPHA_BONF:.6f} (0.05/{N_TOTAL_TESTS}), "
          f"널 대역=[{null_lo_c:.2f},{null_hi_c:.2f}]도")

    with open(RESULTS_DIR / "11_phase3_rotation_v3_raw.json", encoding="utf-8") as f:
        v3_unsigned = {(r["pair"], r["focus"]): r["verdict_bonf"] for r in json.load(f)["unsigned_final"]}

    lines = ["# 3-1 v5 — 결함 19 수정: unsigned 9건도 v4 벡터 정의로 재산출 (2026-08-22)\n"]
    lines.append("## 사전 등록 (실행 전 기록)\n")
    lines.append("**결함 19**: v4는 \"unsigned 9건은 v3 유지\"라고 면제했다. 그 근거는 결함 18의 원인을 "
                 "\"signed 부호 분기 경계\"로 좁게 진단한 데 있었다 — 실제 원인(회전 벡터를 "
                 "`normalize(e_max)-normalize(e_min)`로 정의해 약한 효과 지점에서 반경 잡음이 방향을 "
                 "압도하는 것)은 부호와 무관하며 unsigned 9건도 같은 정의를 쓴다. 잘못된 원인 진단에 "
                 "근거해 수정 범위를 실제 영향 범위보다 좁게 잡은 사례다.\n")
    lines.append("**예측**: unsigned 9건 중 이미 v3에서 \"context 부차적/우세/대등\"으로 (널과 구분되어) "
                 "나온 항목은 각도 값은 바뀌어도 판정이 유지될 가능성이 높다(원래도 방향 신호가 잡음보다 "
                 "컸다는 뜻이므로). 반대로 v3에서 \"context 무관\"으로 나온 항목 중 일부는 결함 18과 같은 "
                 "메커니즘으로 signed-6건처럼 \"무관\" → \"부차적\"로 뒤집힐 수 있다. 어느 쪽이든 실행 후 "
                 "표로 대조한다.\n")

    lines.append("## unsigned 9개 — v5 재산출(결함 19 수정)\n")
    lines.append("| 쌍 | focus | context | rot_context 최대(95%CI, context값) | rot_source(95%CI) | "
                 "판정(v5) | 판정(v3, 참고) | 뒤집힘 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    unsigned_final = []
    n_flipped_unsigned = 0
    for pair_name in PAIRS_2D:
        for focus_is_axis1 in [True, False]:
            r = run_pair_direction_v5(pair_name, focus_is_axis1, null_disp_p95, pct_lo, pct_hi, null_lo_c, null_hi_c)
            if r["context"] == "gain":
                continue  # signed — v4에서 이미 처리
            unsigned_final.append(r)
            v3_v = v3_unsigned.get((r["pair"], r["focus"]), "?")
            flipped = (v3_v != r["verdict"])
            n_flipped_unsigned += int(flipped)
            if r.get("rot_context_max") is None:
                lines.append(f"| {r['pair']} | {r['focus']} | {r['context']} | — | — | {r['verdict']} | {v3_v} | — |")
                continue
            me, rs = r["rot_context_max"], r["rot_source"]
            lines.append(f"| {r['pair']} | {r['focus']} | {r['context']} | "
                         f"{me['mean_deg']:.1f}° [{me['ci_lo']:.1f},{me['ci_hi']:.1f}] (val={me['context_val']:.3g}) | "
                         f"{rs['mean_deg']:.1f}° [{rs['ci_lo']:.1f},{rs['ci_hi']:.1f}] | "
                         f"**{r['verdict']}** | {v3_v} | {'✓' if flipped else '✗'} |")
            print(f"완료: {r['pair']} focus={r['focus']} -> {r['verdict']} (v3: {v3_v})")

    lines.append(f"\n**대조 결과**: unsigned 9건 중 {n_flipped_unsigned}/9건이 v3 대비 판정이 바뀌었다.\n")

    lines.append("## signed 6개 — v4 그대로 재사용(이미 이 벡터 정의로 계산됨, 변경 없음)\n")
    with open(RESULTS_DIR / "11_phase3_rotation_v4_raw.json", encoding="utf-8") as f:
        v4_raw = json.load(f)
    lines.append("| 쌍 | 분기 | rot_context 최대(95%CI) | rot_source | 판정 |")
    lines.append("|---|---|---|---|---|")
    for pair_name, br in v4_raw["signed_results"].items():
        for sign_key in ["pos", "neg"]:
            b = br[sign_key]
            me, rs = b["rot_context_max"], b["rot_source"]
            lines.append(f"| {pair_name} | gain{b['branch']} | "
                         f"{me['mean_deg']:.1f}° [{me['ci_lo']:.1f},{me['ci_hi']:.1f}] | "
                         f"{rs['mean_deg']:.1f}° [{rs['ci_lo']:.1f},{rs['ci_hi']:.1f}] | **{b['verdict']}** |")

    lines.append("\n## 결함 19\n")
    lines.append("> v4는 결함 18(회전 벡터 정의 `normalize(e_max)-normalize(e_min)`가 약한 효과 지점에서 "
                 "잡음에 압도됨)을 signed 6건에만 적용하고 unsigned 9건은 \"부호 문제 없음\"이라며 v3 결과를 "
                 "그대로 유지했다. 이는 결함 18의 원인을 \"부호 분기 경계 처리\"로 오진한 데서 비롯된 조치였다 "
                 "— 실제 원인은 정규화 순서 자체이고 부호 유무와 무관하므로, unsigned 9건도 같은 결함의 영향권에 "
                 "있었다. 잘못된 원인 진단에 근거해 수정 범위를 실제 영향 범위보다 좁게 설정한 사례다. "
                 "1차 결함 1(지표 불일치)과 마찬가지로 '부분적으로만 맞는 진단이 불완전한 수정으로 이어진' "
                 "유형이다.\n")

    out_path = RESULTS_DIR / "11_phase3_rotation_v5.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n저장: {out_path}")

    with open(RESULTS_DIR / "11_phase3_rotation_v5_raw.json", "w", encoding="utf-8") as f:
        json.dump({"unsigned_final": unsigned_final, "signed_reused_from": "11_phase3_rotation_v4_raw.json",
                   "null_band_bonf": [float(null_lo_c), float(null_hi_c)], "alpha_bonf": ALPHA_BONF,
                   "null_disp_p95": null_disp_p95, "n_flipped_unsigned_of_9": n_flipped_unsigned},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
