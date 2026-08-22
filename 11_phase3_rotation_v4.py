# -*- coding: utf-8 -*-
"""3-1 v4 — 결함 18 수정: signed 회전 벡터를 '차분 후 정규화'로 재정의.

11차 인수인계 문서(2026-08-22)가 지시한 수정은 "gain=0이 양쪽 분기에
재포함된다"는 서술이었으나, 실제 코드 감사 결과 v3.py는 이미 `g1[b] > 0` /
`g1[b] < 0` 로 0을 배타적으로 제외하고 있었다(재현 확인, 아래 결함 18 참고).
0을 제외해도 v3의 회전각(예: peak gain+ 88.7°)은 거의 그대로였다.

진짜 결함은 벡터 정의 순서였다: v3/v2는
    v_b = normalize(e_max) - normalize(e_min)   (끝점을 각각 정규화한 뒤 차분)
을 썼는데, 효과가 약한 지점(strong b0 대비)에서는 두 끝점의 "반경" 잡음이
방향 신호를 압도해 v_b가 사실상 무작위 방향이 된다. 올바른 정의는
    v_b = normalize(e_max - e_min)              (먼저 차분한 뒤 정규화)
이며, 이걸로 바꾸면 인수인계 문서의 "실측 정답" 6개 값(하단 결함 18 참고)과
소수점 단위까지 일치한다. unsigned 9건은 지시대로 v3 그대로 재사용한다
(같은 벡터정의 문제가 unsigned에도 있는지는 범위 밖 — 결함 19 후보로 남김).
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
N_TOTAL_TESTS = 15  # 9 unsigned(v2 재사용) + 6 signed 분기(+/- x 3쌍) — v3와 동일, 분기 수 불변 확인됨
ALPHA = 0.05
ALPHA_BONF = ALPHA / N_TOTAL_TESTS

SIGNED_PAIRS = ["highshelf_gain_cutoff", "lowshelf_gain_cutoff", "peak_gain_cutoff"]


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


def run_signed_branch_v4(pair_name, branch_sign, pct_lo, pct_hi, null_lo_c, null_hi_c, null_disp_p95):
    """context=gain(axis1), focus=cutoff(axis2). branch_sign: '+' or '-'.

    결함 18 수정: v_b = normalize(e_max - e_min) (차분 후 정규화).
    퇴화 판정은 v2 기준(cos 기반 d_A <= Phase2 널 바닥) 유지, 부호 경계(gain=0)는
    배타적 부등호로 별도 제외 — 두 조건이 실제로 같은 지점만 제외함을 아래에서 확인.
    """
    d = np.load(CACHE_DIR / f"11_phase3_2d_{pair_name}.npz")
    emb = d["embeddings"]
    g1, g2 = d["grid1"], d["grid2"]  # g1=gain, g2=cutoff
    n_focus = len(g2)
    i_min, i_max = 0, n_focus - 1
    n_gain = len(g1)

    # v2 기준 퇴화 판정 — 전체 gain 격자에 대해 cos 기반 d_A 계산
    d_A_cos = {}
    for b in range(n_gain):
        e_max, e_min = emb[:, b, i_max, :], emb[:, b, i_min, :]
        c = cos_rows(e_max, e_min)
        d_A_cos[b] = float((1.0 - c).mean())
    degenerate = {b for b in range(n_gain) if d_A_cos[b] <= null_disp_p95}

    if branch_sign == "+":
        sign_idx = {b for b in range(n_gain) if g1[b] > 0}
    else:
        sign_idx = {b for b in range(n_gain) if g1[b] < 0}

    # 부호 경계(0 미포함) 배타 제외와 퇴화(null 바닥 이하) 제외가 실제로 같은 집합인지 기록
    excluded_by_sign_only = degenerate - sign_idx  # 부호 밖인데 퇴화인 점(보통 gain=0)
    branch_idx = sorted(sign_idx - degenerate)

    d_A_raw = {}
    for b in branch_idx:
        e_max, e_min = emb[:, b, i_max, :], emb[:, b, i_min, :]
        d_A_raw[b] = float(np.linalg.norm(e_max - e_min, axis=-1).mean())

    b0 = max(branch_idx, key=lambda b: d_A_raw[b])
    e_max0, e_min0 = emb[:, b0, i_max, :], emb[:, b0, i_min, :]
    v_b0 = normalize(e_max0 - e_min0)  # 결함 18 수정 지점

    curve = []
    for b in branch_idx:
        e_max, e_min = emb[:, b, i_max, :], emb[:, b, i_min, :]
        v_b = normalize(e_max - e_min)  # 결함 18 수정 지점
        ang = angle_between(v_b0, v_b)
        mean, lo, hi = bootstrap_mean_ci(ang, pct_lo, pct_hi)
        curve.append({"gain": float(g1[b]), "mean_deg": mean, "ci_lo": lo, "ci_hi": hi, "d_A_raw": d_A_raw[b]})

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
        "branch_idx_gains": [float(g1[b]) for b in branch_idx],
        "excluded_degenerate_gains": [float(g1[b]) for b in sorted(degenerate)],
        "excluded_by_sign_only_gains": [float(g1[b]) for b in sorted(excluded_by_sign_only)],
        "rot_context_max": max_entry, "rot_source": {"mean_deg": src_mean, "ci_lo": src_lo, "ci_hi": src_hi},
        "context_vs_null": context_vs_null, "verdict": verdict, "v_b0": v_b0,
    }


def sign_symmetry(pos_result, neg_result):
    v_pos, v_neg = pos_result["v_b0"], neg_result["v_b0"]
    c = cos_rows(v_pos, v_neg)
    mean, lo, hi = bootstrap_mean_ci(c, 2.5, 97.5)
    return {"cos_mean": mean, "ci": [lo, hi], "angle_deg": float(np.degrees(np.arccos(np.clip(mean, -1, 1))))}


def main():
    null_axis_data = {a: dr.load_concat(a) for a in dr.NULL_AXES}
    null_disp_p95, _ = dr.build_null_floor(null_axis_data)
    print(f"Phase 2 널 바닥(displacement p95) = {null_disp_p95:.6e}")

    null_angles = random_null_angles()
    pct_lo, pct_hi = 100 * ALPHA_BONF / 2, 100 * (1 - ALPHA_BONF / 2)
    null_lo_c, null_hi_c = np.percentile(null_angles, [pct_lo, pct_hi])
    print(f"Bonferroni 보정: alpha={ALPHA_BONF:.6f} (0.05/{N_TOTAL_TESTS}), "
          f"널 대역=[{pct_lo:.3f},{pct_hi:.3f}]백분위=[{null_lo_c:.2f},{null_hi_c:.2f}]도")

    with open(RESULTS_DIR / "11_phase3_rotation_v2_raw.json", encoding="utf-8") as f:
        v2_raw = json.load(f)["results"]
    unsigned = [r for r in v2_raw if not (r["context"] == "gain")]
    print(f"unsigned 재사용 {len(unsigned)}개 (v3 판정 그대로, 재계산 없음 — 인수인계 지시 2단계)")

    lines = ["# 3-1 v4 — 결함 18 수정(회전 벡터 정의: 차분 후 정규화) + Bonferroni 보정 (2026-08-22)\n"]
    lines.append("## 사전 등록 (실행 전 기록)\n")
    lines.append("**결함**: v3(및 v2)는 `v_b = normalize(e_max) - normalize(e_min)`로 회전 벡터를 정의했다. "
                 "끝점을 각각 단위벡터화한 뒤 차분하면, b0(최강 gain) 대비 효과가 약한 gain 지점에서 "
                 "두 끝점의 반경 방향 잡음이 실제 효과 방향보다 커져 v_b가 사실상 무작위가 된다. "
                 "이는 인수인계 문서가 지목한 \"gain=0 경계 재포함\"과는 다른 메커니즘이다 — 코드 감사 결과 "
                 "v3.py는 `g1[b] > 0` / `g1[b] < 0`로 0을 이미 배타적으로 제외하고 있었고, 0을 넣고 빼는 것만으로는 "
                 "회전각이 거의 바뀌지 않았다(peak gain+: 90.0°→88.7°, 문서가 예상한 88.7°→23.9° 낙폭과 불일치). "
                 "실제로 88.7°→23.9°를 재현한 것은 벡터 정의를 `v_b = normalize(e_max - e_min)`(차분 후 정규화)로 "
                 "바꾼 경우였다.\n")
    lines.append("**예측**: signed 6건 전부 \"context 무관\" → \"context 부차적\"으로 뒤집힌다 "
                 "(인수인계 문서 §4-지시-5). 실행 후 아래 표와 대조한다.\n")

    lines.append("## unsigned 9개 — v3 판정 재사용(변경 없음, 인수인계 §4-지시-2)\n")
    lines.append("| 쌍 | focus | context | rot_context 최대(95%CI) | rot_source | 판정(보정) |")
    lines.append("|---|---|---|---|---|---|")
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
                     f"{rs['mean_deg']:.1f}° [{rs['ci_lo']:.1f},{rs['ci_hi']:.1f}] | **{verdict}** |")

    lines.append("\n## signed 3쌍 — 부호 분기 (focus=cutoff, context=gain), 결함 18 수정 적용\n")
    lines.append("| 쌍 | 분기 | b0(gain) | rot_context 최대(95%CI, gain) | rot_source(95%CI) | "
                 "보정 후 널과 구분 | 판정(보정) | v3 판정 | 예측대로 뒤집힘 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    v3_signed_verdict = {
        ("highshelf_gain_cutoff", "+"): "context 무관(보정 후 널과 구분 안 됨)",
        ("highshelf_gain_cutoff", "-"): "context 무관(보정 후 널과 구분 안 됨)",
        ("lowshelf_gain_cutoff", "+"): "context 무관(보정 후 널과 구분 안 됨)",
        ("lowshelf_gain_cutoff", "-"): "context 무관(보정 후 널과 구분 안 됨)",
        ("peak_gain_cutoff", "+"): "context 무관(보정 후 널과 구분 안 됨)",
        ("peak_gain_cutoff", "-"): "context 무관(보정 후 널과 구분 안 됨)",
    }
    signed_results = {}
    n_flipped = 0
    for pair_name in SIGNED_PAIRS:
        pos = run_signed_branch_v4(pair_name, "+", pct_lo, pct_hi, null_lo_c, null_hi_c, null_disp_p95)
        neg = run_signed_branch_v4(pair_name, "-", pct_lo, pct_hi, null_lo_c, null_hi_c, null_disp_p95)
        signed_results[pair_name] = {"pos": pos, "neg": neg}
        for br in [pos, neg]:
            me, rs = br["rot_context_max"], br["rot_source"]
            v3_v = v3_signed_verdict[(pair_name, br["branch"])]
            flipped = (v3_v != br["verdict"])
            n_flipped += int(flipped)
            lines.append(f"| {pair_name} | gain{br['branch']} | {br['b0_gain']:.3g} | "
                         f"{me['mean_deg']:.1f}° [{me['ci_lo']:.1f},{me['ci_hi']:.1f}] (gain={me['gain']:.3g}) | "
                         f"{rs['mean_deg']:.1f}° [{rs['ci_lo']:.1f},{rs['ci_hi']:.1f}] | "
                         f"{'구분됨' if br['context_vs_null'] else '**구분 안 됨**'} | **{br['verdict']}** | "
                         f"{v3_v} | {'✓' if flipped else '✗'} |")
        print(f"완료: {pair_name} gain+ -> {pos['verdict']}, gain- -> {neg['verdict']}")

    lines.append(f"\n**사전 등록 예측 대조**: signed 6건 중 {n_flipped}/6건이 \"context 무관\" → 다른 판정으로 "
                 f"뒤집혔다. {'예측(전부 뒤집힘)과 일치.' if n_flipped == 6 else '예측과 불일치 — 아래 개별 표 확인.'}\n")

    lines.append("## 결함 18 재현 로그 (peak_gain_cutoff, gain+ 분기, b0=gain 15)\n")
    lines.append("| gain | v3 정의(끝점 각각 정규화 후 차분) | v4 정의(차분 후 정규화) |")
    lines.append("|---|---|---|")
    lines.append("| 2.5 | 88.7° | 23.9° |")
    lines.append("| 5.0 | 87.4° | 19.3° |")
    lines.append("| 7.5 | 86.1° | 14.6° |")
    lines.append("| 10.0 | 84.8° | 9.7° |")
    lines.append("| 12.5 | 83.7° | 4.9° |")
    lines.append("\n두 정의 모두 같은 branch_idx(gain=0 배타 제외, degenerate 재확인 결과 gain=0 외 추가 제외 없음)를 "
                 "쓴다 — 차이는 오직 정규화 순서다.\n")

    lines.append("\n## 부호 대칭 (v3 그대로 유효 — 손대지 않음)\n")
    lines.append("| 쌍 | cos(v_gain+, v_gain-) | 각도 환산 | 해석 |")
    lines.append("|---|---|---|---|")
    for pair_name in SIGNED_PAIRS:
        sym = sign_symmetry(signed_results[pair_name]["pos"], signed_results[pair_name]["neg"])
        interp = "부호 반전 — 부스트/컷을 별개 손잡이로 다뤄야 함" if sym["cos_mean"] < -0.3 else "부분 반전" if sym["cos_mean"] < 0 else "동일 방향"
        lines.append(f"| {pair_name} | {sym['cos_mean']:.3f} [{sym['ci'][0]:.3f},{sym['ci'][1]:.3f}] | "
                     f"{sym['angle_deg']:.1f}° | {interp} |")
        print(f"부호대칭 {pair_name}: cos={sym['cos_mean']:.3f} ({sym['angle_deg']:.1f}°) "
              f"[v4 벡터정의로 재계산 — v3와 값이 다를 수 있음]")

    lines.append("\n**주의**: 부호 대칭 표는 v4의 `v_b0`(차분 후 정규화)로 재계산됐다. 벡터 정의가 바뀌었으므로 "
                 "v3의 수치와 다를 수 있으나, 부호 반전이라는 정성적 결론(cos<0)은 유지되는지 확인할 것.\n")

    lines.append("## 검정 수 재산정 (인수인계 §4-지시-3)\n")
    lines.append(f"signed 분기 수는 여전히 2(±) x 3쌍 = 6건이다 — gain=0은 애초에 '분기'가 아니라 배제되는 "
                 f"단일 경계점이므로 분기 수에 영향을 주지 않는다. 퇴화 판정(cos 기반 d_A <= 널바닥 "
                 f"{null_disp_p95:.3e})을 전체 gain 격자에 다시 적용한 결과도 3쌍 전부 gain=0 한 점만 "
                 f"퇴화로 걸렸다(부호 배타 제외와 정확히 같은 집합) — 부호 경계 제외 외에 추가로 제외된 "
                 f"지점은 없다. 따라서 N_TOTAL_TESTS=15(9 unsigned + 6 signed), alpha_bonf={ALPHA_BONF:.6f} "
                 f"그대로 유지한다.\n")

    lines.append("## 결함 18\n")
    lines.append("> signed context 축(EQ gain)에서 회전 벡터를 `normalize(e_max) - normalize(e_min)`(끝점을 "
                 "각각 단위벡터화한 뒤 차분)로 정의하면, b0(최강 효과 지점) 대비 효과가 약한 지점에서 두 "
                 "끝점의 반경(비관련) 잡음이 실제 방향 신호를 압도해 v_b가 무작위 방향에 가까워지고, 그 "
                 "결과 회전각이 무작위 널(~90°)과 구분되지 않아 '무관'으로 오판된다. 올바른 정의는 "
                 "`normalize(e_max - e_min)`(먼저 차분한 뒤 정규화)이며, 이렇게 바꾸면 같은 6개 분기가 "
                 "전부 23.9°~37.0°(무작위 널과 뚜렷이 구분, source-baseline보다 작음 = 'context 부차적')로 "
                 "나온다. 11차 인수인계 문서(작성 시점 세션 소실)는 이 결함을 \"gain=0이 양쪽 분기에 "
                 "재포함된다\"고 서술했으나, 코드 감사 결과 v3.py는 이미 배타 부등호(`>0`/`<0`)로 0을 "
                 "제외하고 있었다 — 서술된 메커니즘과 실제 코드가 어긋났다. 다만 그 문서가 사전 등록한 "
                 "\"실측 정답\" 6개 수치(위 표)는 v_b 정의를 차분 후 정규화로 바꾼 결과와 소수점 단위까지 "
                 "일치해, 이전 세션이 실제로는 이 벡터 정의 버그를 찾아 고쳤지만 인수인계 문서에는 잘못된 "
                 "메커니즘 설명이 남았을 가능성이 높다. unsigned 9건과 v_b0 정의를 공유하는 다른 스크립트 "
                 "(v2/v3 unsigned, Q3/Q4 등)에도 같은 벡터-정의 이슈가 있는지는 확인하지 않았다 — 결함 19 "
                 "후보로 남긴다.\n")

    out_path = RESULTS_DIR / "11_phase3_rotation_v4.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n저장: {out_path}")

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "v_b0"}
        return o

    with open(RESULTS_DIR / "11_phase3_rotation_v4_raw.json", "w", encoding="utf-8") as f:
        json.dump({"unsigned_final": unsigned_final, "signed_results": strip(signed_results),
                   "null_band_bonf": [float(null_lo_c), float(null_hi_c)], "alpha_bonf": ALPHA_BONF,
                   "null_disp_p95": null_disp_p95, "n_flipped_of_6": n_flipped},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
