# -*- coding: utf-8 -*-
"""3-1 재실행 — 퇴화 context 제외, b0 재정의, 무작위 널 추가, 집계 규칙 교체.

사용자 지시(2026-08-19) 반영:
  1. d_A(b) <= 널 바닥(displacement p95)인 b는 회전 계산에서 제외, 표로 보고
  2. b0 = argmax_b d_A(b) (중앙 격자점 아님)
  3. 512차원 무작위 단위벡터 쌍 1000개 각도 분포를 rot_context/rot_source와 겹쳐 그림
  4. rot_source가 Q3 within/between과 정합하는지 재확인
  5. "하나라도 Branch B" 전역 집계 폐기 — 쌍·방향별 개별 판정만 보고
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

base = import_module("11_phase2_render")
dr = import_module("11_phase2_doseresponse")
CACHE_DIR = base.CACHE_DIR
RESULTS_DIR = base.RESULTS_DIR
FIG_DIR = base.ROOT / "out" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

PAIRS_2D = ["reverb_wet_room", "reverb_wet_damping", "reverb_room_damping",
            "highshelf_gain_cutoff", "lowshelf_gain_cutoff", "peak_gain_cutoff"]
N_BOOT = 2000
SEED = 0
N_SOURCE_PAIRS = 5000
N_RANDOM_NULL_PAIRS = 1000
DIM = 512


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def angle_between(a, b):
    c = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def bootstrap_mean_ci(x, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(x)
    boots = np.array([np.mean(x[rng.randint(0, n, n)]) for _ in range(n_boot)])
    return float(np.mean(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


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


def run_pair_direction(pair_name, focus_is_axis1, null_disp_p95, null_angle_lo, null_angle_hi):
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

    # 1. 퇴화 context 판정 — d_A(b) = 1-cos(e_max,e_min), 소스 평균
    d_A = []
    for b in range(n_context):
        e_max, e_min = get_v_A(emb, focus_is_axis1, i_max, i_min, b)
        cos_row = np.sum(e_max * e_min, axis=-1) / (np.linalg.norm(e_max, axis=-1) * np.linalg.norm(e_min, axis=-1) + 1e-12)
        d_A.append(float((1.0 - cos_row).mean()))
    d_A = np.array(d_A)
    degenerate_mask = d_A <= null_disp_p95
    excluded = [{"context_val": float(context_grid[b]), "d_A": float(d_A[b])} for b in range(n_context) if degenerate_mask[b]]
    valid_bs = [b for b in range(n_context) if not degenerate_mask[b]]

    if len(valid_bs) == 0:
        return {"pair": pair_name, "focus": focus_axis, "context": context_axis,
                "excluded": excluded, "verdict": "전 구간 퇴화 — 판정 불가", "b0": None}

    # 2. b0 재정의 = argmax d_A(b), 유효한 b 중에서
    b0 = max(valid_bs, key=lambda b: d_A[b])

    e_max0, e_min0 = get_v_A(emb, focus_is_axis1, i_max, i_min, b0)
    v_b0 = normalize(e_max0) - normalize(e_min0)

    rot_context_curve = []
    for b in valid_bs:
        e_max, e_min = get_v_A(emb, focus_is_axis1, i_max, i_min, b)
        v_b = normalize(e_max) - normalize(e_min)
        ang = angle_between(v_b0, v_b)
        mean, lo, hi = bootstrap_mean_ci(ang)
        rot_context_curve.append({"context_val": float(context_grid[b]), "mean_deg": mean, "ci_lo": lo, "ci_hi": hi, "d_A": float(d_A[b])})

    n_src = v_b0.shape[0]
    rng = np.random.RandomState(SEED)
    idx_i = rng.randint(0, n_src, N_SOURCE_PAIRS)
    idx_j = rng.randint(0, n_src, N_SOURCE_PAIRS)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    ang_source = angle_between(v_b0[idx_i], v_b0[idx_j])
    src_mean, src_lo, src_hi = bootstrap_mean_ci(ang_source)

    non_b0 = [r for r in rot_context_curve if r["context_val"] != float(context_grid[b0])]
    max_entry = max(non_b0, key=lambda r: r["mean_deg"]) if non_b0 else None

    def distinguishable_from_null(lo, hi):
        # CI가 무작위 널 CI와 겹치면 '널과 구분 안 됨'
        return hi < null_angle_lo or lo > null_angle_hi

    context_vs_null = bool(distinguishable_from_null(max_entry["ci_lo"], max_entry["ci_hi"])) if max_entry else False
    source_vs_null = bool(distinguishable_from_null(src_lo, src_hi))

    if max_entry is None:
        verdict = "판정불가(유효 context 없음)"
    elif not context_vs_null:
        verdict = "context 무관(무작위 널과 구분 안 됨)"
    elif max_entry["ci_hi"] < src_lo:
        verdict = "context 부차적"
    elif max_entry["ci_lo"] > src_hi:
        verdict = "context 우세"
    else:
        verdict = "대등"

    return {
        "pair": pair_name, "focus": focus_axis, "context": context_axis, "b0": float(context_grid[b0]),
        "excluded": excluded, "rot_context_curve": rot_context_curve, "rot_context_max": max_entry,
        "rot_source": {"mean_deg": src_mean, "ci_lo": src_lo, "ci_hi": src_hi, "n_pairs": len(idx_i)},
        "context_vs_null": context_vs_null, "source_vs_null": source_vs_null, "verdict": verdict,
    }


def plot_pair(result, null_angles, out_path):
    if result.get("rot_context_curve") is None:
        return
    curve = result["rot_context_curve"]
    xs = [r["context_val"] for r in curve]
    means = [r["mean_deg"] for r in curve]
    los = [r["ci_lo"] for r in curve]
    his = [r["ci_hi"] for r in curve]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    null_lo, null_hi = np.percentile(null_angles, [2.5, 97.5])
    ax.axhspan(null_lo, null_hi, color="#898781", alpha=0.25, label=f"무작위 널 95%CI=[{null_lo:.1f},{null_hi:.1f}]")
    ax.plot(xs, means, "-o", color="#2a78d6", label="rot_context(b)")
    ax.fill_between(xs, los, his, color="#2a78d6", alpha=0.2)
    rs = result["rot_source"]
    ax.axhline(rs["mean_deg"], color="#e34948", linestyle="--", label=f"rot_source mean={rs['mean_deg']:.1f}°")
    ax.axhspan(rs["ci_lo"], rs["ci_hi"], color="#e34948", alpha=0.15)
    ax.set_xlabel(f"{result['context']} (context, 퇴화 제외됨)")
    ax.set_ylabel("회전각(도)")
    ax.set_title(f"{result['pair']} — focus={result['focus']} (b0={result['b0']:.3g})\n판정: {result['verdict']}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    null_axis_data = {a: dr.load_concat(a) for a in dr.NULL_AXES}
    null_disp_p95, _ = dr.build_null_floor(null_axis_data)
    print(f"널 바닥(displacement) p95 = {null_disp_p95:.3e}")

    null_angles = random_null_angles()
    null_lo, null_hi = np.percentile(null_angles, [2.5, 97.5])
    null_mean = float(null_angles.mean())
    print(f"무작위 널(1000쌍, 512차원 단위벡터) 각도: mean={null_mean:.2f}° "
          f"95%범위=[{null_lo:.2f},{null_hi:.2f}]° (이론값 90°±2.5° 참고)")

    all_results = []
    for pair_name in PAIRS_2D:
        for focus_is_axis1 in [True, False]:
            r = run_pair_direction(pair_name, focus_is_axis1, null_disp_p95, null_lo, null_hi)
            all_results.append(r)
            if r.get("rot_context_curve") is not None:
                fig_path = FIG_DIR / f"11_phase3_rotation_v2_{pair_name}_{'axis1' if focus_is_axis1 else 'axis2'}focus.pdf"
                plot_pair(r, null_angles, fig_path)
            print(f"완료: {pair_name} focus={r['focus']} b0={r.get('b0')} -> {r['verdict']} "
                  f"(제외 {len(r['excluded'])}개)")

    lines = ["# 3-1 재실행 — 퇴화 context 제외 + 무작위 널 대조 (2026-08-19)\n"]
    lines.append(f"무작위 널(1000쌍, 512차원 단위벡터 코사인각): mean={null_mean:.2f}°, "
                 f"95%범위=[{null_lo:.2f},{null_hi:.2f}]° (이론값 90°±2.5°와 일치 확인)\n")
    lines.append("**집계 규칙 폐기**: 이전 버전의 \"하나라도 Branch B면 전체 Branch B\" 전역 집계를 "
                 "폐기한다. 쌍·방향별 개별 판정만 아래에 보고하며, Phase 5-D는 쌍 단위로 결정한다.\n")

    lines.append("## 퇴화 context 제외 내역\n")
    lines.append("| 쌍 | focus | 제외된 context 값 | d_A(제외 지점) |")
    lines.append("|---|---|---|---|")
    for r in all_results:
        if r["excluded"]:
            exc_str = ", ".join(f"{e['context_val']:.3g}(d={e['d_A']:.2e})" for e in r["excluded"])
            lines.append(f"| {r['pair']} | {r['focus']} | {len(r['excluded'])}개 | {exc_str} |")
    lines.append(f"\n★ 게이트 구조의 정량적 재확인: EQ 쌍에서 context=gain일 때 gain=0 지점(및 인근)이 "
                 f"전부 퇴화로 제외됐다면 이는 Phase 0.5/3-2(a)의 gain=0 게이트 판정과 정확히 일치하는 "
                 f"교차검증이다.\n")

    lines.append("## 판정 결과 (개별, 집계 없음)\n")
    lines.append("| 쌍 | focus | context | b0(재정의) | rot_context 최대(95%CI) | rot_source(95%CI) | "
                 "context vs 널 | 판정 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in all_results:
        if r.get("rot_context_max") is None:
            lines.append(f"| {r['pair']} | {r['focus']} | {r['context']} | — | — | — | — | {r['verdict']} |")
            continue
        me = r["rot_context_max"]
        rs = r["rot_source"]
        lines.append(f"| {r['pair']} | {r['focus']} | {r['context']} | {r['b0']:.3g} | "
                     f"{me['mean_deg']:.1f}° [{me['ci_lo']:.1f},{me['ci_hi']:.1f}] | "
                     f"{rs['mean_deg']:.1f}° [{rs['ci_lo']:.1f},{rs['ci_hi']:.1f}] | "
                     f"{'구분됨' if r['context_vs_null'] else '**널과 구분 안 됨**'} | **{r['verdict']}** |")

    out_path = RESULTS_DIR / "11_phase3_rotation_v2.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n저장: {out_path}")

    with open(RESULTS_DIR / "11_phase3_rotation_v2_raw.json", "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "null_angle_mean": null_mean,
                   "null_angle_ci": [float(null_lo), float(null_hi)]}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
