# -*- coding: utf-8 -*-
"""3-1. 손잡이 회전 지도 — out/prereg/11_phase3.md 확정 방법론.

6쌍 x 2방향 = 12검정. rot_context(context 바뀔 때 손잡이 회전) vs rot_source(소스
바뀔 때 손잡이 회전)를 각도로 비교한다.
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
B0_IDX = 6  # 13레벨 중앙


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


def get_v_A(emb, g1, g2, focus, i_max, i_min, b_idx, focus_is_axis1):
    """v_A(b) : focus 축 최대-최소, context=b_idx 고정. shape (n_src, 512)."""
    if focus_is_axis1:
        e_max = emb[:, i_max, b_idx, :]
        e_min = emb[:, i_min, b_idx, :]
    else:
        e_max = emb[:, b_idx, i_max, :]
        e_min = emb[:, b_idx, i_min, :]
    return normalize(e_max) - normalize(e_min)


def run_pair_direction(pair_name, focus_is_axis1):
    d = np.load(CACHE_DIR / f"11_phase3_2d_{pair_name}.npz")
    emb = d["embeddings"]
    g1, g2 = d["grid1"], d["grid2"]
    axis1, axis2 = str(d["axis1"]), str(d["axis2"])
    focus_axis = axis1 if focus_is_axis1 else axis2
    context_axis = axis2 if focus_is_axis1 else axis1
    context_grid = g2 if focus_is_axis1 else g1
    n_context = len(context_grid)
    n_focus = len(g1) if focus_is_axis1 else len(g2)
    i_max, i_min = n_focus - 1, 0

    v_b0 = get_v_A(emb, g1, g2, focus_axis, i_max, i_min, B0_IDX, focus_is_axis1)  # (n_src,512)

    rot_context_curve = []
    for b in range(n_context):
        v_b = get_v_A(emb, g1, g2, focus_axis, i_max, i_min, b, focus_is_axis1)
        ang = angle_between(v_b0, v_b)  # (n_src,)
        mean, lo, hi = bootstrap_mean_ci(ang)
        rot_context_curve.append({"context_val": float(context_grid[b]), "mean_deg": mean, "ci_lo": lo, "ci_hi": hi})

    # rot_source: v_b0 소스간 비교 (무작위 소스쌍 5000개, 자기 제외, 복원추출)
    n_src = v_b0.shape[0]
    rng = np.random.RandomState(SEED)
    idx_i = rng.randint(0, n_src, N_SOURCE_PAIRS)
    idx_j = rng.randint(0, n_src, N_SOURCE_PAIRS)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    ang_source = angle_between(v_b0[idx_i], v_b0[idx_j])
    src_mean, src_lo, src_hi = bootstrap_mean_ci(ang_source)

    # 요약: rot_context 최댓값(중앙 제외) 지점의 CI를 판정에 사용
    non_b0 = [r for i, r in enumerate(rot_context_curve) if i != B0_IDX]
    max_entry = max(non_b0, key=lambda r: r["mean_deg"]) if non_b0 else None

    if max_entry is None:
        verdict = "판정불가"
    elif max_entry["ci_hi"] < src_lo:
        verdict = "context 부차적 (Branch A)"
    elif max_entry["ci_lo"] > src_hi:
        verdict = "context 우세 (Branch B)"
    else:
        verdict = "대등 (Branch B)"

    return {
        "pair": pair_name, "focus": focus_axis, "context": context_axis,
        "rot_context_curve": rot_context_curve,
        "rot_context_max": max_entry,
        "rot_source": {"mean_deg": src_mean, "ci_lo": src_lo, "ci_hi": src_hi, "n_pairs": len(idx_i)},
        "verdict": verdict,
    }


def plot_pair(result, out_path):
    curve = result["rot_context_curve"]
    xs = [r["context_val"] for r in curve]
    means = [r["mean_deg"] for r in curve]
    los = [r["ci_lo"] for r in curve]
    his = [r["ci_hi"] for r in curve]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(xs, means, "-o", color="#2a78d6", label="rot_context(b)")
    ax.fill_between(xs, los, his, color="#2a78d6", alpha=0.2)
    rs = result["rot_source"]
    ax.axhline(rs["mean_deg"], color="#e34948", linestyle="--", label=f"rot_source mean={rs['mean_deg']:.1f}°")
    ax.axhspan(rs["ci_lo"], rs["ci_hi"], color="#e34948", alpha=0.15)
    ax.set_xlabel(f"{result['context']} (context)")
    ax.set_ylabel("회전각(도)")
    ax.set_title(f"{result['pair']} — focus={result['focus']}\n판정: {result['verdict']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    all_results = []
    for pair_name in PAIRS_2D:
        for focus_is_axis1 in [True, False]:
            r = run_pair_direction(pair_name, focus_is_axis1)
            all_results.append(r)
            fig_path = FIG_DIR / f"11_phase3_rotation_{pair_name}_{'axis1' if focus_is_axis1 else 'axis2'}focus.pdf"
            plot_pair(r, fig_path)
            print(f"완료: {pair_name} focus={r['focus']} -> {r['verdict']}")

    branch_b_count = sum(1 for r in all_results if "Branch B" in r["verdict"])
    overall_branch = "Branch B" if branch_b_count > 0 else "Branch A"

    lines = ["# 3-1. 손잡이 회전 지도 — 결과\n"]
    lines.append(f"사전 등록: `out/prereg/11_phase3.md`. 12개 검정(6쌍×2방향).\n")
    lines.append("| 쌍 | focus | context | rot_context 최대(deg, 95%CI) | 위치 | rot_source(deg, 95%CI) | 판정 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in all_results:
        me = r["rot_context_max"]
        rs = r["rot_source"]
        lines.append(f"| {r['pair']} | {r['focus']} | {r['context']} | "
                     f"{me['mean_deg']:.1f}° [{me['ci_lo']:.1f},{me['ci_hi']:.1f}] | "
                     f"{r['context']}={me['context_val']:.3g} | "
                     f"{rs['mean_deg']:.1f}° [{rs['ci_lo']:.1f},{rs['ci_hi']:.1f}] | {r['verdict']} |")

    lines.append(f"\n**12개 중 Branch B 판정: {branch_b_count}개**\n")
    lines.append(f"## 전체 판정: **{overall_branch}** (사전 등록 집계 규칙: 하나라도 Branch B면 전체 Branch B)\n")

    out_path = RESULTS_DIR / "11_phase3_rotation.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n저장: {out_path}")
    print(f"전체 판정: {overall_branch} (Branch B {branch_b_count}/12)")

    with open(RESULTS_DIR / "11_phase3_rotation_raw.json", "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "overall_branch": overall_branch}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
