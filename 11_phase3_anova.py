# -*- coding: utf-8 -*-
"""3-3. 전조합 ANOVA — out/prereg/11_phase3.md 확정 방법론.

3-D+ 캐시에서 소스별 변위 스칼라 d[s,i,j,k,(l)]를 만들고, 소스 내부에서 N-way
ANOVA(제곱합) 분해: 주효과 + 모든 2차 쌍 + 3차 이상 잔차. 소스 평균 + 부트스트랩 95% CI.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

base = import_module("11_phase2_render")
CACHE_DIR = base.CACHE_DIR
RESULTS_DIR = base.RESULTS_DIR

GRIDS_3DPLUS = ["highshelf_gain_cutoff_q", "lowshelf_gain_cutoff_q", "peak_gain_cutoff_q",
                "reverb_wet_room_damping_width"]
SEED = 0
N_BOOT = 2000


def ref_index_for_axis(axis_name, grid):
    if axis_name == "gain":
        return int(np.argmin(np.abs(grid)))
    return 0


def bootstrap_mean_ci(x, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(x)
    boots = np.array([np.mean(x[rng.randint(0, n, n)]) for _ in range(n_boot)])
    return float(np.mean(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def anova_decompose(D):
    """D: (n_sources, *dims). 반환: ss_total, ss_main[list], ss_inter{(a,b):arr}, ss_resid — 전부 (n_sources,)."""
    n_sources = D.shape[0]
    dims = D.shape[1:]
    k = len(dims)
    axes_range = tuple(range(1, k + 1))
    grand_mean = D.mean(axis=axes_range, keepdims=True)

    main = []
    for ax in range(k):
        other = tuple(a + 1 for a in range(k) if a != ax)
        m = D.mean(axis=other, keepdims=True) - grand_mean
        main.append(np.broadcast_to(m, D.shape))

    inter = {}
    for a, b in combinations(range(k), 2):
        other = tuple(x + 1 for x in range(k) if x not in (a, b))
        m_ab = D.mean(axis=other, keepdims=True)
        it = m_ab - grand_mean - main[a] - main[b]
        inter[(a, b)] = np.broadcast_to(it, D.shape)

    additive = np.broadcast_to(grand_mean, D.shape) + sum(main)
    plus2nd = additive + sum(inter.values())
    residual = D - plus2nd

    ss_total = np.sum((D - grand_mean) ** 2, axis=axes_range)
    ss_main = [np.sum(m ** 2, axis=axes_range) for m in main]
    ss_inter = {p: np.sum(v ** 2, axis=axes_range) for p, v in inter.items()}
    ss_resid = np.sum(residual ** 2, axis=axes_range)
    return ss_total, ss_main, ss_inter, ss_resid


def analyze(name):
    d = np.load(CACHE_DIR / f"11_phase3_3dplus_{name}.npz")
    emb = d["embeddings"]
    axes_names = [str(a) for a in d["axes"]]
    n_axes = len(axes_names)
    grids = [d[f"grid_{i}"] for i in range(n_axes)]

    ref_idx = tuple(ref_index_for_axis(a, g) for a, g in zip(axes_names, grids))

    # 소스별 e_ref, d 계산
    e_ref = emb[(slice(None),) + ref_idx]  # (n_sources, 512)
    shape = emb.shape[1:-1]
    flat_emb = emb.reshape(emb.shape[0], -1, 512)
    e_ref_norm = np.linalg.norm(e_ref, axis=-1, keepdims=True)
    flat_norm = np.linalg.norm(flat_emb, axis=-1)
    dot = np.einsum('sd,scd->sc', e_ref, flat_emb)
    cos = dot / (e_ref_norm * flat_norm + 1e-12)
    D = (1.0 - cos).reshape((emb.shape[0],) + shape)

    ss_total, ss_main, ss_inter, ss_resid = anova_decompose(D)

    results = {"axes": axes_names, "components": []}
    for ax_i, ax_name in enumerate(axes_names):
        frac = ss_main[ax_i] / (ss_total + 1e-12)
        mean, lo, hi = bootstrap_mean_ci(frac)
        results["components"].append({"term": ax_name, "order": 1, "fraction_mean": mean, "ci_lo": lo, "ci_hi": hi})
    for (a, b), ss in ss_inter.items():
        frac = ss / (ss_total + 1e-12)
        mean, lo, hi = bootstrap_mean_ci(frac)
        results["components"].append({"term": f"{axes_names[a]} x {axes_names[b]}", "order": 2,
                                       "fraction_mean": mean, "ci_lo": lo, "ci_hi": hi})
    frac_resid = ss_resid / (ss_total + 1e-12)
    mean, lo, hi = bootstrap_mean_ci(frac_resid)
    results["components"].append({"term": "3차 이상 잔차", "order": 3, "fraction_mean": mean, "ci_lo": lo, "ci_hi": hi})
    results["d_range"] = [float(D.min()), float(D.max())]
    return results


def main():
    lines = ["# 3-3. 전조합 ANOVA — 상호작용 수치 (마스킹 어블레이션 아님, 제곱합 직접 분해)\n"]
    lines.append("소스 내부에서 N-way ANOVA 분해(주효과+2차+3차 이상 잔차), 소스 평균 + "
                 "부트스트랩 95% CI(n_boot=2000, seed=0). 이 표가 논문에 실릴 유일한 상호작용 수치다.\n")

    all_results = {}
    for name in GRIDS_3DPLUS:
        r = analyze(name)
        all_results[name] = r
        lines.append(f"## {name}  (d 범위: [{r['d_range'][0]:.4f}, {r['d_range'][1]:.4f}])\n")
        lines.append("| 성분 | 차수 | 분산 비율 | 95% CI |")
        lines.append("|---|---|---|---|")
        for c in r["components"]:
            order_label = {1: "주효과", 2: "2차", 3: "고차잔차"}[c["order"]]
            lines.append(f"| {c['term']} | {order_label} | {c['fraction_mean']*100:.1f}% | "
                         f"[{c['ci_lo']*100:.1f}%, {c['ci_hi']*100:.1f}%] |")
        lines.append("")
        print(f"완료: {name}")
        for c in r["components"]:
            print(f"  {c['term']:25s} {c['fraction_mean']*100:6.1f}%")

    out_path = RESULTS_DIR / "11_phase3_anova.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n저장: {out_path}")

    with open(RESULTS_DIR / "11_phase3_anova_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
