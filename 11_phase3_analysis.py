# -*- coding: utf-8 -*-
"""Phase 3 분석 — 2-D 히트맵 + 3-D+ 분산분해 (마스킹-어블레이션 방법론 적응판).

round1-4의 마스킹-어블레이션(M0/M_th/M_e/M_the, 대리모델 입력 슬롯 마스킹)을
11차 규칙(대리모델 금지)에 맞게 적응: 대리모델 없이 실측 격자 자체에서 같은
분해를 한다 — 격자 가장자리(한 축만 변화)가 M_th/M_e 역할을, 원점이 M0,
전체 격자가 M_the 역할을 한다.

  d(theta1,theta2) = 1 - cos(e_ref, e(theta1,theta2))   실측
  d_add(theta1,theta2) = d(theta1,ref2) + d(ref1,theta2)  가산 예측(주효과만)
  d_int(theta1,theta2) = d(theta1,theta2) - d_add(theta1,theta2)   상호작용 잔차

분산분해: SS_total = SS_main1 + SS_main2 + SS_int (2-way), N-way는 고차항까지 일반화.
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
GRIDS_3DPLUS = ["highshelf_gain_cutoff_q", "lowshelf_gain_cutoff_q", "peak_gain_cutoff_q",
                "reverb_wet_room_damping_width"]


def cos_rows(a, b):
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return num / den


def ref_index_for_axis(axis_name, grid):
    if axis_name == "gain":
        return int(np.argmin(np.abs(grid)))
    return 0


def analyze_2d(name):
    d = np.load(CACHE_DIR / f"11_phase3_2d_{name}.npz")
    emb = d["embeddings"]  # (1200,13,13,512)
    g1, g2 = d["grid1"], d["grid2"]
    axis1, axis2 = str(d["axis1"]), str(d["axis2"])
    mean_emb = emb.mean(axis=0)  # (13,13,512) 소스 평균

    r1 = ref_index_for_axis(axis1, g1)
    r2 = ref_index_for_axis(axis2, g2)
    e_ref = mean_emb[r1, r2]

    D = 1.0 - np.array([[np.dot(e_ref, mean_emb[i, j]) / (np.linalg.norm(e_ref) * np.linalg.norm(mean_emb[i, j]) + 1e-12)
                          for j in range(len(g2))] for i in range(len(g1))])

    main1 = D[:, r2]  # (13,) axis1만 변화(axis2=ref)
    main2 = D[r1, :]  # (13,) axis2만 변화(axis1=ref)
    D_add = main1[:, None] + main2[None, :]
    D_int = D - D_add

    ss_total = np.sum((D - D.mean()) ** 2)
    ss_add = np.sum((D_add - D_add.mean()) ** 2)
    ss_int = np.sum((D_int - D_int.mean()) ** 2)
    int_fraction = float(ss_int / (ss_total + 1e-12))

    # 히트맵 1: 실측 D
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    im0 = axes[0].imshow(D, origin="lower", aspect="auto", cmap="viridis",
                          extent=[g2.min(), g2.max(), g1.min(), g1.max()])
    axes[0].set_xlabel(axis2); axes[0].set_ylabel(axis1)
    axes[0].set_title(f"{name}\n실측 변위 d(θ1,θ2)")
    plt.colorbar(im0, ax=axes[0])

    vmax = np.abs(D_int).max()
    im1 = axes[1].imshow(D_int, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          extent=[g2.min(), g2.max(), g1.min(), g1.max()])
    axes[1].set_xlabel(axis2); axes[1].set_ylabel(axis1)
    axes[1].set_title(f"상호작용 잔차 d_int (분산기여 {int_fraction*100:.1f}%)")
    plt.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    fig_path = FIG_DIR / f"11_phase3_heatmap_{name}.png"
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"완료(2D): {name}  int_fraction={int_fraction:.4f}  저장={fig_path}")
    return {
        "name": name, "axis1": axis1, "axis2": axis2,
        "d_range": [float(D.min()), float(D.max())],
        "ss_total": float(ss_total), "ss_main_fraction": float(ss_add.size and (ss_total - ss_int) / (ss_total + 1e-12)),
        "ss_int_fraction": int_fraction,
        "d_int_max_abs": float(np.abs(D_int).max()),
        "d_int_max_at": [float(g1[np.unravel_index(np.abs(D_int).argmax(), D_int.shape)[0]]),
                          float(g2[np.unravel_index(np.abs(D_int).argmax(), D_int.shape)[1]])],
        "figure": str(fig_path),
    }


def analyze_3dplus(name):
    d = np.load(CACHE_DIR / f"11_phase3_3dplus_{name}.npz")
    emb = d["embeddings"]
    axes_names = [str(a) for a in d["axes"]]
    n_axes = len(axes_names)
    grids = [d[f"grid_{i}"] for i in range(n_axes)]
    mean_emb = emb.mean(axis=0)  # (5,5,5,[5,]512)

    ref_idx = tuple(ref_index_for_axis(a, g) for a, g in zip(axes_names, grids))
    e_ref = mean_emb[ref_idx]

    shape = mean_emb.shape[:-1]
    flat_emb = mean_emb.reshape(-1, 512)
    D_flat = 1.0 - cos_rows(np.tile(e_ref, (flat_emb.shape[0], 1)), flat_emb)
    D = D_flat.reshape(shape)

    # 축별 주효과 (다른 축은 ref 고정)
    main_effects = []
    for ax_i in range(n_axes):
        idx = list(ref_idx)
        vals = []
        for k in range(shape[ax_i]):
            idx[ax_i] = k
            vals.append(D[tuple(idx)])
        main_effects.append(np.array(vals))

    # 가산 예측(주효과 합, 절편 보정)
    D_add = np.zeros(shape)
    it = np.ndindex(shape)
    for idx in it:
        val = 0.0
        for ax_i in range(n_axes):
            val += main_effects[ax_i][idx[ax_i]]
        D_add[idx] = val
    D_int = D - D_add

    ss_total = np.sum((D - D.mean()) ** 2)
    ss_int = np.sum((D_int - D_int.mean()) ** 2)
    int_fraction = float(ss_int / (ss_total + 1e-12))

    per_axis_ss = []
    for ax_i in range(n_axes):
        me = main_effects[ax_i]
        per_axis_ss.append(float(np.sum((me - me.mean()) ** 2)))

    print(f"완료(3D+): {name}  axes={axes_names}  int_fraction(고차잔차 비중)={int_fraction:.4f}")
    return {
        "name": name, "axes": axes_names, "d_range": [float(D.min()), float(D.max())],
        "ss_total": float(ss_total),
        "per_axis_main_ss": dict(zip(axes_names, per_axis_ss)),
        "higher_order_residual_fraction": int_fraction,
    }


def main():
    lines = ["# Phase 3 분석 — 2-D 히트맵 + N-D 분산분해\n"]
    lines.append("방법론: round1-4 마스킹-어블레이션(대리모델 입력 슬롯 마스킹)을 11차 규칙(대리모델 "
                 "금지)에 맞게 적응 — 대리모델 없이 실측 격자에서 직접 주효과/상호작용을 분해한다. "
                 "기준점(θ_min)에서 한 축만 변화시킨 값이 그 축의 주효과, 두 값의 합이 가산(additive) "
                 "예측, 실측과 가산 예측의 차이가 상호작용 잔차다. Phase 1/2와 동일하게 게이트 축은 "
                 "예상된 강한 상호작용으로 나타나야 하며(예: wet_level=0 근방에서 room_size 주효과가 "
                 "억제됨), 이는 방법론 자체의 타당성 검증(sanity check)으로도 쓴다.\n")

    lines.append("## 2-D 페어 — 히트맵 + 상호작용 분산기여\n")
    lines.append("| 페어 | d 범위 | 주효과 분산기여 | 상호작용 분산기여 | 최대 |d_int| 지점 |")
    lines.append("|---|---|---|---|---|")
    results_2d = {}
    for name in PAIRS_2D:
        r = analyze_2d(name)
        results_2d[name] = r
        lines.append(f"| {name} | [{r['d_range'][0]:.4f}, {r['d_range'][1]:.4f}] | "
                     f"{r['ss_main_fraction']*100:.1f}% | **{r['ss_int_fraction']*100:.1f}%** | "
                     f"({r['axis1']}={r['d_int_max_at'][0]:.3g}, {r['axis2']}={r['d_int_max_at'][1]:.3g}) |")

    lines.append("\n★ **게이트 sanity check**: `reverb_wet_room`·`reverb_wet_damping`은 wet_level=0이 "
                 "다른 축을 무효화하는 것으로 이미 알려진 쌍이다 — 상호작용 분산기여가 "
                 "`reverb_room_damping`(게이트 없음, wet_level=0.3 고정)보다 뚜렷이 높게 나오면 이 "
                 "분해 방법론이 실제로 알려진 상호작용을 잡아낸다는 검증이 된다.\n")

    lines.append("\n## 3-D+ 격자 — 수치전용 분산분해 (그림 없음)\n")
    lines.append("| 격자 | d 범위 | 축별 주효과 분산(SS) | 고차잔차(3차 이상 상호작용) 분산기여 |")
    lines.append("|---|---|---|---|")
    results_3d = {}
    for name in GRIDS_3DPLUS:
        r = analyze_3dplus(name)
        results_3d[name] = r
        ss_str = ", ".join(f"{k}={v:.4f}" for k, v in r["per_axis_main_ss"].items())
        lines.append(f"| {name} | [{r['d_range'][0]:.4f}, {r['d_range'][1]:.4f}] | {ss_str} | "
                     f"**{r['higher_order_residual_fraction']*100:.1f}%** |")

    lines.append("\n(고차잔차 = 실측 - 가산(주효과 합) 예측의 분산기여. 2차 상호작용과 3차 이상 "
                 "상호작용이 섞여 있으나 격자당 5레벨/축이라 이보다 세밀한 분해는 검정력이 부족하다 — "
                 "이 한계를 명시한다.)\n")

    out_path = RESULTS_DIR / "11_phase3_analysis.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n저장: {out_path}")

    with open(RESULTS_DIR / "11_phase3_analysis_raw.json", "w", encoding="utf-8") as f:
        json.dump({"pairs_2d": results_2d, "grids_3dplus": results_3d}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
