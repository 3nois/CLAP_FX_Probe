# -*- coding: utf-8 -*-
"""Q3 확장 그림 — 1,200소스 캐시가 있는 전 주축 15개 × 4구간 = 60조합.

원 Phase 5 는 5축 20조합만 냈다. 이 그림은 사후 확장분(Phase 5b)까지 포함한다.
왼쪽  within(채움) · between(속빈), 구간 4개를 한 행에 겹쳐 표시
오른쪽 gap 과 95% CI. 0선과의 거리가 Q3 의 주장이다.

출력: figures/fig_q3_allaxes.{png,pdf,svg}
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "out" / "results" / "11_phase5b_q3_allaxes.json"
OUT = Path(__file__).resolve().parent

for cand in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(cand).exists():
        try:
            fm.fontManager.addfont(cand)
        except Exception:
            pass
for fam in ["Noto Sans CJK KR", "Noto Sans CJK JP", "Apple SD Gothic Neo", "NanumGothic"]:
    if any(f.name == fam for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
plt.rcParams["axes.unicode_minus"] = False

INTERVALS = ["전범위", "하위1/3", "중위1/3", "상위1/3"]
ICOLOR = {"전범위": "#1F4E79", "하위1/3": "#3E7CB1",
          "중위1/3": "#79A7CE", "상위1/3": "#B4CDE4"}
OFFSET = {"전범위": -0.255, "하위1/3": -0.085, "중위1/3": 0.085, "상위1/3": 0.255}

INK, MUT, RULE, WARN = "#22252A", "#5F6368", "#DDE1E6", "#C0392B"


def main():
    rows = json.loads(RAW.read_text(encoding="utf-8"))["rows"]
    by = {}
    for r in rows:
        by.setdefault(r["axis"], []).append(r)
    order = sorted(by, key=lambda a: np.median([x["gap_mean"] for x in by[a]]))

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.2, 7.6), sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.055})

    ticks, labels = [], []
    for yi, axis in enumerate(order):
        recs = {r["interval"]: r for r in by[axis]}
        is5 = by[axis][0]["is_original_5"]
        for iv in INTERVALS:
            r, y = recs[iv], yi + OFFSET[iv]
            c = ICOLOR[iv]
            axL.plot([r["between_mean"], r["within_mean"]], [y, y],
                     color=c, lw=1.1, alpha=0.55, zorder=2)
            axL.scatter([r["between_mean"]], [y], s=17, facecolor="white",
                        edgecolor=c, linewidth=1.1, zorder=3)
            axL.scatter([r["within_mean"]], [y], s=21, color=c, zorder=4)

            lo, hi = r["gap_ci"]
            axR.plot([lo, hi], [y, y], color=c, lw=1.5, zorder=3)
            axR.scatter([r["gap_mean"]], [y], s=20, color=c, zorder=4)

        ticks.append(yi)
        labels.append(axis + ("  ○" if is5 else ""))
        if yi:
            for a in (axL, axR):
                a.axhline(yi - 0.5, color=RULE, lw=0.6, zorder=0)

    axL.set_yticks(ticks)
    axL.set_yticklabels(labels, fontsize=9.6, color=INK)
    axL.tick_params(axis="y", length=0, pad=6)
    axL.set_ylim(-0.6, len(order) - 0.4)
    axL.invert_yaxis()

    axL.set_xlim(0, 0.48)
    axL.set_xlabel("코사인 유사도 — 속빈 between,  채운 within", fontsize=10.3,
                   color=INK, labelpad=9)
    axR.set_xlim(-0.006, 0.17)
    axR.set_xlabel("gap = within − between   (95% CI)", fontsize=10.3,
                   color=INK, labelpad=9)
    axR.axvline(0, color=WARN, lw=1.3, zorder=2)
    axR.text(0, 1.005, "0", transform=axR.get_xaxis_transform(), fontsize=9.5,
             color=WARN, ha="center", va="bottom")

    for a in (axL, axR):
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(RULE)
        a.grid(axis="x", color=RULE, lw=0.7, alpha=0.75, zorder=0)
        a.set_axisbelow(True)
        a.tick_params(axis="x", colors=MUT, labelsize=9.3)
    axR.tick_params(axis="y", length=0)

    # cascade 강조
    ci = order.index("eq_cascade_intensity")
    for a in (axL, axR):
        a.axhspan(ci - 0.5, ci + 0.5, color="#FBEEEC", zorder=0)
    axR.annotate("결함 15 — 소스 안에서도 손잡이가 일관되지 않음",
                 xy=(0.042, ci), xytext=(0.056, ci),
                 fontsize=9.2, color=WARN, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=WARN, lw=1.1))

    handles = [Line2D([], [], marker="o", ls="", color=ICOLOR[i], markersize=6, label=i)
               for i in INTERVALS]
    handles.append(Line2D([], [], ls="", marker="", label="축 이름의 ○ = 원 5축"))
    axL.legend(handles=handles, loc="lower center", bbox_to_anchor=(1.03, -0.145),
               ncol=5, frameon=False, fontsize=9.6, handletextpad=0.35,
               columnspacing=1.5)

    fig.subplots_adjust(left=0.175, right=0.98, top=0.965, bottom=0.125)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_q3_allaxes.{ext}",
                    dpi=220 if ext == "png" else None, facecolor="white")

    n_pos = sum(1 for r in rows if r["gap_ci"][0] > 0)
    print("저장:", OUT / "fig_q3_allaxes.png")
    print(f"축 {len(order)} × 구간 4 = {len(rows)}조합 · CI 하한 > 0 : {n_pos}/{len(rows)}")


if __name__ == "__main__":
    main()
