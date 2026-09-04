# -*- coding: utf-8 -*-
"""Q3 그림 — within / between 막대와 gap 의 95% CI, 20개 조합 전부.

왼쪽  within · between 을 나란히 놓은 막대
오른쪽 gap = within − between 과 소스 단위 부트스트랩 95% CI

Q3 의 주장은 "gap 의 CI 가 0 을 배제한다"이므로, CI 와 0선의 관계가 보여야 한다.

출력: figures/fig_q3_within_between.{png,pdf,svg}
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "out" / "results" / "11_phase5_q3q4_raw.json"
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

AXES = [
    ("distortion_drive_db", "distortion\ndrive_db"),
    ("reverb_room_size", "reverb\nroom_size"),
    ("highshelf_gain", "highshelf\ngain"),
    ("lowshelf_gain", "lowshelf\ngain"),
    ("peak_gain", "peak\ngain"),
]
INTERVALS = ["전범위", "상위1/3", "중위1/3", "하위1/3"]

INK, MUT = "#22252A", "#5F6368"
WITHIN, BETWEEN, GAPC, RULE = "#2F6FAF", "#C3CBD4", "#3F8F6B", "#DDE1E6"


def load():
    q3 = json.loads(RAW.read_text(encoding="utf-8"))["q3"]
    rows = []
    for ax_key, ax_label in AXES:
        for iv in INTERVALS:
            r = q3[f"{ax_key}::{iv}"]
            rows.append({
                "axis": ax_label, "interval": iv,
                "within": r["within_mean"], "between": r["between_mean"],
                "gap": r["gap_mean"], "ci": r["gap_ci"],
            })
    return rows


def main():
    rows = load()
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.4, 8.4), sharey=True,
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.06})

    h = 0.36
    y, ticks, labels, spans = 0.0, [], [], []
    for _, ax_label in AXES:
        start = y
        for iv in INTERVALS:
            r = next(x for x in rows if x["axis"] == ax_label and x["interval"] == iv)

            axL.barh(y - h / 2, r["within"], height=h, color=WITHIN, zorder=3)
            axL.barh(y + h / 2, r["between"], height=h, color=BETWEEN, zorder=3)
            axL.text(r["within"] + 0.008, y - h / 2, f"{r['within']:.3f}",
                     va="center", ha="left", fontsize=8.6, color=INK)
            axL.text(r["between"] + 0.008, y + h / 2, f"{r['between']:.3f}",
                     va="center", ha="left", fontsize=8.6, color=MUT)

            lo, hi = r["ci"]
            axR.plot([lo, hi], [y, y], color=GAPC, lw=2.0, zorder=3,
                     solid_capstyle="butt")
            for xv in (lo, hi):
                axR.plot([xv, xv], [y - 0.17, y + 0.17], color=GAPC, lw=1.6, zorder=3)
            axR.scatter([r["gap"]], [y], s=34, color=GAPC, zorder=4)
            axR.text(hi + 0.004, y, f"{r['gap']:.3f}", va="center", ha="left",
                     fontsize=8.8, color=INK)

            ticks.append(y); labels.append(iv); y += 1
        spans.append((ax_label, start, y - 1))
        y += 0.9

    for ax_label, y0, y1 in spans:
        axL.text(-0.20, (y0 + y1) / 2, ax_label, transform=axL.get_yaxis_transform(),
                 ha="center", va="center", fontsize=10.5, color=INK,
                 fontweight="bold", linespacing=1.35)
        axL.plot([-0.085, -0.085], [y0 - 0.45, y1 + 0.45],
                 transform=axL.get_yaxis_transform(), color=RULE, lw=1.4,
                 clip_on=False, zorder=0)

    axL.set_yticks(ticks); axL.set_yticklabels(labels, fontsize=9.8, color=MUT)
    axL.tick_params(axis="y", length=0, pad=4)
    axL.set_ylim(-0.9, y - 0.5); axL.invert_yaxis()

    axL.set_xlim(0, 0.52)
    axL.set_xlabel("코사인 유사도", fontsize=10.5, color=INK, labelpad=9)
    axR.set_xlim(0, 0.185)
    axR.set_xlabel("gap = within − between   (95% CI)", fontsize=10.5,
                   color=INK, labelpad=9)
    axR.axvline(0, color="#C0392B", lw=1.3, zorder=2)
    axR.text(0, 1.006, "0", transform=axR.get_xaxis_transform(), fontsize=9.5,
             color="#C0392B", ha="center", va="bottom")

    for a in (axL, axR):
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(RULE)
        a.grid(axis="x", color=RULE, lw=0.7, alpha=0.7, zorder=0)
        a.set_axisbelow(True)
        a.tick_params(axis="x", colors=MUT, labelsize=9.5)
    axR.tick_params(axis="y", length=0)

    from matplotlib.patches import Patch
    axL.legend(handles=[Patch(facecolor=WITHIN, label="within — 같은 소스 안"),
                        Patch(facecolor=BETWEEN, label="between — 다른 소스끼리")],
               loc="lower center", bbox_to_anchor=(0.5, -0.105), ncol=2,
               frameon=False, fontsize=10, handletextpad=0.6, columnspacing=1.8)

    fig.subplots_adjust(left=0.185, right=0.975, top=0.965, bottom=0.105)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_q3_within_between.{ext}",
                    dpi=220 if ext == "png" else None, facecolor="white")

    n_pos = sum(1 for r in rows if r["ci"][0] > 0)
    print("저장:", OUT / "fig_q3_within_between.png")
    print(f"조합 {len(rows)}개 · gap CI 하한 > 0 인 조합 {n_pos}/{len(rows)}")
    g = [r["gap"] for r in rows]
    print(f"gap 범위 {min(g):.4f} ~ {max(g):.4f}")


if __name__ == "__main__":
    main()
