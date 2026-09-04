# -*- coding: utf-8 -*-
"""Q1 그림 — R² 의 범위 종속성, 그리고 "배수" 통계의 불안정성.

축별 R² 두 점 (좁은 폭 하위1/3 → 전범위) + 널 바닥 띠.
**절대값으로만 보고한다** — 배수(전범위/좁은폭)는 분모가 0 에 가까우면 폭주해
널 축이 23.6배 · 29.4배를 내므로 결과 지표로 쓰지 않는다(한계 절 참조).

기존 그림(11_phase2_q1_r2_vs_window.png)의 문제를 고친 것:
  · y축 0~1 공유 → 대부분 패널이 바닥에 눌려 판독 불가였음
  · x축 시작점이 축마다 달랐음(gp6 20%, gn6 30~35%) — 이제 명시
  · 제목의 배수가 주 정보였으나 불안정한 통계였음

출력: figures/fig_q1_r2_range.{png,pdf,svg}
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
RES = ROOT / "out" / "results"
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

NULLS = {"null_12k_gain", "null_15k_gain"}
NARROW, INK, MUT, RULE = "#9AA3AD", "#22252A", "#5F6368", "#DDE1E6"
FULL, NULLC, WARN = "#1F5FA0", "#C0392B", "#C0392B"


def load():
    raw = json.loads((RES / "11_phase2_q1_finegrid_raw.json").read_text(encoding="utf-8"))
    rows = []
    for axis, w in raw.items():
        full = w.get("1.0", {}).get("전체범위", {}).get("r2")
        if full is None:
            continue
        # 좁은 폭 — 0.2 를 우선, 없으면 유효한 최소 폭
        narrow, nw = None, None
        for key in sorted(w, key=float):
            v = w[key].get("하위1/3", {}).get("r2")
            if v is not None:
                narrow, nw = v, float(key)
                if nw >= 0.2:
                    break
        if narrow is None:
            continue
        rows.append({"axis": axis, "narrow": narrow, "full": full, "nw": nw,
                     "mult": full / narrow if narrow != 0 else np.nan,
                     "is_null": axis in NULLS})
    return rows


def main():
    rows = load()
    null_ceiling = max(r["full"] for r in rows if r["is_null"])
    rows.sort(key=lambda r: r["full"])

    fig, axA = plt.subplots(figsize=(9.6, 7.4))

    # ── A ────────────────────────────────────────────────
    axA.axvspan(-0.02, null_ceiling, color="#FBEEEC", zorder=0)
    axA.text(null_ceiling, 1.006, f"널 바닥 {null_ceiling:.4f}",
             transform=axA.get_xaxis_transform(), fontsize=8.8, color=WARN,
             ha="center", va="bottom")

    for yi, r in enumerate(rows):
        c = NULLC if r["is_null"] else FULL
        axA.plot([r["narrow"], r["full"]], [yi, yi], color=RULE, lw=2.0,
                 zorder=2, solid_capstyle="round")
        axA.scatter([r["narrow"]], [yi], s=40, facecolor="white",
                    edgecolor=NARROW, lw=1.5, zorder=3)
        axA.scatter([r["full"]], [yi], s=56, color=c, zorder=4)
        axA.text(max(r["full"], r["narrow"]) + 0.017, yi,
                 f"{r['narrow']:.3f} → {r['full']:.3f}", va="center", ha="left",
                 fontsize=8.6, color=c if r["is_null"] else INK)

    labels = [r["axis"] + ("" if r["nw"] == 0.2 else f"  (폭 {r['nw']:.0%})")
              for r in rows]
    axA.set_yticks(range(len(rows)))
    axA.set_yticklabels(labels, fontsize=9.2,
                        color=INK)
    for t, r in zip(axA.get_yticklabels(), rows):
        if r["is_null"]:
            t.set_color(NULLC)
    axA.tick_params(axis="y", length=0, pad=6)
    axA.set_ylim(-0.7, len(rows) - 0.3)
    axA.set_xlim(-0.03, 0.99)
    axA.set_xlabel("held-out $R^2$ — 좁은 폭 하위1/3(속빈) → 전범위(채움)",
                   fontsize=10.3, color=INK, labelpad=9)
    axA.axvline(0, color=RULE, lw=1.0, zorder=1)

    for s in ("top", "right", "left"):
        axA.spines[s].set_visible(False)
    axA.spines["bottom"].set_color(RULE)
    axA.grid(axis="x", color=RULE, lw=0.7, alpha=0.75, zorder=0)
    axA.set_axisbelow(True)
    axA.tick_params(colors=MUT, labelsize=9.2)

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor=NARROW, markeredgewidth=1.5, markersize=6.5,
               label="좁은 폭 (하위1/3)"),
        Line2D([], [], marker="o", ls="", color=FULL, markersize=7, label="전범위"),
        Line2D([], [], marker="o", ls="", color=NULLC, markersize=7,
               label="널 축 — 신호가 없어야 하는 통제")],
        loc="lower center", bbox_to_anchor=(0.5, 0.002), ncol=3, frameon=False,
        fontsize=9.6, handletextpad=0.45, columnspacing=2.0)

    fig.subplots_adjust(left=0.225, right=0.975, top=0.945, bottom=0.115)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_q1_r2_range.{ext}",
                    dpi=220 if ext == "png" else None, facecolor="white")

    print("저장:", OUT / "fig_q1_r2_range.png")
    print(f"축 {len(rows)}개 · 널 바닥(전범위 최대) {null_ceiling:.4f}")
    for r in rows:
        if r["is_null"]:
            print(f"  ★ 널 {r['axis']:18s} {r['narrow']:.4f} → {r['full']:.4f}"
                  f"   (참고용 배수 {r['mult']:.1f} — 보고하지 않음)")
    weak = [r for r in rows if r["narrow"] < 0.02 and not r["is_null"]]
    print(f"분모 R² < 0.02 인 비-널 축: {len(weak)} — "
          + ", ".join(r["axis"] for r in weak))


if __name__ == "__main__":
    main()
