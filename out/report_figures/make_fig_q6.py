# -*- coding: utf-8 -*-
"""Q6 그림 — 검색 경로 결과.

A  M1 회수율 R@1        R0 / R3 / R1 / Ror 네 팔.
   ★ §8 판정은 R@10 을 쓰지만 R@10 은 R1 에서 0.96~1.00 으로 포화해 팔이 구분되지
     않는다. 그림은 분별이 되는 R@1 로 그리고, R@10 판정 결과를 주석으로 병기한다.
B  M2 dry 비율 @10      R0 대 R1.  실사용 조건(자기 소스 제외)
C  M3 패밀리 보존 @1    R0 대 R1.  "대가 없이 얻었는가"의 확인

세 조건이 모두 충족되어야 성공이므로 한 그림에 함께 둔다.
막대 옆 수치는 평균, 선은 소스 단위 부트스트랩 95% CI.

출력: figures/fig_q6_retrieval.{png,pdf,svg}
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

AXES = [("distortion_drive_db", "distortion\ndrive_db"),
        ("reverb_room_size", "reverb\nroom_size")]
LEVELS = ["24", "18", "12"]          # 강 → 약 (아래로 갈수록 약하게 걸림)
LEVEL_LABEL = {"24": "강 (lvl 24)", "18": "중 (lvl 18)", "12": "약 (lvl 12)"}

R0C, R3C, R1C, RORC = "#9AA3AD", "#7FA8C9", "#1F5FA0", "#CBD6E0"
INK, MUT, RULE, GOOD = "#22252A", "#5F6368", "#DDE1E6", "#3F8F6B"


def load():
    a = json.loads((RES / "11_phase9_r0r3ror.json").read_text(encoding="utf-8"))
    b = json.loads((RES / "11_phase9_r1.json").read_text(encoding="utf-8"))
    m = json.loads((RES / "11_phase9_m2m3.json").read_text(encoding="utf-8"))
    rows = []
    for key, label in AXES:
        for lv in LEVELS:
            r1 = b[key][lv]["R1"]
            rows.append({
                "axis": label, "lv": lv,
                "R0": a[key][lv]["R0"]["1"], "R3": a[key][lv]["R3"]["1"],
                "R1": r1["1"], "Ror": a[key][lv]["Ror"]["1"],
                "r10_R1": r1["10"]["mean"],
                "r1_vs_r0_ci": r1["vs_R0_r10_diff_ci"],
                "m2_R0": m["M2"][key][lv]["R0"]["10"],
                "m2_R1": m["M2"][key][lv]["R1"]["10"],
                "m3_R0": m["M3_on_M1"][key][lv]["R0_family_preserve"],
                "m3_R1": m["M3_on_M1"][key][lv]["R1_family_preserve"],
            })
    return rows


def dot(ax, x, y, c, s=52, hollow=False):
    if hollow:
        ax.scatter([x], [y], s=s, facecolor="white", edgecolor=c, lw=1.6, zorder=4)
    else:
        ax.scatter([x], [y], s=s, color=c, zorder=4)


def ci_line(ax, lo, hi, y, c, lw=1.6):
    ax.plot([lo, hi], [y, y], color=c, lw=lw, zorder=3, solid_capstyle="butt")


def main():
    rows = load()
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 5.9), sharey=True,
                            gridspec_kw={"wspace": 0.07})
    axA, axB, axC = axs

    y, ticks, labels, spans = 0.0, [], [], []
    for _, ax_label in AXES:
        start = y
        for lv in LEVELS:
            r = next(x for x in rows if x["axis"] == ax_label and x["lv"] == lv)

            # A — M1 R@10
            axA.plot([r["R0"]["mean"], r["Ror"]["mean"]], [y, y], color=RULE, lw=2.2,
                     zorder=1, solid_capstyle="round")
            for key, col, hol in [("Ror", RORC, True), ("R0", R0C, False),
                                  ("R3", R3C, False), ("R1", R1C, False)]:
                ci_line(axA, *r[key]["ci"], y, col, lw=1.4)
                dot(axA, r[key]["mean"], y, col, hollow=hol)
            axA.text(r["R1"]["mean"], y - 0.28, f"{r['R1']['mean']:.3f}",
                     ha="center", va="bottom", fontsize=8.8, color=R1C, fontweight="bold")
            axA.text(1.012, y, f"R@10 {r['r10_R1']:.3f}", va="center", ha="left",
                     fontsize=8.0, color=MUT)

            # B — M2 dry@10
            for key, col in [("m2_R0", R0C), ("m2_R1", R1C)]:
                ci_line(axB, *r[key]["dry_ratio_ci"], y, col, lw=1.4)
                dot(axB, r[key]["dry_ratio_mean"], y, col)
            axB.annotate("", xy=(r["m2_R1"]["dry_ratio_mean"] - 0.02, y),
                         xytext=(r["m2_R0"]["dry_ratio_mean"] + 0.02, y),
                         arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.3), zorder=2)
            axB.text(r["m2_R1"]["dry_ratio_mean"] + 0.025, y,
                     f"{r['m2_R1']['dry_ratio_mean']:.3f}", va="center", ha="left",
                     fontsize=8.8, color=INK)

            # C — M3 family@1
            for key, col in [("m3_R0", R0C), ("m3_R1", R1C)]:
                ci_line(axC, *r[key]["ci"], y, col, lw=1.4)
                dot(axC, r[key]["mean"], y, col)

            ticks.append(y); labels.append(LEVEL_LABEL[lv]); y += 1
        spans.append((ax_label, start, y - 1))
        y += 0.85

    for ax_label, y0, y1 in spans:
        axA.text(-0.335, (y0 + y1) / 2, ax_label, transform=axA.get_yaxis_transform(),
                 ha="center", va="center", fontsize=10.5, color=INK,
                 fontweight="bold", linespacing=1.35)
        axA.plot([-0.145, -0.145], [y0 - 0.42, y1 + 0.42],
                 transform=axA.get_yaxis_transform(), color=RULE, lw=1.4,
                 clip_on=False, zorder=0)

    axA.set_yticks(ticks); axA.set_yticklabels(labels, fontsize=9.8, color=MUT)
    axA.tick_params(axis="y", length=0, pad=5)
    axA.set_yticklabels(labels, fontsize=9.8, color=MUT, ha="right")
    axA.set_ylim(-0.75, y - 1.35); axA.invert_yaxis()

    axA.set_xlim(0.15, 1.16)
    axA.set_xticks([0.2,0.4,0.6,0.8,1.0])
    axB.set_xlim(-0.02, 0.95)
    axC.set_xlim(0.78, 1.02)
    axA.set_xlabel("M1  자기 dry 회수율  R@1", fontsize=10.5, color=INK, labelpad=9)
    axB.set_xlabel("M2  top-10 중 dry 비율   (실사용 조건)", fontsize=10.5, color=INK, labelpad=9)
    axC.set_xlabel("M3  패밀리 보존율  @1", fontsize=10.5, color=INK, labelpad=9)

    axB.axvline(1 / 3, color="#C0392B", lw=1.2, ls=(0, (4, 3)), zorder=2)
    axB.text(1 / 3, 1.005, "무작위 0.333", transform=axB.get_xaxis_transform(),
             fontsize=8.8, color="#C0392B", ha="center", va="bottom")

    for a in axs:
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(RULE)
        a.grid(axis="x", color=RULE, lw=0.7, alpha=0.75, zorder=0)
        a.set_axisbelow(True)
        a.tick_params(axis="x", colors=MUT, labelsize=9.3)
    for a in (axB, axC):
        a.tick_params(axis="y", length=0)

    handles = [
        Line2D([], [], marker="o", ls="", color=R0C, markersize=7, label="R0  무처리"),
        Line2D([], [], marker="o", ls="", color=R3C, markersize=7, label="R3  전역 평균 방향"),
        Line2D([], [], marker="o", ls="", color=R1C, markersize=7, label="R1  예측 방향"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=RORC,
               markeredgewidth=1.6, markersize=7, label="Ror  실제 보정 방향 (상한)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.005),
               ncol=4, frameon=False, fontsize=10, handletextpad=0.4, columnspacing=2.2)

    fig.subplots_adjust(left=0.165, right=0.985, top=0.925, bottom=0.20)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_q6_retrieval.{ext}",
                    dpi=220 if ext == "png" else None, facecolor="white")

    n = sum(1 for r in rows if r["r1_vs_r0_ci"][0] > 0)
    n3 = sum(1 for r in rows if r["m3_R1"]["mean"] >= r["m3_R0"]["mean"])
    print("저장:", OUT / "fig_q6_retrieval.png")
    print(f"R1 이 R0 대비 R@10 유의 개선: {n}/{len(rows)}")
    print(f"M3 가 하락하지 않은 조합: {n3}/{len(rows)}")
    print("M2 dry 비율 최대 변화:",
          max((r['m2_R1']['dry_ratio_mean'] / max(r['m2_R0']['dry_ratio_mean'], 1e-9))
              for r in rows))


if __name__ == "__main__":
    main()
