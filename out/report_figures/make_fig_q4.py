# -*- coding: utf-8 -*-
"""Q4 그림 — B2(파라미터 미지) 방향 예측 대 between 기준선, 20개 조합 전부.

표에는 7개만 실려 있었으나 실제로는 5축 × 4구간 = 20개 조합이 모두 산출되어 있다.
"20/20 이 기준선을 넘었다"는 진술은 20개를 다 보여야 성립한다.

출력: figures/fig_q4_direction_prediction.{png,pdf,svg}
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

# 한글 폰트 — matplotlib 에 "KR" 패밀리가 등록되지 않는 환경이 있어 파일을 직접 추가한다
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
INTERVALS = ["전범위", "상위1/3", "중위1/3", "하위1/3"]   # 아래에서 위로 그리므로 역순

INK, MUT, ACC, BASE, RULE = "#22252A", "#5F6368", "#2F6FAF", "#B9BFC7", "#DDE1E6"


def load():
    q4 = json.loads(RAW.read_text(encoding="utf-8"))["q4"]
    rows = []
    for ax_key, ax_label in AXES:
        for iv in INTERVALS:
            k = f"{ax_key}::{iv}"
            rows.append({
                "axis": ax_label, "interval": iv,
                "b2": q4[k]["b2"]["mlp"]["cos_mean"],
                "base": q4[k]["between_baseline"],
            })
    return rows


def main():
    rows = load()
    fig, ax = plt.subplots(figsize=(8.6, 8.2))

    y, ticks, labels, group_spans = 0.0, [], [], []
    for gi, (_, ax_label) in enumerate(AXES):
        start = y
        for iv in INTERVALS:
            r = next(x for x in rows if x["axis"] == ax_label and x["interval"] == iv)
            ax.plot([r["base"], r["b2"]], [y, y], color=RULE, lw=2.4, zorder=1,
                    solid_capstyle="round")
            ax.scatter([r["base"]], [y], s=46, facecolor="white", edgecolor=BASE,
                       linewidth=1.6, zorder=3)
            ax.scatter([r["b2"]], [y], s=58, color=ACC, zorder=4)
            ax.text(r["b2"] + 0.022, y, f"{r['b2']:.3f}", va="center", ha="left",
                    fontsize=9.5, color=INK)
            ax.text(r["base"] - 0.022, y, f"{r['base']:.3f}", va="center", ha="right",
                    fontsize=8.5, color=MUT)
            ticks.append(y)
            labels.append(iv)
            y += 1
        group_spans.append((ax_label, start, y - 1))
        y += 0.9

    # 축 이름 — 그룹 왼쪽에 한 번씩
    for ax_label, y0, y1 in group_spans:
        ax.text(-0.185, (y0 + y1) / 2, ax_label, transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold",
                linespacing=1.35)
        ax.plot([-0.075, -0.075], [y0 - 0.42, y1 + 0.42],
                transform=ax.get_yaxis_transform(), color=RULE, lw=1.4,
                clip_on=False, zorder=0)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=10, color=MUT)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(-0.9, y - 0.5)
    ax.invert_yaxis()

    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("코사인  cos(예측 방향, 실제 손잡이)", fontsize=10.5, color=INK, labelpad=9)
    ax.tick_params(axis="x", colors=MUT, labelsize=9.5)

    # 위쪽 축 — 각도 환산
    top = ax.secondary_xaxis("top", functions=(lambda c: c, lambda c: c))
    tv = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    top.set_xticks(tv)
    top.set_xticklabels([f"{np.degrees(np.arccos(v)):.0f}°" for v in tv],
                        fontsize=9.5, color=MUT)
    top.set_xlabel("각도", fontsize=10.5, color=INK, labelpad=8)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="x", color=RULE, lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # 범례
    h1 = ax.scatter([], [], s=58, color=ACC, label="B2 — 파라미터 미지 조건의 예측")
    h2 = ax.scatter([], [], s=46, facecolor="white", edgecolor=BASE, linewidth=1.6,
                    label="between 기준선 — 소스 정보 미사용")
    ax.legend(handles=[h1, h2], loc="lower center", bbox_to_anchor=(0.5, -0.115),
              ncol=2, frameon=False, fontsize=10, handletextpad=0.5, columnspacing=1.8)

    fig.subplots_adjust(left=0.235, right=0.965, top=0.935, bottom=0.115)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_q4_direction_prediction.{ext}",
                    dpi=220 if ext == "png" else None, facecolor="white")
    print("저장:", OUT / "fig_q4_direction_prediction.png")
    print(f"조합 {len(rows)}개 · 전부 기준선 초과: "
          f"{all(r['b2'] > r['base'] for r in rows)}")
    gaps = [r["b2"] - r["base"] for r in rows]
    print(f"초과폭 최소 {min(gaps):.4f} / 최대 {max(gaps):.4f}")


if __name__ == "__main__":
    main()
