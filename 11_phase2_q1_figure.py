"""CLAP FX Probe — 11_phase2_q1_figure.py (Q1: R²의 범위 종속성 그래프)

`out/results/11_phase2_q1_finegrid_raw.json`(11_phase2_q1_finegrid.py 산출, 폭
5%~100% 20개 점)의 윈도우 R² 표를 그래프 2장으로 낸다. 측정된 23축 전부, 하위1/3
위치 값(<100%)과 전체범위(100%) 값을 이어서 쓴다. N<5000인 폭은 계산 자체가 안
됐으므로 곡선에서 제외한다.
"""
from pathlib import Path

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"

AXES = [
    "distortion_drive_db",
    "reverb_wet_level", "reverb_room_size", "reverb_damping", "reverb_width",
    "highshelf_gain", "lowshelf_gain", "peak_gain",
    "highshelf_cutoff_gp6", "lowshelf_cutoff_gp6", "peak_cutoff_gp6",
    "highshelf_q_gp6", "lowshelf_q_gp6", "peak_q_gp6",
    "highshelf_cutoff_gn6", "lowshelf_cutoff_gn6", "peak_cutoff_gn6",
    "highshelf_q_gn6", "lowshelf_q_gn6", "peak_q_gn6",
    "null_12k_gain", "null_15k_gain",
    "eq_cascade_intensity",
]
WIDTHS = [round(w, 2) for w in np.arange(0.05, 1.0001, 0.05)]
LINE_COLOR = "#2f6fab"
NOT_SIG_COLOR = "#c0392b"

OUT_DIR = Path("out/figures")


def is_significant(dose_raw, axis):
    """레벨 전이 중 하나라도 null 대비 부트스트랩 CI로 유의(clears_null)하면 유의 축으로 본다."""
    return any(r["clears_null"] for r in dose_raw[axis]["jnd"])


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def r2_curve(raw, axis):
    """100% 미만은 하위1/3, 100%는 전체범위. N<5000인 폭은 계산 자체가 안 돼 제외."""
    wr = raw[axis]
    xs, ys = [], []
    for w in WIDTHS:
        row = wr[str(w)]
        pos = "하위1/3" if w < 1.0 else "전체범위"
        r2 = row[pos]["r2"]
        if r2 is not None:
            xs.append(w)
            ys.append(r2)
    return xs, ys


def main():
    raw = json.load(open("out/results/11_phase2_q1_finegrid_raw.json"))
    dose_raw = json.load(open("out/results/11_phase2_doseresponse_raw.json"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    curves = {axis: r2_curve(raw, axis) for axis in AXES}
    multipliers = {axis: curves[axis][1][-1] / curves[axis][1][0] for axis in AXES}
    order = sorted(AXES, key=lambda a: -multipliers[a])
    sig = {axis: is_significant(dose_raw, axis) for axis in AXES}

    # 1) 소축 그리드 — 23축을 한 그래프에 겹치면 못 읽으므로 패널당 1축
    n_cols = 5
    n_rows = -(-len(AXES) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.3 * n_rows), dpi=150)
    fig.suptitle("Q1", fontsize=13)
    for i, axis in enumerate(AXES):
        ax = axes.flat[i]
        xs, ys = curves[axis]
        x = np.array(xs) * 100
        color = LINE_COLOR if sig[axis] else NOT_SIG_COLOR
        ax.plot(x, ys, marker="o", markersize=2.5, color=color, linewidth=1.5)
        title = f"{axis} ({multipliers[axis]:.1f}×)" if sig[axis] else f"{axis} (유의하지 않음)"
        ax.set_title(title, fontsize=8, color=INK_SECONDARY if sig[axis] else NOT_SIG_COLOR)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        style_axis(ax)
    for j in range(len(AXES), n_rows * n_cols):
        axes.flat[j].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "11_phase2_q1_r2_vs_window.png")
    plt.close(fig)

    # 2) 배수 막대 그래프 (20%→100%, 내림차순) — 배수만 크고 절댓값은 작은 축이 있어 R² 절댓값도 같이 표시
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
    labels = order
    vals = [multipliers[a] for a in labels]
    bar_colors = [LINE_COLOR if sig[a] else NOT_SIG_COLOR for a in labels]
    ax.bar(labels, vals, color=bar_colors, zorder=3)
    for i, a in enumerate(labels):
        r2_lo, r2_hi = curves[a][1][0], curves[a][1][-1]
        note = "" if sig[a] else " 유의하지 않음"
        ax.text(i, vals[i] + 1.5, f"{vals[i]:.1f}× ({r2_lo:.2f}→{r2_hi:.2f}){note}",
                ha="left", va="bottom", rotation=60, fontsize=7,
                color=INK_SECONDARY if sig[a] else NOT_SIG_COLOR)
    ax.set_ylim(0, max(vals) * 1.6)
    ax.set_ylabel("R² 배수 (전체범위 ÷ 20%폭)")
    ax.set_title("Q1")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    for tick, a in zip(ax.get_xticklabels(), labels):
        if not sig[a]:
            tick.set_color(NOT_SIG_COLOR)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "11_phase2_q1_r2_multiplier.png")
    plt.close(fig)

    print("저장:", OUT_DIR / "11_phase2_q1_r2_vs_window.png")
    print("저장:", OUT_DIR / "11_phase2_q1_r2_multiplier.png")


if __name__ == "__main__":
    main()
