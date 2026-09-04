"""CLAP FX Probe — 11_phase2_q1_jnd.py (Q1: 파라미터 위치별 JND 이동량 곡선, 전체 23축)

`out/results/11_phase2_doseresponse_raw.json`(기존 산출, 재계산 없음)의 `jnd`
필드 — 레벨 i→i+1 사이 임베딩 이동량(mean_delta) — 를 파라미터 위치(%)에 대해
그린다. 측정된 23축 전부를 소축 그리드(small multiples)로 낸다 — 23개를 한
그래프에 겹치면 못 읽으므로 패널당 1축.
"""
import json
from pathlib import Path

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
LINE_COLOR = "#2f6fab"
NOT_SIG_COLOR = "#c0392b"


def is_significant(raw, axis):
    """레벨 전이 중 하나라도 null 대비 부트스트랩 CI로 유의(clears_null)하면 유의 축으로 본다."""
    return any(r["clears_null"] for r in raw[axis]["jnd"])

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
OUT_DIR = Path("out/figures")


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def jnd_curve(raw, axis):
    rows = raw[axis]["jnd"]
    t_lo = rows[0]["theta_from"]
    t_hi = rows[-1]["theta_to"]
    span = t_hi - t_lo
    pos_pct = [((r["theta_from"] + r["theta_to"]) / 2.0 - t_lo) / span * 100 for r in rows]
    delta = [r["mean_delta"] for r in rows]
    return pos_pct, delta


def main():
    raw = json.load(open("out/results/11_phase2_doseresponse_raw.json"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_cols = 5
    n_rows = -(-len(AXES) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.6 * n_rows), dpi=150)
    fig.suptitle("Q1", fontsize=13)
    for i, axis in enumerate(AXES):
        ax = axes.flat[i]
        pos_pct, delta = jnd_curve(raw, axis)
        d_lo, d_hi = min(delta), max(delta)
        d_norm = [(d - d_lo) / (d_hi - d_lo) if d_hi > d_lo else 0.5 for d in delta]
        sig = is_significant(raw, axis)
        color = LINE_COLOR if sig else NOT_SIG_COLOR
        ax.plot(pos_pct, d_norm, marker="o", markersize=2.5, color=color, linewidth=1.5)
        title = axis if sig else f"{axis} (유의하지 않음)"
        ax.set_title(f"{title}\n({d_lo:.4f}~{d_hi:.4f})", fontsize=7, color=INK_SECONDARY if sig else NOT_SIG_COLOR)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)
        style_axis(ax)
    for j in range(len(AXES), n_rows * n_cols):
        axes.flat[j].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "11_phase2_q1_jnd_vs_param.png")
    plt.close(fig)

    print("저장:", OUT_DIR / "11_phase2_q1_jnd_vs_param.png")


if __name__ == "__main__":
    main()
