"""CLAP FX Probe — 11_phase2_q1_paramsweep.py (Q1: 파라미터 값 자체에 따른 R² 변화)

기존 windowed_r2_table은 위치를 하위/중위/상위 1/3 3점만 준다. 파라미터 축을 따라
R²가 실제로 어떻게 움직이는지 보려면 폭을 20%로 고정하고 창의 중심을 파라미터
범위 전체에 촘촘히 슬라이드해야 한다 — `11_phase2_doseresponse.windowed_r2_table`의
핵심 로직(Ridge + GroupShuffleSplit)만 재사용해 중심점 스윕으로 재구성한다.
CLAP 재계산 없음.
"""
import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

dr = import_module("11_phase2_doseresponse")

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
WIDTH = 0.20
N_CENTERS = 17
LINE_COLOR = "#2f6fab"
NOT_SIG_COLOR = "#c0392b"


def is_significant(dose_raw, axis):
    """레벨 전이 중 하나라도 null 대비 부트스트랩 CI로 유의(clears_null)하면 유의 축으로 본다."""
    return any(r["clears_null"] for r in dose_raw[axis]["jnd"])


OUT_DIR = Path("out/figures")


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def center_sweep_r2(emb, theta_raw, src_id, width, centers):
    """windowed_r2_table 내부 로직과 동일 — 위치를 3버킷 대신 임의 중심점으로 스윕.
    N<5000(dr.N_MIN_WINDOW)이면 원본과 동일하게 검정력 부족으로 스킵(None)."""
    t_lo, t_hi = theta_raw.min(), theta_raw.max()
    span = t_hi - t_lo
    w_span = width * span
    out = []
    for center in centers:
        lo_bound = max(center - w_span / 2.0, t_lo)
        hi_bound = min(center + w_span / 2.0, t_hi)
        level_idx = np.where((theta_raw >= lo_bound - 1e-9) & (theta_raw <= hi_bound + 1e-9))[0]
        n_rows = len(level_idx) * emb.shape[0]
        if len(level_idx) < 3 or n_rows < dr.N_MIN_WINDOW:
            out.append(None)
            continue
        X = emb[:, level_idx, :].reshape(-1, 512)
        theta_sub = theta_raw[level_idx]
        theta_norm = (theta_sub - theta_sub.min()) / (theta_sub.max() - theta_sub.min() + 1e-12)
        y = np.tile(theta_norm, emb.shape[0])
        groups = np.repeat(src_id, len(level_idx))
        gss = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
        r2s = []
        for train_idx, test_idx in gss.split(X, y, groups):
            model = Ridge(alpha=1.0)
            model.fit(X[train_idx], y[train_idx])
            r2s.append(r2_score(y[test_idx], model.predict(X[test_idx])))
        out.append(float(np.mean(r2s)))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw = {}
    curves = {}
    for axis in AXES:
        emb, theta_raw, src_id = dr.load_concat(axis)
        t_lo, t_hi = theta_raw.min(), theta_raw.max()
        span = t_hi - t_lo
        half = WIDTH * span / 2.0
        centers = np.linspace(t_lo + half, t_hi - half, N_CENTERS)
        r2s = center_sweep_r2(emb, theta_raw, src_id, WIDTH, centers)
        pos_frac = (centers - t_lo) / span * 100  # 파라미터 범위 내 위치 (0~100%)
        curves[axis] = (pos_frac, r2s)
        raw[axis] = {"theta_center": centers.tolist(), "pos_pct": pos_frac.tolist(), "r2": r2s}
        print(f"  {axis} 완료 ({time.time() - t0:.1f}s)")

    with open("out/results/11_phase2_q1_paramsweep_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    dose_raw = json.load(open("out/results/11_phase2_doseresponse_raw.json"))

    n_cols = 5
    n_rows = -(-len(AXES) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.6 * n_rows), dpi=150)
    fig.suptitle("Q1", fontsize=13)
    for i, axis in enumerate(AXES):
        ax = axes.flat[i]
        pos_frac, r2s = curves[axis]
        xs = [x for x, y in zip(pos_frac, r2s) if y is not None]
        ys = [y for y in r2s if y is not None]
        sig = is_significant(dose_raw, axis)
        color = LINE_COLOR if sig else NOT_SIG_COLOR
        if ys:
            y_lo, y_hi = min(ys), max(ys)
            y_norm = [(y - y_lo) / (y_hi - y_lo) if y_hi > y_lo else 0.5 for y in ys]
            ax.plot(xs, y_norm, marker="o", markersize=2.5, color=color, linewidth=1.5)
            title = axis if sig else f"{axis} (유의하지 않음)"
            ax.set_title(f"{title}\n({y_lo:.3f}~{y_hi:.3f})", fontsize=7, color=INK_SECONDARY if sig else NOT_SIG_COLOR)
        else:
            title = axis if sig else f"{axis} (유의하지 않음)"
            ax.set_title(f"{title}\n(N 부족)", fontsize=7, color=INK_SECONDARY if sig else NOT_SIG_COLOR)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)
        style_axis(ax)
    for j in range(len(AXES), n_rows * n_cols):
        axes.flat[j].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "11_phase2_q1_r2_vs_param.png")
    plt.close(fig)

    print(f"저장: {OUT_DIR / '11_phase2_q1_r2_vs_param.png'} (총 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
