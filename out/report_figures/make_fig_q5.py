# -*- coding: utf-8 -*-
"""Q5 그림 — 생성 경로에서의 전달.

A  directional_agreement 20조합(5축 × 4구간)과 95% CI.  0 이 무작위 기대값이다.
B  읽기(임베딩) 대 쓰기(오디오) 를 각도 축에서 직접 대조.
   무작위 널 89.97° [84.98, 94.77] 을 띠로 깔아, 쓰기 값이 그 안/경계에 놓임을 보인다.

Q5 의 요점은 "유의하다"가 아니라 **"유의하지만 무작위에서 2~6도밖에 안 벗어난다"**
이므로, 널과의 거리가 보이지 않으면 그림이 결론을 반대로 전달한다.

출력: figures/fig_q5_generation.{png,pdf,svg}
"""
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "out" / "results" / "11_phase6_directional_agreement.md"
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

AXES = [("highshelf_gain", "highshelf\ngain"), ("distortion_drive_db", "distortion\ndrive_db"),
        ("peak_gain", "peak\ngain"), ("reverb_room_size", "reverb\nroom_size"),
        ("lowshelf_gain", "lowshelf\ngain")]
INTERVALS = ["전범위", "하위1/3", "중위1/3", "상위1/3"]
ICOLOR = {"전범위": "#1F4E79", "하위1/3": "#3E7CB1",
          "중위1/3": "#79A7CE", "상위1/3": "#B4CDE4"}
OFFSET = {"전범위": -0.255, "하위1/3": -0.085, "중위1/3": 0.085, "상위1/3": 0.255}

# Q4 B2 전범위 (읽기 단계) — out/results/11_phase5_q3q4_raw.json
B2_FULL = {"distortion_drive_db": 0.8015, "reverb_room_size": 0.7065,
           "highshelf_gain": 0.8239, "lowshelf_gain": 0.7252, "peak_gain": 0.8007}
NULL_DEG = (84.98, 94.77)      # 무작위 널 95% 범위 (512차원, 1,000쌍 실측)
NULL_MEAN_DEG = 89.97

INK, MUT, RULE, WARN, READ = "#22252A", "#5F6368", "#DDE1E6", "#C0392B", "#3F8F6B"


def parse_md():
    """전체 n 표(첫 표)만 읽는다. 균형 서브샘플 표는 축 간 비교용이라 제외."""
    rows, seen = {}, set()
    pat = re.compile(
        r"^\|\s*(\w+)\s*\|\s*(전범위|하위1/3|중위1/3|상위1/3)\s*\|\s*(\d+)\s*\|"
        r"\s*([+-][\d.]+)\s*\[([+-][\d.]+),\s*([+-][\d.]+)\]")
    for line in MD.read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        axis, iv = m.group(1), m.group(2)
        if (axis, iv) in seen:      # 두 번째 표(균형 서브샘플)는 건너뜀
            continue
        seen.add((axis, iv))
        rows[(axis, iv)] = {"n": int(m.group(3)), "mean": float(m.group(4)),
                            "ci": [float(m.group(5)), float(m.group(6))]}
    return rows


def main():
    data = parse_md()
    missing = [(a, i) for a, _ in AXES for i in INTERVALS if (a, i) not in data]
    if missing:
        raise SystemExit(f"파싱 실패: {missing}")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.2, 6.0),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.30})

    # ── A. da 20조합 ─────────────────────────────────────────
    ticks, labels = [], []
    for yi, (key, label) in enumerate(AXES):
        for iv in INTERVALS:
            r, y, c = data[(key, iv)], yi + OFFSET[iv], ICOLOR[iv]
            axA.plot(r["ci"], [y, y], color=c, lw=1.6, zorder=3)
            axA.scatter([r["mean"]], [y], s=26, color=c, zorder=4)
        ticks.append(yi)
        labels.append(label)
        if yi:
            axA.axhline(yi - 0.5, color=RULE, lw=0.6, zorder=0)
    axA.axvline(0, color=WARN, lw=1.4, zorder=2)
    axA.text(0, 1.008, "0 = 무작위", transform=axA.get_xaxis_transform(),
             fontsize=9, color=WARN, ha="center", va="bottom")

    axA.set_yticks(ticks)
    axA.set_yticklabels(labels, fontsize=10, color=INK, linespacing=1.3)
    axA.tick_params(axis="y", length=0, pad=6)
    axA.set_ylim(-0.6, len(AXES) - 0.4)
    axA.invert_yaxis()
    axA.set_xlim(-0.018, 0.125)
    axA.set_xlabel("directional_agreement  cos(생성 변위, 원래 변위)",
                   fontsize=10.4, color=INK, labelpad=9)

    # ── B. 읽기 대 쓰기 (각도) ───────────────────────────────
    axB.axhspan(-1, len(AXES), xmin=0, xmax=1, color="white", zorder=0)
    axB.axvspan(*NULL_DEG, color="#FBEEEC", zorder=0)
    axB.axvline(NULL_MEAN_DEG, color=WARN, lw=1.2, ls=(0, (4, 3)), zorder=2)
    axB.text(NULL_MEAN_DEG, 1.008, "무작위 널 89.97°\n[84.98, 94.77]",
             transform=axB.get_xaxis_transform(), fontsize=8.8, color=WARN,
             ha="center", va="bottom", linespacing=1.3)

    for yi, (key, label) in enumerate(AXES):
        deg_read = np.degrees(np.arccos(B2_FULL[key]))
        deg_write = np.degrees(np.arccos(data[(key, "전범위")]["mean"]))
        axB.plot([deg_read, deg_write], [yi, yi], color="#B9BFC7", lw=2.2,
                 zorder=2, solid_capstyle="round")
        axB.scatter([deg_read], [yi], s=62, color=READ, zorder=4)
        axB.scatter([deg_write], [yi], s=62, color="#1F4E79", zorder=4)
        axB.text(deg_read - 1.6, yi, f"{deg_read:.1f}°", va="center", ha="right",
                 fontsize=9.2, color=READ)
        axB.text(deg_write + 1.6, yi, f"{deg_write:.1f}°", va="center", ha="left",
                 fontsize=9.2, color="#1F4E79")
        if yi:
            axB.axhline(yi - 0.5, color=RULE, lw=0.6, zorder=0)

    axB.set_yticks(range(len(AXES)))
    axB.set_yticklabels([l for _, l in AXES], fontsize=10, color=INK, linespacing=1.3)
    axB.tick_params(axis="y", length=0, pad=6)
    axB.set_ylim(-0.6, len(AXES) - 0.4)
    axB.invert_yaxis()
    axB.set_xlim(25, 100)
    axB.set_xlabel("각도 — 예측 방향과 실제 손잡이 사이 (전범위)",
                   fontsize=10.4, color=INK, labelpad=9)

    for a in (axA, axB):
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(RULE)
        a.grid(axis="x", color=RULE, lw=0.7, alpha=0.75, zorder=0)
        a.set_axisbelow(True)
        a.tick_params(axis="x", colors=MUT, labelsize=9.3)

    hA = [Line2D([], [], marker="o", ls="", color=ICOLOR[i], markersize=6.5, label=i)
          for i in INTERVALS]
    axA.legend(handles=hA, loc="lower center", bbox_to_anchor=(0.5, -0.235), ncol=4,
               frameon=False, fontsize=9.6, handletextpad=0.35, columnspacing=1.4)
    hB = [Line2D([], [], marker="o", ls="", color=READ, markersize=7,
                 label="읽기 — 임베딩 단계 (Q4 B2)"),
          Line2D([], [], marker="o", ls="", color="#1F4E79", markersize=7,
                 label="쓰기 — 오디오 단계 (Q5)")]
    axB.legend(handles=hB, loc="lower center", bbox_to_anchor=(0.5, -0.235), ncol=2,
               frameon=False, fontsize=9.6, handletextpad=0.4, columnspacing=1.8)

    fig.subplots_adjust(left=0.105, right=0.985, top=0.885, bottom=0.20)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_q5_generation.{ext}",
                    dpi=220 if ext == "png" else None, facecolor="white")

    n_pos = sum(1 for v in data.values() if v["ci"][0] > 0)
    degs = [np.degrees(np.arccos(data[(k, "전범위")]["mean"])) for k, _ in AXES]
    print("저장:", OUT / "fig_q5_generation.png")
    print(f"CI 하한 > 0 : {n_pos}/{len(data)}")
    print(f"전범위 각도 {min(degs):.1f}° ~ {max(degs):.1f}°  "
          f"(무작위 널 하한 {NULL_DEG[0]}° 와의 최소 거리 "
          f"{NULL_DEG[0]-max(degs):.1f}°)")


if __name__ == "__main__":
    main()
