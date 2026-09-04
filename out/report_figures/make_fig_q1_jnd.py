# -*- coding: utf-8 -*-
"""Q1' 그림 — JND(분해능), 실무범위 대비 %.

JND 는 널 바닥을 넘는 최소 파라미터 간격이다. R² 와 달리 측정 범위 선택에
무관하므로 **축 간 비교는 이 값으로만 한다**(§5.1 참조).

세 부류를 구분한다.
  검출     20단계 미세 격자 5~19번째에서 문턱을 찾음 — 정밀 측정값
  좌측 절단  격자의 **첫 점**에서 이미 널을 넘음 — 더 잘게 못 쟀다는 뜻이므로
           측정값이 아니라 상한이다. reverb_wet_level 1축이 해당.
  미검출    어느 개별 구간도 단독으로 널을 넘지 못함 — 상한만 확정
           (원 25레벨 격자 1스텝 이하). 문턱 없이 점진 누적되는 축.

  ★ 두 절단 부류 모두 "≤" 이며 정밀 측정값과 같은 기호로 그리면 안 된다.

세로 순서는 오름차순, 가로는 로그축(0.004% ~ 4.02%, 약 1,000배 범위).

출처: out/results/11_phase2jnd_final.md (표 파싱)
출력: figures/fig_q1_jnd.{png,pdf,svg}
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
MD = ROOT / "out" / "results" / "11_phase2jnd_final.md"
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

# 미세 격자 첫 점에서 검출된 축 — 상한만 확정 (검증: jnd_theta == theta_fine_pos[0])
LEFT_CENSORED = {"reverb_wet_level"}

UNIT = {"distortion_drive_db": "dB", "reverb_wet_level": "", "reverb_room_size": "",
        "reverb_damping": "", "reverb_width": "", "eq_cascade_intensity": "",
        "highshelf_cutoff_gp6": "Hz", "lowshelf_cutoff_gp6": "Hz", "peak_cutoff_gp6": "Hz",
        "highshelf_q_gp6": "", "lowshelf_q_gp6": "", "peak_q_gp6": "",
        "highshelf_gain": "dB", "lowshelf_gain": "dB", "peak_gain": "dB"}

DET, UND, LCEN = "#1F5FA0", "#C8871F", "#8E5AA8"
INK, MUT, RULE = "#22252A", "#5F6368", "#DDE1E6"


def parse():
    pat = re.compile(r"^\|\s*([\w]+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
                     r"\s*~?([\d.e+-]+)[^|]*\|\s*(~?)\s*([\d.]+)%\s*\|")
    rows, seen = [], set()
    for line in MD.read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        axis, direction = m.group(1), m.group(2).strip()
        detected = m.group(5) != "~"
        key = (axis, direction)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"axis": axis, "dir": direction,
                     "detected": detected and axis not in LEFT_CENSORED,
                     "censored_left": axis in LEFT_CENSORED,
                     "native": float(m.group(4)), "pct": float(m.group(6))})
    return rows


def main():
    rows = parse()
    # 부호축은 boost/cut 이 동일하므로 한 행으로 합친다(값이 같음을 확인 후)
    merged, gain_seen = [], {}
    for r in rows:
        if r["dir"] in ("boost(+)", "cut(-)"):
            gain_seen.setdefault(r["axis"], []).append(r)
        else:
            merged.append(r)
    for axis, rs in gain_seen.items():
        pcts = {round(x["pct"], 4) for x in rs}
        assert len(pcts) == 1, f"{axis}: boost/cut 값이 달라 합칠 수 없음 {pcts}"
        r = dict(rs[0]); r["dir"] = "boost · cut 동일"
        merged.append(r)

    merged.sort(key=lambda r: r["pct"])
    n = len(merged)

    fig, ax = plt.subplots(figsize=(10.6, 0.42 * n + 3.0))
    for yi, r in enumerate(merged):
        c = DET if r["detected"] else (LCEN if r.get("censored_left") else UND)
        if r["detected"]:
            ax.scatter([r["pct"]], [yi], s=68, color=c, zorder=4)
        else:
            # 상한만 확정 — 왼쪽을 향한 화살로 "이하" 를 표시
            ax.scatter([r["pct"]], [yi], s=68, facecolor="white", edgecolor=c,
                       lw=1.8, zorder=4)
            ax.annotate("", xy=(r["pct"] * 0.40, yi), xytext=(r["pct"] * 0.93, yi),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.3), zorder=3)
        unit = UNIT.get(r["axis"], "")
        native = f"{r['native']:.4g}{(' ' + unit) if unit else ''}"
        lab = f"{'' if r['detected'] else '≤ '}{r['pct']:.3f}%"
        ax.text(r["pct"] * 1.18, yi, f"{lab}    ({native})", va="center", ha="left",
                fontsize=9.2, color=INK if r["detected"] else c)
        if yi:
            ax.axhline(yi - 0.5, color=RULE, lw=0.55, zorder=0)

    labels = [r["axis"] + (f"\n{r['dir']}" if r["dir"] not in ("—", "") else "")
              for r in merged]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9.5, color=INK, linespacing=1.3)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()

    ax.set_xscale("log")
    ax.set_xlim(0.0022, 26)
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(["0.01%", "0.1%", "1%", "10%"])
    ax.set_xlabel("JND — 실무 범위 대비 (로그축)", fontsize=10.6, color=INK, labelpad=10)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(axis="x", color=RULE, lw=0.7, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUT, labelsize=9.5)

    det = [r for r in merged if r["detected"]]
    span = det[-1]["pct"] / det[0]["pct"]
    ax.set_title(f"정밀 측정된 {len(det)}축의 분해능 차이 약 {span:,.0f}배"
                 f"   (절단 {len(merged)-len(det)}축 제외)",
                 fontsize=11, color=INK, pad=12, loc="left")

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=DET, markersize=7.5,
               label="검출 — 미세 격자에서 실제 문턱을 찾음"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=LCEN,
               markeredgewidth=1.8, markersize=7.5,
               label="좌측 절단 — 격자 첫 점에서 이미 넘음. 더 잘게 못 쟀다 (상한)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=UND,
               markeredgewidth=1.8, markersize=7.5,
               label="미검출 — 문턱 없이 점진 누적. 원 격자 1스텝 이하 (상한)")],
        loc="upper center", bbox_to_anchor=(0.5, -0.105),
        ncol=1, frameon=False, fontsize=9.6, handletextpad=0.5)

    fig.subplots_adjust(left=0.235, right=0.975, top=0.935, bottom=0.20)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_q1_jnd.{ext}",
                    dpi=220 if ext == "png" else None, facecolor="white")

    nd = sum(1 for r in merged if not r["detected"])
    print("저장:", OUT / "fig_q1_jnd.png")
    print(f"행 {n}개 · 검출 {n-nd} / 미검출 {nd}")
    print(f"정밀 측정 {len(det)}축 범위 {det[0]['pct']:.4f}% ({det[0]['axis']}) ~ "
          f"{det[-1]['pct']:.4f}% ({det[-1]['axis']})  = {span:,.0f}배")


if __name__ == "__main__":
    main()
