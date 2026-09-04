# -*- coding: utf-8 -*-
"""Q6 축 선택 그림 — 천장 효과.

무처리(R0) 기준선이 이미 포화된 축은 "개선"을 정의할 수 없다. 이 그림은
그 판단 근거와 원인을 함께 보인다.

A  cos(wet, 자기 dry) 분포 — 이펙트가 임베딩을 얼마나 움직이는가 (원인)
B  무처리 회수율 R@1 · R@10 과 남은 여지 (결과)

수치는 캐시에서 직접 재계산한다(하드코딩 아님). recall 은 group-aware —
라이브러리 1,200곡 중 20곡이 바이트 단위 중복이라 동일본 회수를 오답으로
세면 검색 품질이 아니라 정렬 동점 순서를 재게 된다.

출력: figures/fig_q6_ceiling.{png,pdf,svg}
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
CACHE, RES = ROOT / "out" / "caches", ROOT / "out" / "results"
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

# 축, 표시명, 최대 레벨 인덱스, 최대 레벨 설명
TARGETS = [("highshelf_gain", "highshelf_gain", 24, "+15 dB"),
           ("reverb_room_size", "reverb_room_size", 24, "room 0.85"),
           ("distortion_drive_db", "distortion_drive_db", 24, "20 dB")]

CEIL, KEEP = "#C0392B", "#1F5FA0"
INK, MUT, RULE, HEAD = "#22252A", "#5F6368", "#DDE1E6", "#E8ECEF"


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def load_library():
    b1, b2 = np.load(CACHE / "11_phase2_bypass.npz"), np.load(CACHE / "11_phase2ext_bypass.npz")
    sid = np.concatenate([b1["src_id"], b2["src_id"]])
    emb = np.concatenate([b1["embeddings"], b2["embeddings"]])
    o = np.argsort(sid)
    return unit(emb[o].astype(np.float64)), sid[o]


def load_axis(axis, lvl):
    a, b = np.load(CACHE / f"11_phase2_{axis}.npz"), np.load(CACHE / f"11_phase2ext_{axis}.npz")
    sid = np.concatenate([a["src_id"], b["src_id"]])
    emb = np.concatenate([a["embeddings"], b["embeddings"]])
    o = np.argsort(sid)
    return unit(emb[o].astype(np.float64)[:, lvl, :])


def measure():
    lib, sid = load_library()
    N = len(sid)
    grp = json.loads((RES / "11_phase9_dupgroups.json").read_text(encoding="utf-8"))["group_of_src_id"]
    ok = [np.array(grp.get(str(i), [i])) for i in range(N)]   # 정답으로 인정할 집합

    out = []
    for axis, label, lvl, lvl_desc in TARGETS:
        W = load_axis(axis, lvl)
        S = W @ lib.T
        self_cos = S[np.arange(N), np.arange(N)]
        order = np.argsort(-S, axis=1)
        hit1 = np.array([order[i, 0] in ok[i] for i in range(N)])
        hit10 = np.array([len(set(order[i, :10]) & set(ok[i].tolist())) > 0 for i in range(N)])
        out.append({"axis": label, "lvl_desc": lvl_desc,
                    "cos": self_cos, "cos_med": float(np.median(self_cos)),
                    "r1": float(hit1.mean()), "r10": float(hit10.mean())})
    return out


def main():
    rows = measure()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.6), sharey=True,
                                   gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.08})

    for yi, r in enumerate(rows):
        ceil = r["r10"] > 0.99
        c = CEIL if ceil else KEEP

        # A — cos(wet, 자기 dry) 분포
        q = np.percentile(r["cos"], [5, 25, 50, 75, 95])
        axA.plot([q[0], q[4]], [yi, yi], color=c, lw=1.4, alpha=0.55, zorder=3)
        axA.plot([q[1], q[3]], [yi, yi], color=c, lw=6.5, alpha=0.30,
                 zorder=3, solid_capstyle="butt")
        axA.scatter([q[2]], [yi], s=62, color=c, zorder=5)
        axA.text(q[2], yi - 0.30, f"{q[2]:.3f}", ha="center", va="bottom",
                 fontsize=9.2, color=c, fontweight="bold")

        # B — 무처리 회수율과 남은 여지
        for j, (key, lab) in enumerate([("r1", "R@1"), ("r10", "R@10")]):
            yy = yi + (-0.17 if j == 0 else 0.17)
            v = r[key]
            axB.barh(yy, 1.0, height=0.28, color=HEAD, zorder=2)
            axB.barh(yy, v, height=0.28, color=c, alpha=0.85 if j else 0.55, zorder=3)
            axB.text(-0.012, yy, lab, va="center", ha="right", fontsize=8.6, color=MUT)
            axB.text(1.022, yy, f"{v:.4f}", va="center", ha="left",
                     fontsize=9.0, color=INK)
            if 1 - v > 0.16:
                axB.text(v + (1 - v) / 2, yy, f"여지 {1-v:.3f}", va="center",
                         ha="center", fontsize=8.2, color=MUT)

        axB.text(1.135, yi, "천장 — 제외" if ceil else "채택",
                 va="center", ha="left", fontsize=9.6, color=c, fontweight="bold")
        if yi:
            for a in (axA, axB):
                a.axhline(yi - 0.5, color=RULE, lw=0.6, zorder=0)

    axA.set_yticks(range(len(rows)))
    axA.set_yticklabels([f"{r['axis']}\n{r['lvl_desc']}" for r in rows],
                        fontsize=9.8, color=INK, linespacing=1.35)
    axA.tick_params(axis="y", length=0, pad=6)
    axA.set_ylim(-0.6, len(rows) - 0.4)
    axA.invert_yaxis()

    axA.set_xlim(0.30, 1.02)
    axA.set_xlabel("cos(wet, 자기 dry)   중앙값 · 사분위 · 5–95%",
                   fontsize=10.3, color=INK, labelpad=9)
    axB.set_xlim(-0.10, 1.40)
    axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axB.set_xlabel("무처리(R0) 회수율과 남은 여지", fontsize=10.3, color=INK, labelpad=9)

    for a in (axA, axB):
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(RULE)
        a.grid(axis="x", color=RULE, lw=0.7, alpha=0.75, zorder=0)
        a.set_axisbelow(True)
        a.tick_params(axis="x", colors=MUT, labelsize=9.3)
    axB.tick_params(axis="y", length=0)

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=CEIL, markersize=7.5,
               label="이펙트가 임베딩을 거의 안 움직임 → 무처리로 이미 회수 → 개선을 정의할 수 없음"),
        Line2D([], [], marker="o", ls="", color=KEEP, markersize=7.5,
               label="여지 있음 → 주 실험 채택")],
        loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False,
        fontsize=9.6, handletextpad=0.5)

    fig.subplots_adjust(left=0.135, right=0.995, top=0.955, bottom=0.315)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_q6_ceiling.{ext}",
                    dpi=220 if ext == "png" else None, facecolor="white")

    print("저장:", OUT / "fig_q6_ceiling.png")
    for r in rows:
        print(f"  {r['axis']:22s} cos중앙 {r['cos_med']:.4f}  "
              f"R@1 {r['r1']:.4f}  R@10 {r['r10']:.4f}")


if __name__ == "__main__":
    main()
