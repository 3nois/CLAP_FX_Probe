# -*- coding: utf-8 -*-
"""보고서용 핵심 그림 2개. 수치는 기존 results 파일에서만 읽는다 (재계산 없음).

그림 1 데이터 출처: out/results/results_2.json (controls, effects[*].probe_nmi)
그림 2 데이터 출처: out/results/results_8.json (reverse_b2[*].mlp.cos_mean)
                    out/results/results_9_phase_f4.json (directional_agreement.by_effect)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available = {f.name for f in fm.fontManager.ttflist}
for _f in _KOREAN_FONT_CANDIDATES:
    if _f in _available:
        plt.rcParams["font.family"] = _f
        break
else:
    raise RuntimeError(f"한글 폰트를 찾지 못했다. 설치된 폰트: {sorted(_available)[:20]}...")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "out" / "results"
FIGDIR = ROOT / "out" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

INK = "#2b2a27"
INK2 = "#5a5954"
GRID = "#e6e4de"
GOLD = "#c9962f"       # 악기 패밀리 (기준·상한)
EFFECT_GRAY = "#8a8880"  # 이펙트 공통 색 (그림1)
BLUE = "#3573d1"        # 임베딩 단계
ORANGE = "#d1743f"      # 오디오 단계
RED = "#c0392b"


def style(ax, title, subtitle=None, ylabel=None):
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=INK2)
    y0 = 1.06 if subtitle else 1.0
    ax.text(0.0, y0 + 0.05, title, transform=ax.transAxes, fontsize=15,
             fontweight="bold", color=INK, ha="left", va="bottom")
    if subtitle:
        ax.text(0.0, y0 - 0.01, subtitle, transform=ax.transAxes, fontsize=10.5,
                 color=INK2, ha="left", va="bottom")


# ============================================================
# 그림 1 — 악기 정체성 vs 이펙트 정보량
# ============================================================
def fig1():
    d2 = json.load(open(RESULTS / "results_2.json"))
    family_nmi = d2["controls"]["instrument_family_7class_subsampled"]["nmi"]
    effect_nmi = {e: d2["effects"][e]["probe_nmi"] for e in ["distortion", "reverb", "highshelf"]}

    labels = ["악기 패밀리", "distortion", "reverb", "highshelf"]
    values = [family_nmi, effect_nmi["distortion"], effect_nmi["reverb"], effect_nmi["highshelf"]]
    colors = [GOLD, EFFECT_GRAY, EFFECT_GRAY, EFFECT_GRAY]

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    x = np.arange(4)
    bars = ax.bar(x, values, color=colors, width=0.6, zorder=3,
                   edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5, color=INK)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 1.0)

    for xi, v in zip(x, values):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=11.5, fontweight="bold", color=INK)

    for xi, e in zip(x[1:], ["distortion", "reverb", "highshelf"]):
        ratio = family_nmi / effect_nmi[e]
        ax.text(xi, effect_nmi[e] + 0.10, f"{ratio:.1f}배 차이", ha="center", va="bottom",
                 fontsize=9.5, color=RED, fontweight="bold")
        ax.annotate("", xy=(xi, effect_nmi[e] + 0.095), xytext=(xi, family_nmi - 0.01),
                    arrowprops=dict(arrowstyle="-", color=RED, lw=1.0, linestyle=(0, (3, 2))))

    style(ax, "CLAP 임베딩은 악기는 잘 담고 이펙트는 약하게 담는다",
          subtitle="동일 지표(NMI) · 동일 클래스 수(7) 비교",
          ylabel="NMI (normalized mutual information)")

    fig.text(0.02, 0.02,
            "※ 악기 패밀리와 이펙트 모두 동일한 7클래스 분류 프로브·동일 NMI 지표로 산출했다.\n"
            "지표가 다르면(R² vs accuracy) 이 비교 자체가 성립하지 않는다 — 1차에서 실제로 발생한 오류.",
            fontsize=8.5, color=INK2, ha="left", va="bottom")

    fig.tight_layout(rect=[0.0, 0.11, 1.0, 0.90])
    out = FIGDIR / "report_fig1_instrument_vs_effect.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)
    print(f"  family_nmi={family_nmi:.4f}  distortion={effect_nmi['distortion']:.4f} "
          f"reverb={effect_nmi['reverb']:.4f} highshelf={effect_nmi['highshelf']:.4f}")


# ============================================================
# 그림 2 — 임베딩 단계 vs 오디오 단계
# ============================================================
def cos_to_deg(c):
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def fig2():
    d8 = json.load(open(RESULTS / "results_8.json"))
    d9 = json.load(open(RESULTS / "results_9_phase_f4.json"))

    effects = ["reverb", "distortion", "highshelf"]
    embed_cos = {e: d8["reverse_b2"][e]["mlp"]["cos_mean"] for e in effects}
    audio = {e: d9["directional_agreement"]["by_effect"][e] for e in effects}
    overall = d9["directional_agreement"]["overall"]

    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    x = np.arange(len(effects)) * 1.15
    w = 0.36

    embed_vals = [embed_cos[e] for e in effects]
    audio_vals = [audio[e]["mean"] for e in effects]
    audio_err_low = [audio[e]["mean"] - audio[e]["ci95"][0] for e in effects]
    audio_err_high = [audio[e]["ci95"][1] - audio[e]["mean"] for e in effects]

    ax.bar(x - w / 2, embed_vals, w, color=BLUE, zorder=3, edgecolor="white", linewidth=0.5,
           label="임베딩 단계 (방향 예측, B2)")

    audio_colors = []
    audio_hatches = []
    for e in effects:
        ci_low, ci_high = audio[e]["ci95"]
        is_null = ci_low < 0 < ci_high
        audio_colors.append("#e8b9a8" if is_null else ORANGE)
        audio_hatches.append("////" if is_null else None)

    b2 = ax.bar(x + w / 2, audio_vals, w, color=audio_colors, zorder=3,
                edgecolor="white", linewidth=0.5,
                yerr=[audio_err_low, audio_err_high], ecolor=INK2, capsize=4,
                error_kw=dict(lw=1.1, zorder=4),
                label="오디오 단계 (방향 일치도)")
    for bar, hatch in zip(b2, audio_hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("#a9573c")

    ax.set_xlim(-0.7, x[-1] + 0.95)
    ax.set_ylim(-0.16, 1.0)

    ax.axhline(0.0, color=INK2, linewidth=0.9, zorder=2)
    ax.text(x[-1] + 0.72, 0.0, "무작위\n(90도)", fontsize=8.5, color=INK2, ha="center", va="center",
            linespacing=1.3)

    baseline = 0.34
    ax.axhline(baseline, color=RED, linewidth=1.0, linestyle=(0, (4, 2)), zorder=2)
    ax.text(x[-1] + 0.72, baseline, "악기군\n평균 기준선", fontsize=8.5, color=RED, ha="center", va="center",
            linespacing=1.3)

    for xi, e in zip(x - w / 2, effects):
        v = embed_cos[e]
        deg = cos_to_deg(v)
        ax.text(xi, v + 0.03, f"{v:.3f}\n({deg:.0f}°)", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=INK, linespacing=1.3)

    for xi, e in zip(x + w / 2, effects):
        v = audio[e]["mean"]
        deg = cos_to_deg(v)
        is_null = audio[e]["ci95"][0] < 0 < audio[e]["ci95"][1]
        y_txt = audio[e]["ci95"][1] + 0.025 if v >= 0 else audio[e]["ci95"][0] - 0.06
        label = f"null\n({v:+.3f})" if is_null else f"{v:+.3f}\n({deg:.0f}°)"
        ax.text(xi, y_txt, label, ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=10, fontweight="bold", linespacing=1.3,
                 color=INK2 if is_null else INK)

    ax.set_xticks(x)
    ax.set_xticklabels(effects, fontsize=12.5, color=INK)

    ax2 = ax.twinx()
    ax2.set_ylim(-0.16, 1.0)
    cos_ticks = np.array([1.0, 0.71, 0.5, 0.0])
    ax2.set_yticks(cos_ticks)
    ax2.set_yticklabels([f"{cos_to_deg(c):.0f}°" for c in cos_ticks], fontsize=9, color=INK2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax2.tick_params(axis="y", colors=INK2, length=3)
    ax2.text(1.085, 0.5, "각도 (도)", transform=ax2.transAxes, fontsize=10, color=INK2,
              ha="left", va="center", rotation=90)

    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.0), frameon=False, fontsize=9,
              handlelength=1.3, handletextpad=0.5)

    style(ax, "손잡이 방향은 예측되지만 오디오로는 전달되지 않는다",
          subtitle="임베딩 단계 31~45도  →  오디오 단계 86~88도",
          ylabel="코사인 유사도")

    fig.text(0.02, 0.02,
            f"※ 오디오 단계는 통계적으로 유의하나(전체 +{overall['mean']:.4f}, "
            f"95% CI [{overall['ci95'][0]:.3f}, {overall['ci95'][1]:.3f}] — 0 배제, n={overall['n']}),\n"
            "효과 크기가 손잡이로 쓰기에는 부족하다. reverb(빗금)는 CI 가 0을 포함해 null이다.",
            fontsize=8.5, color=INK2, ha="left", va="bottom")

    fig.tight_layout(rect=[0.0, 0.08, 0.93, 0.92])
    out = FIGDIR / "report_fig2_embedding_vs_audio.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)
    print(f"  embed: reverb={embed_cos['reverb']:.4f} distortion={embed_cos['distortion']:.4f} "
          f"highshelf={embed_cos['highshelf']:.4f}")
    print(f"  audio: reverb={audio['reverb']['mean']:.4f} distortion={audio['distortion']['mean']:.4f} "
          f"highshelf={audio['highshelf']['mean']:.4f}  overall={overall['mean']:.4f}")


if __name__ == "__main__":
    fig1()
    fig2()
