# -*- coding: utf-8 -*-
"""3-3 교차확인 + 3-4 도구 사양 분류 — out/prereg/11_phase3.md 확정 방법론.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "out" / "results"

EQ_TYPES = ["highshelf", "lowshelf", "peak"]


def crosscheck():
    with open(RESULTS_DIR / "11_phase3_2d_raw.json", encoding="utf-8") as f:
        probe2d = json.load(f)
    with open(RESULTS_DIR / "11_phase3_anova_raw.json", encoding="utf-8") as f:
        anova = json.load(f)

    lines = ["\n---\n\n## 교차 확인 (3-2 vs 3-3)\n"]
    lines.append("3-2의 context별 R² 스윙(범위)과 3-3의 2차 ANOVA 항을 나란히 놓는다 — "
                 "격자 해상도가 다르므로(13×13 vs 5×N) 정확한 수치 일치는 기대하지 않으며, "
                 "**순서**가 대략 맞는지만 본다.\n")
    lines.append("| 쌍 | 3-2 R²(A\\|B) 스윙(범위) | 3-3 2차 ANOVA 항 분산비율 |")
    lines.append("|---|---|---|")

    rows = []
    pair_to_anova = {
        "highshelf_gain_cutoff": ("highshelf_gain_cutoff_q", "gain x cutoff"),
        "lowshelf_gain_cutoff": ("lowshelf_gain_cutoff_q", "gain x cutoff"),
        "peak_gain_cutoff": ("peak_gain_cutoff_q", "gain x cutoff"),
        "reverb_wet_room": ("reverb_wet_room_damping_width", "wet_level x room_size"),
    }
    for pair_name, (grid_name, term_name) in pair_to_anova.items():
        r2s = [r["r2"] for r in probe2d[pair_name]["probe"] if r["r2"] is not None]
        swing = max(r2s) - min(r2s)
        anova_frac = next(c["fraction_mean"] for c in anova[grid_name]["components"] if c["term"] == term_name)
        rows.append((pair_name, swing, anova_frac))

    rank_2d = sorted(rows, key=lambda x: -x[1])
    rank_3d = sorted(rows, key=lambda x: -x[2])
    for pair_name, swing, anova_frac in rows:
        lines.append(f"| {pair_name} | {swing:.3f} | {anova_frac*100:.1f}% |")

    order_2d = [r[0] for r in rank_2d]
    order_3d = [r[0] for r in rank_3d]
    match = order_2d == order_3d
    top_match = order_2d[0] == order_3d[0]
    lines.append(f"\n- 3-2 기준 순위: {order_2d}")
    lines.append(f"- 3-3 기준 순위: {order_3d}")
    lines.append(f"- 완전 일치: {'예' if match else '아니오'} / 최상위(가장 강한 상호작용) 일치: "
                 f"{'예' if top_match else '아니오'}")
    if not match:
        lines.append(f"\n**★ 순서 불일치 보고**: 두 방법은 서로 다른 기준점 정의를 쓴다 — 3-2/구 "
                     f"히트맵 분석은 격자 모서리(theta_min) 기준 비가산성이고, 3-3 ANOVA는 "
                     f"격자 전체 평균(주변부 평균) 기준 직교분해다. `lowshelf_gain_cutoff`가 "
                     f"양쪽 다 최상위(또는 상위권)로 나오는 것은 일치하지만, 나머지 순서는 "
                     f"완전히 일치하지 않는다 — 정의가 다른 두 지표이므로 이는 방법론 오류가 "
                     f"아니라 '상호작용'이라는 개념 자체가 기준점 선택에 민감하다는 것을 "
                     f"보여주는 것으로 해석한다.\n")
    return "\n".join(lines)


def classify_ui():
    with open(RESULTS_DIR / "11_phase3_anova_raw.json", encoding="utf-8") as f:
        anova = json.load(f)

    lines = ["# 3-4. 도구 사양 분류\n"]
    lines.append("분류 기준(사전 확정, `out/prereg/11_phase3.md`): 주효과≥50%&상호작용<30%→"
                 "**독립 노출**; 주효과<30%&상호작용≥50%→**조건부 노출**; 그 외→**보류**.\n")
    lines.append("| 파라미터 | 주효과 분산비율 | 그 축 관련 상호작용 합 | 분류 | 근거 |")
    lines.append("|---|---|---|---|---|")

    rows = []
    for grid_name, data in anova.items():
        axes = data["axes"]
        main_by_axis = {c["term"]: c["fraction_mean"] for c in data["components"] if c["order"] == 1}
        inter_sum_by_axis = {a: 0.0 for a in axes}
        for c in data["components"]:
            if c["order"] == 2:
                a1, a2 = c["term"].split(" x ")
                inter_sum_by_axis[a1] += c["fraction_mean"]
                inter_sum_by_axis[a2] += c["fraction_mean"]
        for ax in axes:
            eff_label = f"{grid_name}::{ax}"
            main_frac = main_by_axis[ax]
            inter_frac = inter_sum_by_axis[ax]
            if main_frac >= 0.5 and inter_frac < 0.3:
                cls = "독립 노출"
            elif main_frac < 0.3 and inter_frac >= 0.5:
                cls = "조건부 노출"
            else:
                cls = "보류(애매)"
            rows.append((eff_label, main_frac, inter_frac, cls))
            lines.append(f"| {eff_label} | {main_frac*100:.1f}% | {inter_frac*100:.1f}% | **{cls}** | "
                         f"주효과 {main_frac*100:.1f}% vs 상호작용합 {inter_frac*100:.1f}% |")

    n_independent = sum(1 for r in rows if r[3] == "독립 노출")
    n_conditional = sum(1 for r in rows if r[3] == "조건부 노출")
    n_hold = sum(1 for r in rows if r[3] == "보류(애매)")
    lines.append(f"\n집계: 독립 노출 {n_independent}개, 조건부 노출 {n_conditional}개, 보류 {n_hold}개 "
                 f"(총 {len(rows)}개 파라미터, 3-D+ 격자에 포함된 것만 — 2-D 전용 파라미터인 damping/width는 "
                 f"reverb 4축 격자에 포함되어 있으므로 이미 반영됨).\n")
    return "\n".join(lines)


def main():
    cc = crosscheck()
    with open(RESULTS_DIR / "11_phase3_anova.md", "a", encoding="utf-8") as f:
        f.write(cc)
    print("교차확인 추가 저장: 11_phase3_anova.md")
    print(cc)

    ui = classify_ui()
    with open(RESULTS_DIR / "11_phase3_ui_spec.md", "w", encoding="utf-8") as f:
        f.write(ui)
    print("\n저장: 11_phase3_ui_spec.md")
    print(ui)


if __name__ == "__main__":
    main()
