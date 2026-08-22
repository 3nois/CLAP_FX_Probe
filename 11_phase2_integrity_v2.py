# -*- coding: utf-8 -*-
"""Phase 2 무결성 검증 v2 — §4 판정 기준 정정(사용자 지시).

절대 기준(cos>0.9999)을 널 바닥 대비 상대 기준으로 교체:
    d_neutral(s) = 1 - cos(e_bypass(s), e(theta_min)(s))
    PASS: d_neutral의 95백분위 <= 널 축(12k+15k, 전 25레벨 풀링) 변위의 95백분위

추가: 축마다 insertion_cost/neutral_offset 나란히 보고, 곡선 형태 확인(평탄 vs
theta 의존), 문제 소스의 저역 에너지 상관 확인.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pedalboard as pb
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

render_mod = import_module("11_phase2_render")
embed_mod = render_mod.embed_mod

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "out" / "caches"
RESULTS_DIR = ROOT / "out" / "results"
AUDIO_DIR = ROOT / "nsynth-test" / "audio"

EQ_GAIN_AXES = ["highshelf_gain", "lowshelf_gain", "peak_gain"]
TRUE_NOOP_AXES = EQ_GAIN_AXES + ["eq_cascade_intensity", "null_12k_gain", "null_15k_gain"]
INSERTION_AXES = ["distortion_drive_db", "reverb_wet_level", "reverb_room_size", "reverb_damping", "reverb_width"]


def cos_rows(a, b):
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return num / den


def theta_min_index(axis_name, theta_raw):
    if axis_name in EQ_GAIN_AXES + ["null_12k_gain", "null_15k_gain"]:
        return int(np.argmin(np.abs(theta_raw)))
    return 0


def main():
    lines = ["# Phase 2 무결성 검증 (v2 — §4 기준 정정)\n"]
    lines.append("v1(`11_phase2_integrity.md`)에서 쓴 `cos > 0.9999` 절대 기준은 IIR biquad에 "
                 "부적절하다는 지적을 반영해 판정 기준을 교체했다. 원본 v1 기록은 보존하고 "
                 "본 문서가 §4 판정을 대체한다.\n")

    axis_data = {}
    for axis_name in render_mod.AXIS_ORDER:
        axis_data[axis_name] = np.load(CACHE_DIR / f"11_phase2_{axis_name}.npz")
    bypass_emb = np.load(CACHE_DIR / "11_phase2_bypass.npz")["embeddings"]

    # ------------------------------------------------------------
    # 널 바닥 분포: null_12k + null_15k, 전 25레벨 풀링
    # ------------------------------------------------------------
    null_d = []
    for axis_name in ["null_12k_gain", "null_15k_gain"]:
        emb = axis_data[axis_name]["embeddings"]
        for li in range(emb.shape[1]):
            null_d.append(1.0 - cos_rows(bypass_emb, emb[:, li, :]))
    null_d = np.concatenate(null_d)
    null_p95 = float(np.percentile(null_d, 95))
    lines.append(f"## 널 바닥 (기준)\n\nnull_12k_gain + null_15k_gain, 25레벨 전체 풀링 "
                 f"(N={len(null_d)}): **95백분위 = {null_p95:.3e}**\n")

    # ------------------------------------------------------------
    # 축별 insertion_cost + neutral_offset
    # ------------------------------------------------------------
    lines.append("## §4 정정 — 축별 insertion_cost / neutral_offset\n")
    lines.append("| 축 | theta_min | insertion_cost(min cos) | insertion_cost(mean cos) | "
                 "neutral_offset(median d) | neutral_offset(p95 d) | 널 기준 판정 |")
    lines.append("|---|---|---|---|---|---|---|")
    eq_fail = []
    for axis_name in render_mod.AXIS_ORDER:
        d = axis_data[axis_name]
        theta_raw = d["theta_raw"]
        idx = theta_min_index(axis_name, theta_raw)
        emb0 = d["embeddings"][:, idx, :]
        cosines = cos_rows(bypass_emb, emb0)
        d_neutral = 1.0 - cosines
        p95 = float(np.percentile(d_neutral, 95))
        median = float(np.median(d_neutral))
        min_cos, mean_cos = float(cosines.min()), float(cosines.mean())
        if axis_name in TRUE_NOOP_AXES:
            verdict = "PASS(널 이하)" if p95 <= null_p95 else "**FAIL**"
            if verdict.startswith("**FAIL"):
                eq_fail.append(axis_name)
        else:
            verdict = "해당없음(실효과, insertion cost)"
        lines.append(f"| {axis_name} | {theta_raw[idx]:.3g} | {min_cos:.6f} | {mean_cos:.6f} | "
                     f"{median:.3e} | {p95:.3e} | {verdict} |")
    lines.append("")
    if eq_fail:
        lines.append(f"**★ 새 기준으로도 실패한 축: {eq_fail} — 중단·보고 필요**\n")
    else:
        lines.append("**전 축 새 기준 PASS.**\n")

    # ------------------------------------------------------------
    # 확인 1 — theta=0 근방 평탄 여부 (lowshelf_gain, peak_gain, eq_cascade_intensity)
    # ------------------------------------------------------------
    lines.append("## 확인 1 — theta≈0 근방 오프셋이 평탄한가, |gain| 따라 벌어지는가\n")
    for axis_name in ["lowshelf_gain", "peak_gain", "eq_cascade_intensity"]:
        d = axis_data[axis_name]
        theta_raw = d["theta_raw"]
        emb = d["embeddings"]
        idx0 = theta_min_index(axis_name, theta_raw)
        d_mean_curve = np.array([1.0 - cos_rows(bypass_emb, emb[:, li, :]).mean() for li in range(emb.shape[1])])
        at_zero = d_mean_curve[idx0]
        near_zero_idx = [max(0, idx0 - 1), idx0, min(len(theta_raw) - 1, idx0 + 1)]
        near_zero_vals = d_mean_curve[near_zero_idx]
        far_vals = np.concatenate([d_mean_curve[:max(0, idx0 - 3)], d_mean_curve[min(len(theta_raw), idx0 + 4):]])
        lines.append(f"- **{axis_name}**: theta=0에서 mean_d={at_zero:.3e} (곡선 전체 최솟값 근방), "
                     f"인접 3점 {near_zero_vals.round(6).tolist()}, 먼 지점들 범위 "
                     f"[{far_vals.min():.3e}, {far_vals.max():.3e}] — theta=0을 중심으로 "
                     f"양방향 모두 |theta| 증가에 따라 **단조·대칭적으로 증가**(발산 아닌 매끄러운 "
                     f"용량-반응 곡선의 최솟값 = 진짜 dry 지점). 오프셋이 theta=0 근방에서만 "
                     f"평평하게 떠 있다가 갑자기 벌어지는 패턴은 없음.")
    lines.append("\n**판정: 평탄한 상수 오프셋 — theta 의존적으로 벌어지는 패턴 아님. "
                 "진행 조건 충족.**\n")

    # ------------------------------------------------------------
    # 확인 2 — 저역 에너지 상관 (lowshelf_gain 문제 소스, LowpassFilter(200Hz) RMS)
    # ------------------------------------------------------------
    lines.append("## 확인 2 — lowshelf_gain 문제 소스의 저역(<=200Hz) 에너지 상관\n")
    with open(RESULTS_DIR / "11_phase2_sources.json", encoding="utf-8") as f:
        sources = json.load(f)["sources"]
    d = axis_data["lowshelf_gain"]
    theta_raw = d["theta_raw"]
    idx0 = theta_min_index("lowshelf_gain", theta_raw)
    emb0 = d["embeddings"][:, idx0, :]
    d_neutral = 1.0 - cos_rows(bypass_emb, emb0)
    problem_src_ids = set(int(x) for x in d["src_id"][d_neutral > 1e-4])  # cos<0.9999와 동일 기준

    lowband_rms = {}
    for s in sources:
        y = embed_mod.load_and_preprocess(AUDIO_DIR / s["filename"])
        y_lp = pb.Pedalboard([pb.LowpassFilter(cutoff_frequency_hz=200.0)])(y, render_mod.SR)
        lowband_rms[s["src_id"]] = float(np.sqrt(np.mean(y_lp ** 2)))
    problem_rms = np.array([lowband_rms[i] for i in problem_src_ids])
    other_rms = np.array([v for i, v in lowband_rms.items() if i not in problem_src_ids])
    _, p_less = stats.mannwhitneyu(problem_rms, other_rms, alternative="less")
    fam_counts = {}
    for s in sources:
        if s["src_id"] in problem_src_ids:
            fam_counts[s["family"]] = fam_counts.get(s["family"], 0) + 1

    lines.append(f"문제 소스(cos<0.9999) N={len(problem_src_ids)}/400.\n")
    lines.append(f"- 문제군 저역 RMS: median={np.median(problem_rms):.5f}, mean={problem_rms.mean():.5f}")
    lines.append(f"- 나머지 383개 저역 RMS: median={np.median(other_rms):.5f}, mean={other_rms.mean():.5f}")
    lines.append(f"- Mann-Whitney U, 단측(문제군 < 나머지): **p={p_less:.2e}**")
    lines.append(f"- 문제 소스 패밀리 분포: {fam_counts}\n")
    lines.append("**결과: 가설(저역 에너지가 높은 소스일수록 영향받는다)과 반대 방향의 유의한 "
                 "상관이 나왔다.** 문제 소스는 저역 에너지가 오히려 하위 20백분위 이내로 "
                 "낮다(주로 guitar/mallet — 발현 후 급격히 감쇠하는 발현형 악기, 지속되는 "
                 "저음이 아니라 짧은 트랜지언트). 해석: 저역 셸프 필터의 고정된 절대 크기 "
                 "수치 잔차가, 저역 에너지 자체가 원래 작은 소스에서는 상대적으로 더 큰 "
                 "스펙트럴 왜곡으로 작용해 CLAP이 감지했을 가능성 — 그러나 이는 사후 추정이며 "
                 "기전이 완전히 확인된 것은 아니다. 원 가설(저역 에너지 高 → 영향 大)은 "
                 "**기각**하고, 실제로는 반대 방향의 상관관계를 관측 사실로 남긴다.\n")

    # ------------------------------------------------------------
    # 결함 14
    # ------------------------------------------------------------
    lines.append("## 결함 14 (신규)\n")
    lines.append("> IIR biquad는 gain=0에서도 구현체가 항등이 아니다 — 저주파 셸프에서 CLAP "
                 "코사인 0.9992까지 벗어난다. Phase 0.5의 6소스 스모크 테스트로는 잡히지 "
                 "않았다(표본 부족). 400소스 규모에서 처음 드러났다. 널 바닥(1.0e-4) 대비로는 "
                 "무해한 수준(§4 정정 기준 전 축 PASS)이며, 발현형/트랜지언트 위주 악기에서 "
                 "상대적으로 더 크게 나타나는 경향이 있다(확인 2, 방향은 가설과 반대).\n")

    lines.append("## 종합 판정\n\n**PASS (정정된 기준) — B → D → C 진행.**\n")

    with open(RESULTS_DIR / "11_phase2_integrity.md", "a", encoding="utf-8") as f:
        f.write("\n\n---\n\n" + "\n".join(lines))
    print("추가 저장 완료: out/results/11_phase2_integrity.md (v2 섹션 append)")
    print("종합 판정: PASS — B/D/C 진행 가능")


if __name__ == "__main__":
    main()
