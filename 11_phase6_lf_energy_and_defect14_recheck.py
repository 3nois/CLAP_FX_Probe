# -*- coding: utf-8 -*-
"""Phase 6 준비 — 저주파(<=200Hz) 에너지 공변량 산출 + 결함14 기전 재확인(1200소스 규모).

사용자 지시(2026-08-22): 소스별 200Hz 이하 대역 RMS를 한 번 계산해 저장한다
(out/results/11_source_lf_energy.json). 두 곳에 쓴다:
  (a) Phase 6 lowshelf 결과 해석 — 저역 에너지가 없는 소스는 효과가 원천적으로
      작다. Phase 6 스크립트가 이 파일을 읽어 상관을 보고한다.
  (b) 결함 14(IIR biquad neutral 실패, 원래 17/400) 기전 확인 — 걸린 소스가
      저역 에너지 상위에 몰렸는지. 원래 분석(11_phase2_integrity_v2.py 확인 2)은
      400소스 규모·이분법(문제/비문제) 검정이었다 — 여기서는 현재 1200소스
      전체로 다시 확인하고, 연속 상관(Spearman)도 함께 낸다.
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
dr = import_module("11_phase2_doseresponse")
embed_mod = import_module("01_embed")

RESULTS_DIR = render_mod.RESULTS_DIR
AUDIO_DIR = render_mod.AUDIO_DIR


def load_bypass_1200():
    base = np.load(render_mod.CACHE_DIR / "11_phase2_bypass.npz")
    ext = np.load(render_mod.CACHE_DIR / "11_phase2ext_bypass.npz")
    emb = np.concatenate([base["embeddings"], ext["embeddings"]], axis=0)
    src_id = np.concatenate([base["src_id"], ext["src_id"]])
    order = np.argsort(src_id)
    return emb[order], src_id[order]


def all_sources_1200():
    with open(RESULTS_DIR / "11_phase2_sources.json", encoding="utf-8") as f:
        base_sources = json.load(f)["sources"]
    with open(RESULTS_DIR / "11_phase2_sources_ext.json", encoding="utf-8") as f:
        ext_sources = json.load(f)["sources"]
    return sorted(base_sources + ext_sources, key=lambda s: s["src_id"])


def main():
    sources = all_sources_1200()
    assert [s["src_id"] for s in sources] == list(range(1200))
    print(f"소스 {len(sources)}개, 저역(<=200Hz) RMS 계산 중...")

    lf_energy = {}
    for s in sources:
        y = embed_mod.load_and_preprocess(AUDIO_DIR / s["filename"])
        y_lp = pb.Pedalboard([pb.LowpassFilter(cutoff_frequency_hz=200.0)])(y, render_mod.SR)
        lf_energy[s["src_id"]] = float(np.sqrt(np.mean(y_lp ** 2)))
    print("계산 완료.")

    with open(RESULTS_DIR / "11_source_lf_energy.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": "pedalboard.LowpassFilter(200Hz) 적용 후 RMS, 01_embed.load_and_preprocess 전처리(48kHz/4s/peak0.7)",
            "n_sources": len(sources),
            "lf_rms_by_src_id": {str(k): v for k, v in lf_energy.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"저장: {RESULTS_DIR / '11_source_lf_energy.json'}")

    # ---- 결함 14 기전 재확인 (1200소스) ----
    print("\n결함14 재확인 (1200소스, lowshelf_gain neutral 잔차 vs 저역 에너지)...")
    bypass_emb, bypass_src = load_bypass_1200()
    emb, theta_raw, src_id = dr.load_concat("lowshelf_gain")
    assert np.array_equal(bypass_src, src_id)
    idx0 = dr.theta_min_index("lowshelf_gain", theta_raw)
    emb0 = emb[:, idx0, :]
    d_neutral = 1.0 - dr.cos_rows(bypass_emb, emb0)

    lf_arr = np.array([lf_energy[i] for i in src_id])
    rho, p_rho = stats.spearmanr(d_neutral, lf_arr)

    threshold = 1e-4  # 11_phase2_integrity_v2.py와 동일 기준(cos<0.9999)
    problem_mask = d_neutral > threshold
    n_problem = int(problem_mask.sum())
    problem_rms = lf_arr[problem_mask]
    other_rms = lf_arr[~problem_mask]
    if n_problem > 0 and (~problem_mask).sum() > 0:
        _, p_greater = stats.mannwhitneyu(problem_rms, other_rms, alternative="greater")
    else:
        p_greater = None

    high_lf_mask = lf_arr >= np.percentile(lf_arr, 80)
    n_problem_in_high_lf = int((problem_mask & high_lf_mask).sum())
    clustered_in_high_lf = n_problem > 0 and (n_problem_in_high_lf / n_problem) > 0.5

    lines = ["# 결함 14 기전 재확인 (1200소스, 2026-08-22)\n"]
    lines.append("원 분석(`11_phase2_integrity_v2.py` 확인 2)은 400소스 규모에서 이분법 검정으로 "
                 "\"문제 소스가 저역 에너지 高\"라는 가설을 **기각**(반대 방향)했다. 여기서는 현재 "
                 "1200소스 전체로 연속 상관(Spearman)과 상위20%-저역에너지 군집 여부를 다시 확인한다.\n")
    lines.append(f"- 문제 소스(cos<0.9999, threshold d_neutral>{threshold:.0e}) N={n_problem}/{len(src_id)}\n")
    lines.append(f"- Spearman rho(d_neutral, 저역RMS) = {rho:.4f} (p={p_rho:.2e})\n")
    if p_greater is not None:
        lines.append(f"- Mann-Whitney U 단측(문제군 저역RMS > 나머지, 가설 방향): p={p_greater:.2e}\n")
    lines.append(f"- 문제 소스 중 저역에너지 상위20% 안에 든 비율: {n_problem_in_high_lf}/{n_problem} "
                 f"({(n_problem_in_high_lf/n_problem*100 if n_problem else float('nan')):.1f}%)\n")
    lines.append(f"\n**판정**: {'저역 에너지 상위에 몰림 — 기전 확정(가설 방향 확인)' if clustered_in_high_lf else '몰리지 않음(또는 반대 방향) — 기전 미확인으로 유지'}\n")

    out_path = RESULTS_DIR / "11_phase6_defect14_recheck.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {out_path}")
    print(f"rho={rho:.4f} p={p_rho:.2e}  문제소스 N={n_problem}  상위20% 비율={n_problem_in_high_lf}/{n_problem}")
    print("결론:", "기전 확정" if clustered_in_high_lf else "기전 미확인 유지")


if __name__ == "__main__":
    main()
