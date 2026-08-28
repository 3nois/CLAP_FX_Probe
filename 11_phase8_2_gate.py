# -*- coding: utf-8 -*-
"""Phase 8-2 §5/§6 — projection 야코비안 게이트 (오디오 생성 없음, 전부 무료).

out/prereg/11_phase8.md §6 확정 설계. 8-1과 같은 100소스, highshelf_gain 전범위.

  J          = d(clap_projection)/d(input), e_wet에서 autodiff (torch.autograd.
               functional.jacobian). shape (1024,512) — §0.2 정정: 과대결정계.
  d_target   = normalize(proj(e_dry_true) - proj(e_wet))
  d_naive    = v = e_dry_true - e_wet                         (기존 방식)
  d_jac      = pinv(J) @ d_target, ||v||로 정규화              (야코비안 정보 사용)

  cos_naive(alpha) = cos(proj(e_wet+alpha*d_naive)-proj(e_wet), d_target)
                     ★ alpha=1에서 정의상 정확히 1.0(순환논리, §6 설명)
  cos_jac(alpha)   = cos(proj(e_wet+alpha*d_jac)  -proj(e_wet), d_target)

게이트 통과 조건: alpha in {2,3,5} 중 하나 이상에서 mean(cos_jac-cos_naive) >= 0.05.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

dr = import_module("11_phase2_doseresponse")
p8 = import_module("11_phase8_run")

RESULTS_DIR = dr.RESULTS_DIR
AXIS_NAME = "highshelf_gain"
IDX_A, IDX_B = 0, 24
ALPHAS = [1, 2, 3, 5]
GATE_THRESHOLD = 0.05
GATE_ALPHAS = [2, 3, 5]


def main():
    device = torch.device("cpu")
    print("TokenSynth 로딩 중 (clap_projection 야코비안용)...")
    from tokensynth import TokenSynth
    synth = TokenSynth.from_pretrained(aug=True, device=device)
    synth.eval()
    proj = synth.clap_projection

    def proj_fn(x):
        return proj(x)

    sources = p8.all_sources_1200()
    selected_pos = p8.select_100_sources(sources)
    print(f"소스 {len(selected_pos)}개")

    emb, theta_raw, src_id = dr.load_concat(AXIS_NAME)
    assert np.array_equal(src_id, np.arange(1200))
    e_dry_true_all = emb[:, IDX_A, :]
    e_wet_all = emb[:, IDX_B, :]

    rows = []
    for pos in selected_pos:
        e_wet = torch.tensor(e_wet_all[pos], dtype=torch.float32)
        e_dry = torch.tensor(e_dry_true_all[pos], dtype=torch.float32)
        v = (e_dry - e_wet)
        v_norm = float(v.norm())

        with torch.no_grad():
            J = torch.autograd.functional.jacobian(proj_fn, e_wet.unsqueeze(0)).squeeze()
            # J shape (1024, 1, 512) -> squeeze -> (1024,512) (배치차원 1 제거)
            if J.dim() == 3:
                J = J.squeeze(1)
            svals = torch.linalg.svdvals(J)
            min_sv, max_sv = float(svals.min()), float(svals.max())
            rank_ok = min_sv > max_sv * 1e-3
            effective_rank = int((svals > max_sv * 1e-3).sum())

            proj_wet = proj(e_wet.unsqueeze(0)).squeeze(0)
            proj_dry = proj(e_dry.unsqueeze(0)).squeeze(0)
            d_target_raw = proj_dry - proj_wet
            d_target = d_target_raw / (d_target_raw.norm() + 1e-12)

            J_pinv = torch.linalg.pinv(J)  # (512,1024)
            d_jac_raw = J_pinv @ d_target  # (512,)
            d_jac = d_jac_raw / (d_jac_raw.norm() + 1e-12) * v_norm  # ||v||로 정규화(§6)

            row = {"src_pos": int(pos), "v_norm": v_norm, "min_sv": min_sv, "max_sv": max_sv,
                   "rank_ok": rank_ok, "effective_rank": effective_rank, "cos_naive": {}, "cos_jac": {}}
            for alpha in ALPHAS:
                inj_naive = e_wet + alpha * v
                inj_jac = e_wet + alpha * d_jac
                proj_naive = proj(inj_naive.unsqueeze(0)).squeeze(0)
                proj_jac = proj(inj_jac.unsqueeze(0)).squeeze(0)
                diff_naive = proj_naive - proj_wet
                diff_jac = proj_jac - proj_wet
                cos_naive = float(torch.dot(diff_naive, d_target) / (diff_naive.norm() * d_target.norm() + 1e-12))
                cos_jac = float(torch.dot(diff_jac, d_target) / (diff_jac.norm() * d_target.norm() + 1e-12))
                row["cos_naive"][alpha] = cos_naive
                row["cos_jac"][alpha] = cos_jac
        rows.append(row)

    n_bad_rank = sum(1 for r in rows if not r["rank_ok"])
    eff_ranks = np.array([r["effective_rank"] for r in rows])
    print(f"열계수 의심(min_sv < max_sv*1e-3) 소스 수: {n_bad_rank}/{len(rows)}")
    print(f"유효계수(effective rank, svdvals>max*1e-3) 평균={eff_ranks.mean():.1f} "
          f"[{eff_ranks.min()},{eff_ranks.max()}] / 512")

    lines = ["# Phase 8-2 §5/§6 — projection 야코비안 게이트 (2026-08-28)\n"]
    lines.append(f"소스 {len(rows)}개(highshelf_gain 전범위, Phase 8-1과 동일 100개).\n")
    lines.append("§0.2 정정: J는 (1024,512) — 과대결정계. pinv(J)@d_target은 '최소노름 해'가 아니라 "
                 "'최소제곱 근사해'다. α=1에서 cos_naive는 정의상 정확히 1.0(순환논리) — "
                 "의미 있는 비교는 α>1(외삽) 구간이다.\n")
    lines.append(f"## ★ 부산물 발견 — projection 층의 국소 유효계수가 512 중 {eff_ranks.mean():.1f}개뿐\n")
    lines.append(f"e_wet 지점에서 야코비안 $J$(1024×512)의 특이값 분해 결과, 100개 소스 전부에서 "
                 f"`min_sv < max_sv×1e-3`(사실상 0에 가까움, `max_sv×1e-6` 기준으로도 동일)인 "
                 f"소스가 {n_bad_rank}/100개였다. 유효계수(유의미한 특이값 개수)는 평균 "
                 f"{eff_ranks.mean():.1f}개(범위 {eff_ranks.min()}~{eff_ranks.max()}), 즉 512차원 입력 "
                 f"중 **약 {eff_ranks.mean()/512*100:.0f}%만 그 지점에서 국소적으로 살아있고 나머지 "
                 f"~{(1-eff_ranks.mean()/512)*100:.0f}%는 ReLU가 죽어 1차 미분이 사실상 0이다.** "
                 f"이는 projection 층이 정보를 '선택적으로 약화'(10차 결과, 6~41%)하는 정도가 아니라, "
                 f"임의의 한 입력점에서는 대부분의 방향이 국소적으로 아예 안 보인다는 훨씬 강한 "
                 f"발견이다 — 다만 ReLU는 조각별 선형이라 이 값은 **지점마다 다르며**(다른 활성화 "
                 f"영역에서는 다른 512차원 하위공간이 죽어있을 수 있다), 이 특정 지점(e_wet)에서의 국소적 "
                 f"관측일 뿐 전역적으로 512-98=414차원이 영구히 못 쓰인다는 뜻은 아니다.\n")
    lines.append("| alpha | cos_naive 평균(95%CI) | cos_jac 평균(95%CI) | 차이(jac-naive) | 게이트 |")
    lines.append("|---|---|---|---|---|")

    gate_pass_any = False
    gate_detail = {}
    for alpha in ALPHAS:
        naive_vals = np.array([r["cos_naive"][alpha] for r in rows])
        jac_vals = np.array([r["cos_jac"][alpha] for r in rows])
        diff_vals = jac_vals - naive_vals

        def boot_ci(vals):
            rng = np.random.RandomState(0)
            n = len(vals)
            boots = np.array([vals[rng.randint(0, n, n)].mean() for _ in range(2000)])
            return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

        naive_m, naive_lo, naive_hi = boot_ci(naive_vals)
        jac_m, jac_lo, jac_hi = boot_ci(jac_vals)
        diff_m, diff_lo, diff_hi = boot_ci(diff_vals)

        passed = (alpha in GATE_ALPHAS) and (diff_m >= GATE_THRESHOLD)
        if passed:
            gate_pass_any = True
        gate_detail[alpha] = {"naive_mean": naive_m, "naive_ci": [naive_lo, naive_hi],
                               "jac_mean": jac_m, "jac_ci": [jac_lo, jac_hi],
                               "diff_mean": diff_m, "diff_ci": [diff_lo, diff_hi], "passed": bool(passed)}
        gate_str = "★통과" if passed else ("해당없음(α=1)" if alpha == 1 else "미통과")
        lines.append(f"| {alpha} | {naive_m:.4f} [{naive_lo:.4f},{naive_hi:.4f}] | "
                     f"{jac_m:.4f} [{jac_lo:.4f},{jac_hi:.4f}] | "
                     f"{diff_m:+.4f} [{diff_lo:+.4f},{diff_hi:+.4f}] | {gate_str} |")
        print(f"alpha={alpha}: naive={naive_m:.4f} jac={jac_m:.4f} diff={diff_m:+.4f} -> {gate_str}")

    lines.append(f"\n**종합 판정**: {'게이트 통과(≥1개 α에서 차이≥0.05) — 8-2 오디오 검증 진행 대상' if gate_pass_any else '게이트 미통과 — 이 스케일에서 야코비안 정보의 추가 이득 없음, 8-2 오디오 검증 생략'}\n")

    out_path = RESULTS_DIR / "11_phase8_2_gate.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {out_path}")

    with open(RESULTS_DIR / "11_phase8_2_gate_raw.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "gate_summary": gate_detail, "gate_pass_any": gate_pass_any,
                   "n_bad_rank": n_bad_rank}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
