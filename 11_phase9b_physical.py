"""CLAP FX Probe — 11_phase9b_physical.py (Phase 9b: 물리 지표 재검증)

11_phase9_physical.py는 두 가지 결함이 있었다.
  결함23: dry 기준값을 train_idx 1,200개 평균 하나로 고정했다. 서로 다른 음색의
    질의를 전부 같은 스칼라와 비교하면, 왜곡/잔향 자체의 효과가 음색 간 분산에
    묻혀 R0/R1 판정이 지표별로 뒤집혔다(THD·tail_ratio 등). 소스별 자기 자신의
    dry 값을 기준으로 삼아야 그 소스 안에서 "더 dry에 가까워졌는가"를 잰다.
  결함24: reverb는 tail_ratio/edt_proxy 두 지표뿐이었다. C50/C80(초기/후기 에너지
    비, 각각 50/80ms 경계 — 표준 명료도 지표의 변형)을 추가해 반응성을 다시 본다.

지표가 "왜곡/잔향이 세졌다"를 재는 유효한 프록시이려면, 질의 자신의 지표가
레벨 12→18→24로 갈수록 자기 dry로부터 단조 증가해야 한다(당연히 그래야 하는
방향 검증, 사후에 기준을 낮추지 않는다). 게이트 = (점추정 단조 증가) AND
(24-vs-12 차이의 대응 부트스트랩 95% CI 하한 > 0). 이 게이트를 통과 못 하는
지표는 R0/R1 비교에 쓸 자격이 없다 — 여기서는 게이트 결과만 보고하고 멈춘다.
"""
import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np

r0mod = import_module("11_phase9_retrieval")
r1mod = import_module("11_phase9_r1")
m2mod = import_module("11_phase9_m2m3")
r2mod = import_module("11_phase2_render")
physmod = import_module("11_phase9_physical")

unit = r0mod.unit
paired_bootstrap_diff = r0mod.paired_bootstrap_diff


def c_metric(y, sr, note_off, boundary):
    """boundary(s) 이전/이후 에너지비(dB). 늦은 창이 0.1s 미만이면 NaN."""
    dur = len(y) / sr
    post_end = min(note_off + 1.0, dur)
    if post_end - (note_off + boundary) < 0.1:
        return float("nan")
    s_e, e_e = int(note_off * sr), int((note_off + boundary) * sr)
    s_l, e_l = e_e, int(post_end * sr)
    e_early = float(np.sum(y[s_e:e_e] ** 2))
    e_late = float(np.sum(y[s_l:e_l] ** 2))
    return float(10 * np.log10((e_early + 1e-20) / (e_late + 1e-20)))


def add_c_metrics(dry_rev, rev_metrics, test_set):
    """dry_rev/rev_metrics(11_phase9_physical.render_all 결과)에 c50/c80을 덧붙인다."""
    sources = (json.load(open("out/results/11_phase2_sources.json"))["sources"]
               + json.load(open("out/results/11_phase2_sources_ext.json"))["sources"])
    fname_of = {s["src_id"]: s["filename"] for s in sources}
    rev_axis = r2mod.AXES[physmod.REV_AXIS]

    t0 = time.time()
    for src_id in range(1200):
        y = r2mod.embed_mod.load_and_preprocess(r2mod.AUDIO_DIR / fname_of[src_id])
        note_off = physmod.detect_note_off(y, r2mod.SR)
        dry_rev[src_id]["c50"] = c_metric(y, r2mod.SR, note_off, 0.05)
        dry_rev[src_id]["c80"] = c_metric(y, r2mod.SR, note_off, 0.08)

        levels = physmod.LIB_LEVELS + (12,) if src_id in test_set else physmod.LIB_LEVELS
        for lvl in levels:
            wet_r = rev_axis["board_fn"](rev_axis["levels"][lvl])(y, r2mod.SR)
            rev_metrics[lvl][src_id]["c50"] = c_metric(wet_r, r2mod.SR, note_off, 0.05)
            rev_metrics[lvl][src_id]["c80"] = c_metric(wet_r, r2mod.SR, note_off, 0.08)
        if src_id % 200 == 0:
            print(f"  C50/C80 진행 {src_id}/1200 ({time.time() - t0:.0f}s)")
    print(f"C50/C80 계산 완료: {time.time() - t0:.0f}s")


def gate_check(cfg, key, test_idx, n_boot=1000, seed=0):
    """소스별(fix1) dry로부터의 거리가 12→18→24 단조 증가 + CI 하한>0 인가."""
    ref = np.array([cfg["dry"][s][key] for s in test_idx])
    d = {}
    for lvl in (12, 18, 24):
        raw = np.array([cfg["metrics"][lvl][s][key] for s in test_idx])
        d[lvl] = np.abs(raw - ref)
    valid = ~(np.isnan(d[12]) | np.isnan(d[18]) | np.isnan(d[24]))
    n_valid = int(valid.sum())
    means = {lvl: float(np.mean(d[lvl][valid])) for lvl in (12, 18, 24)}
    monotonic = means[18] > means[12] and means[24] > means[18]
    lo, hi = paired_bootstrap_diff(d[24][valid], d[12][valid], n_boot=n_boot, seed=seed)
    return {
        "n_valid": n_valid, "means": means, "monotonic": bool(monotonic),
        "ci_24_minus_12": [lo, hi], "passed": bool(monotonic and lo > 0),
    }


def verdict(axis, cfg, test_idx, train_idx, val_idx, bypass_arr, family_arr, group_of, r1_prior):
    """게이트를 통과한 지표에 한해, 소스별(fix1) dry 기준으로 R0/R1 판정을 다시 낸다."""
    emb, theta = r0mod.load_axis(axis)
    model = r1mod.train_b2(axis, emb, bypass_arr, train_idx, val_idx)
    lib, lib_src, lib_is_dry, lib_family = m2mod.build_m2_library(axis, emb, bypass_arr, family_arr)

    out = {"levels": {}}
    for lvl in r0mod.QUERY_LEVELS:
        e_wet = emb[:, lvl, :]
        v_hat = unit(r1mod.predict_direction(model, e_wet))
        alpha = r1_prior[axis][str(lvl)]["R1"]["alpha"]

        pos_r0 = physmod.top1_lib_pos(test_idx, e_wet[test_idx], lib, lib_src, group_of)
        q_r1 = unit(e_wet[test_idx] + alpha * v_hat[test_idx])
        pos_r1 = physmod.top1_lib_pos(test_idx, q_r1, lib, lib_src, group_of)

        out["levels"][lvl] = {"alpha": alpha, "metrics": {}}
        for key in cfg["keys"]:
            ref = np.array([cfg["dry"][s][key] for s in test_idx])
            query_score = np.array([cfg["metrics"][lvl][s][key] for s in test_idx])
            r0_score = physmod.lib_pos_to_score(pos_r0, lib_src, cfg["dry"], cfg["metrics"][18], cfg["metrics"][24], key)
            r1_score = physmod.lib_pos_to_score(pos_r1, lib_src, cfg["dry"], cfg["metrics"][18], cfg["metrics"][24], key)

            valid = ~(np.isnan(ref) | np.isnan(query_score) | np.isnan(r0_score) | np.isnan(r1_score))
            d_query = np.abs(query_score[valid] - ref[valid])
            d_r0 = np.abs(r0_score[valid] - ref[valid])
            d_r1 = np.abs(r1_score[valid] - ref[valid])

            ci_vs_query = paired_bootstrap_diff(d_query, d_r1)
            ci_vs_r0 = paired_bootstrap_diff(d_r0, d_r1)
            out["levels"][lvl]["metrics"][key] = {
                "n_valid": int(valid.sum()),
                "query_dist_to_dry_mean": float(d_query.mean()),
                "R0_dist_to_dry_mean": float(d_r0.mean()),
                "R1_dist_to_dry_mean": float(d_r1.mean()),
                "R1_vs_query_ci": list(ci_vs_query),
                "R1_vs_R0_ci": list(ci_vs_r0),
            }
    return out


def fmt_ci_verdict(ci):
    lo, hi = ci
    if lo > 0:
        return "R1 우세"
    if hi < 0:
        return "R1 열세"
    return "유의차 없음"


def write_posthoc_md(gate, new_results, old_results, path):
    lines = []
    lines.append("# Phase 9b: 물리 지표 사후 재검증 (M2 R0/R1)\n")
    lines.append("## 결함 로그\n")
    lines.append("- **결함23**: `11_phase9_physical.py`의 dry 기준값이 `train_idx` 1,200개의 "
                  "전역 평균 하나였다. 서로 다른 음색의 질의를 모두 같은 스칼라와 비교하면 "
                  "왜곡/잔향 자체의 효과가 음색 간 분산에 묻혀 THD·tail_ratio 등에서 "
                  "R0/R1 판정이 뒤집혔다. 소스별 자기 자신의 dry 값을 기준으로 교체(fix1).\n")
    lines.append("- **결함24**: reverb 지표가 tail_ratio·edt_proxy 두 개뿐이라 sanity 폭이 좁았다. "
                  "C50/C80(50/80ms 경계 초기·후기 에너지비)을 추가(fix3).\n")
    lines.append("- **유효성 게이트(fix2)**: 질의 자신의 소스별 dry로부터의 거리가 레벨 12→18→24로 "
                  "단조 증가하고, 24-vs-12 차이의 대응 부트스트랩 95% CI 하한이 0을 넘어야 그 지표를 "
                  "R0/R1 비교에 쓴다. 게이트 기준은 사후에 낮추지 않았다.\n")

    lines.append("## 1. 유효성 게이트 결과\n")
    lines.append("| axis | metric | d(dry,12) | d(dry,18) | d(dry,24) | monotonic | CI(24-12) | gate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for axis, metrics in gate.items():
        for key, g in metrics.items():
            tag = "PASS" if g["passed"] else "FAIL"
            lines.append(f"| {axis} | {key} | {g['means'][12]:.4f} | {g['means'][18]:.4f} | "
                          f"{g['means'][24]:.4f} | {g['monotonic']} | "
                          f"[{g['ci_24_minus_12'][0]:.4f}, {g['ci_24_minus_12'][1]:.4f}] | **{tag}** |")

    lines.append("\n## 2. 판정 재산출 (게이트 통과 지표만, 소스별 dry 기준)\n")
    lines.append("컬럼: 기존 판정(전역 train 평균 기준, `11_phase9_physical.json`) vs 신규 판정"
                  "(소스별 기준, fix1). '판정' = (질의-R1) CI 부호 기준, R1이 질의보다 dry에 유의하게 "
                  "가까우면 'R1 우세'.\n")
    for axis in gate:
        lines.append(f"\n### {axis}\n")
        for lvl in (12, 18, 24):
            lines.append(f"**lvl={lvl}**\n")
            lines.append("| metric | gate | 기존 판정 | 기존 CI(질의-R1) | 신규 판정 | 신규 CI(질의-R1) | 비고 |")
            lines.append("|---|---|---|---|---|---|---|")
            for key in gate[axis]:
                gate_pass = gate[axis][key]["passed"]
                old_m = old_results[axis]["levels"][str(lvl)]["metrics"].get(key)
                new_m = new_results[axis]["levels"][lvl]["metrics"][key]
                new_verdict = fmt_ci_verdict(new_m["R1_vs_query_ci"])
                if old_m is None:
                    old_verdict, old_ci = "(없음, C50/C80은 신규 지표)", "-"
                else:
                    old_verdict = fmt_ci_verdict(old_m["R1_vs_query_ci"])
                    old_ci = f"[{old_m['R1_vs_query_ci'][0]:.4f}, {old_m['R1_vs_query_ci'][1]:.4f}]"
                new_ci = f"[{new_m['R1_vs_query_ci'][0]:.4f}, {new_m['R1_vs_query_ci'][1]:.4f}]"
                note = ""
                if old_m is not None and old_verdict != new_verdict:
                    note = f"판정 반전 ({old_verdict} → {new_verdict}), 원인: 결함23(전역 평균 기준 왜곡)"
                gate_tag = "PASS" if gate_pass else "FAIL(제외)"
                lines.append(f"| {key} | {gate_tag} | {old_verdict} | {old_ci} | {new_verdict} | {new_ci} | {note} |")

    Path(path).write_text("\n".join(lines) + "\n")


def main():
    bypass_arr = r0mod.load_bypass()
    family_arr = r0mod.load_family_array()
    group_of = r0mod.load_dup_groups()
    train_idx, val_idx, test_idx = r0mod.stratified_split(family_arr, seed=r0mod.SEED)
    test_set = set(test_idx.tolist())

    dry_dist, dry_rev, dist_metrics, rev_metrics = physmod.render_all(test_set)
    add_c_metrics(dry_rev, rev_metrics, test_set)

    axis_cfg = {
        physmod.DIST_AXIS: {"metrics": dist_metrics, "dry": dry_dist,
                             "keys": ["thd", "high_ratio", "crest", "centroid"]},
        physmod.REV_AXIS: {"metrics": rev_metrics, "dry": dry_rev,
                            "keys": ["tail_ratio", "edt_proxy", "c50", "c80"]},
    }

    gate = {}
    print("\n=== 유효성 게이트: 질의 자신의 (소스별) dry로부터의 거리가 12→18→24 단조 증가하고,"
          " 24-vs-12 CI 하한 > 0 인가 ===")
    for axis, cfg in axis_cfg.items():
        gate[axis] = {}
        print(f"\n--- {axis} ---")
        for key in cfg["keys"]:
            g = gate_check(cfg, key, test_idx)
            gate[axis][key] = g
            tag = "PASS" if g["passed"] else "FAIL"
            print(f"  [{tag}] {key:<10} n_valid={g['n_valid']}/{len(test_idx)}  "
                  f"d(dry,12)={g['means'][12]:.4f}  d(dry,18)={g['means'][18]:.4f}  "
                  f"d(dry,24)={g['means'][24]:.4f}  monotonic={g['monotonic']}  "
                  f"CI(24-12)=[{g['ci_24_minus_12'][0]:.4f}, {g['ci_24_minus_12'][1]:.4f}]")

    Path("out/results").mkdir(parents=True, exist_ok=True)
    with open("out/results/11_phase9b_gate.json", "w") as f:
        json.dump(gate, f, indent=2)
    print("\n저장: out/results/11_phase9b_gate.json")

    r1_prior = json.load(open("out/results/11_phase9_r1.json"))
    new_results = {}
    print("\n=== 판정 재산출 (게이트 통과 지표, 소스별 dry 기준) ===")
    for axis, cfg in axis_cfg.items():
        new_results[axis] = verdict(axis, cfg, test_idx, train_idx, val_idx, bypass_arr, family_arr, group_of, r1_prior)
        for lvl in (12, 18, 24):
            print(f"  {axis} lvl={lvl}:")
            for key, m in new_results[axis]["levels"][lvl]["metrics"].items():
                v = fmt_ci_verdict(m["R1_vs_query_ci"])
                print(f"    {key:<10} n_valid={m['n_valid']}  query={m['query_dist_to_dry_mean']:.4f}  "
                      f"R0={m['R0_dist_to_dry_mean']:.4f}  R1={m['R1_dist_to_dry_mean']:.4f}  "
                      f"(질의-R1)CI={[round(c, 4) for c in m['R1_vs_query_ci']]}  판정={v}")

    with open("out/results/11_phase9b_physical.json", "w") as f:
        json.dump(new_results, f, indent=2)
    print("\n저장: out/results/11_phase9b_physical.json")

    old_results = json.load(open("out/results/11_phase9_physical.json"))
    write_posthoc_md(gate, new_results, old_results, "out/results/11_phase9b_physical_posthoc.md")
    print("저장: out/results/11_phase9b_physical_posthoc.md")


if __name__ == "__main__":
    main()
