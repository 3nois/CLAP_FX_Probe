# -*- coding: utf-8 -*-
"""Phase 2 완료 후 무결성 검증 (사용자 지시 §A).

렌더링 없이 캐시(out/caches/11_phase2_*.npz)만 읽어 검사한다. CLAP 불필요, 빠름.

1. 23개 축 캐시 파일 존재 확인
2. 각 파일 shape == (400, 25, 512)
3. bypass 앵커 400행 확인
4. neutral check: cos(e_bypass, e(theta_min))
     EQ gain 3축 + cascade      >0.9999 필수, 미달 시 즉시 중단·보고
     distortion, reverb 4축     ~0.90~0.98 (insertion cost로 이미 알려진 값)
     null 2축                   참고용(진짜 dry 기대)
5. 게이트 축 실측 — {highshelf,lowshelf,peak}_{cutoff,q}_gp6 6축이 null 축과
   구분되는 변위를 보이는지 (끝점 간 displacement, source-mean ± 95% CI)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

render_mod = import_module("11_phase2_render")

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "out" / "caches"
RESULTS_DIR = ROOT / "out" / "results"

EQ_GAIN_AXES = ["highshelf_gain", "lowshelf_gain", "peak_gain"]
INSERTION_AXES = ["distortion_drive_db", "reverb_wet_level", "reverb_room_size", "reverb_damping", "reverb_width"]
NULL_AXES = ["null_12k_gain", "null_15k_gain"]
GATE_CHECK_AXES = [f"{t}_{p}_gp6" for t in ["highshelf", "lowshelf", "peak"] for p in ["cutoff", "q"]]


def cos_rows(a, b):
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return num / den


def bootstrap_ci_mean(x, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(x)
    boots = np.array([np.mean(x[rng.randint(0, n, n)]) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    lines = ["# Phase 2 무결성 검증\n"]
    lines.append("검증 스크립트: `11_phase2_integrity.py`\n")
    ok_all = True

    # 1~2. 파일 존재 + shape
    lines.append("## 1~2. 축 파일 존재 + shape\n")
    lines.append("| 축 | 존재 | shape | 판정 |\n|---|---|---|---|")
    axis_data = {}
    for axis_name in render_mod.AXIS_ORDER:
        p = CACHE_DIR / f"11_phase2_{axis_name}.npz"
        exists = p.exists()
        if not exists:
            lines.append(f"| {axis_name} | ✗ | — | **FAIL** |")
            ok_all = False
            continue
        d = np.load(p, allow_pickle=True)
        emb = d["embeddings"]
        expected = (400, 25, 512)
        shape_ok = tuple(emb.shape) == expected
        if not shape_ok:
            ok_all = False
        axis_data[axis_name] = d
        lines.append(f"| {axis_name} | O | {tuple(emb.shape)} | {'OK' if shape_ok else '**FAIL**'} |")
    n_found = len(axis_data)
    lines.append(f"\n총 {n_found}/{len(render_mod.AXIS_ORDER)}개 축 확인.\n")

    # 3. bypass
    lines.append("## 3. bypass 앵커\n")
    bypass_path = CACHE_DIR / "11_phase2_bypass.npz"
    if not bypass_path.exists():
        lines.append("**FAIL — bypass 캐시 없음**\n")
        ok_all = False
        bypass_emb = None
    else:
        bd = np.load(bypass_path)
        bypass_emb = bd["embeddings"]
        n_bypass = bypass_emb.shape[0]
        bypass_ok = n_bypass == 400
        if not bypass_ok:
            ok_all = False
        lines.append(f"행 수 = {n_bypass} — {'OK' if bypass_ok else '**FAIL**'}\n")

    if not axis_data or bypass_emb is None:
        lines.append("\n★ 필수 파일 누락으로 이후 검사를 건너뜁니다.\n")
        with open(RESULTS_DIR / "11_phase2_integrity.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("★ FAIL — 필수 파일 누락. out/results/11_phase2_integrity.md 참고.")
        return

    # theta_min index per axis: reverb/distortion/cascade -> 0, EQ gain -> value==0 index, null -> value==0 index
    def theta_min_index(axis_name, theta_raw):
        if axis_name in EQ_GAIN_AXES + NULL_AXES:
            return int(np.argmin(np.abs(theta_raw)))
        return 0  # distortion, reverb 4축, cascade, cutoff/q(gp6/gn6) — 격자 첫 점

    # 4. neutral check
    lines.append("## 4. neutral check — cos(e_bypass, e(theta_min))\n")
    lines.append("| 축 | theta_min | min cos | mean cos | 기대 | 판정 |\n|---|---|---|---|---|---|")
    eq_fail = False
    check_axes = EQ_GAIN_AXES + ["eq_cascade_intensity"] + INSERTION_AXES + NULL_AXES
    for axis_name in check_axes:
        if axis_name not in axis_data:
            continue
        d = axis_data[axis_name]
        theta_raw = d["theta_raw"]
        idx = theta_min_index(axis_name, theta_raw)
        emb = d["embeddings"][:, idx, :]
        cosines = cos_rows(bypass_emb, emb)
        min_c, mean_c = float(cosines.min()), float(cosines.mean())
        if axis_name in EQ_GAIN_AXES + ["eq_cascade_intensity"] + NULL_AXES:
            expect = ">0.9999 (진짜 dry)"
            passed = min_c > 0.9999
            if axis_name in EQ_GAIN_AXES and not passed:
                eq_fail = True
                ok_all = False
        else:
            expect = "0.90~0.98 (insertion cost, 실패 아님)"
            passed = True
        lines.append(f"| {axis_name} | {theta_raw[idx]:.3g} | {min_c:.6f} | {mean_c:.6f} | {expect} | {'OK' if passed else '**★ FAIL**'} |")

    lines.append("")
    if eq_fail:
        lines.append("**★★★ EQ gain 축이 0.9999를 넘지 못했다 — 즉시 중단하고 보고할 것 (지시 §A.4).**\n")
        ok_all = False

    # 5. 게이트 축 실측
    lines.append("## 5. 게이트 축 vs 널 축 — 끝점 간 변위 비교\n")
    lines.append("게이트 축(gp6, gain=+6dB 고정에서 cutoff/q 스윕)의 끝점 변위가 널 축(초음파, "
                  "무효과 기대)의 끝점 변위와 겹치면 게이트 고정이 실패한 것이다.\n")
    lines.append("| 축 | mean displacement(끝점) | 95% CI |\n|---|---|---|")

    def endpoint_displacement(axis_name):
        d = axis_data[axis_name]
        emb = d["embeddings"]
        d_end = 1.0 - cos_rows(emb[:, 0, :], emb[:, -1, :])
        return d_end

    null_disp = np.concatenate([endpoint_displacement(a) for a in NULL_AXES if a in axis_data])
    null_lo, null_hi = bootstrap_ci_mean(null_disp)
    lines.append(f"| **null(12k+15k 통합, 바닥선)** | {null_disp.mean():.6f} | [{null_lo:.6f}, {null_hi:.6f}] |")

    gate_fail_axes = []
    for axis_name in GATE_CHECK_AXES:
        if axis_name not in axis_data:
            continue
        disp = endpoint_displacement(axis_name)
        lo, hi = bootstrap_ci_mean(disp)
        distinguishable = lo > null_hi
        if not distinguishable:
            gate_fail_axes.append(axis_name)
            ok_all = False
        lines.append(f"| {axis_name} | {disp.mean():.6f} | [{lo:.6f}, {hi:.6f}] {'' if distinguishable else '**★ 널과 구분 안 됨**'} |")

    lines.append("")
    if gate_fail_axes:
        lines.append(f"**★★★ 게이트 고정 실패 의심 축: {gate_fail_axes} — 진행하지 말고 보고할 것 (지시 §A.5).**\n")

    lines.append(f"\n## 종합 판정: {'**PASS**' if ok_all else '**★ FAIL — 사람 보고 필요**'}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "11_phase2_integrity.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {RESULTS_DIR / '11_phase2_integrity.md'}")
    print(f"종합 판정: {'PASS' if ok_all else 'FAIL'}")
    if eq_fail or gate_fail_axes:
        print("★★★ 중대 결함 감지 — Phase B/D/C 진행 전 사람 보고 필요 ★★★")


if __name__ == "__main__":
    main()
