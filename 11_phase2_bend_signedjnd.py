# -*- coding: utf-8 -*-
"""곡률 각도(bend) + 부호축 방향분리 JND — 사용자 지시 §2 (재렌더링 없음).

(a) bend(theta_i) = arccos(cos(delta_i, delta_{i+1})), delta_i = e(theta_{i+1})-e(theta_i).
    크기(변위)와 분리된 "궤적이 얼마나 휘는가" — deg 단위.
(b) *_gain 3축: theta_min=0에서 양(boost)/음(cut) 방향 JND를 따로 낸다
    (기존 doseresponse.py는 격자 시작(-15dB)부터 재서 boost/cut을 못 갈랐다).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

dr = import_module("11_phase2_doseresponse")

RESULTS_DIR = dr.RESULTS_DIR
GAIN_AXES = ["highshelf_gain", "lowshelf_gain", "peak_gain"]


def cos_vec(a, b):
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return num / den


def bend_curve(emb):
    """축별 bend(theta_i), i=1..n_levels-2 (delta_i, delta_{i+1} 둘 다 존재하는 구간)."""
    n_levels = emb.shape[1]
    deltas = emb[:, 1:, :] - emb[:, :-1, :]  # (n_src, n_levels-1, 512)
    bends = []
    for i in range(deltas.shape[1] - 1):
        c = cos_vec(deltas[:, i, :], deltas[:, i + 1, :])
        c = np.clip(c, -1.0, 1.0)
        ang = np.degrees(np.arccos(c))
        bends.append(float(np.mean(ang)))
    return np.array(bends)


def signed_jnd(axis_name, emb, theta_raw, null_jnd_p95):
    idx0 = int(np.argmin(np.abs(theta_raw)))
    results = {}
    for dir_name, idx_range in [("boost(+)", range(idx0, emb.shape[1] - 1)),
                                 ("cut(-)", range(idx0, 0, -1))]:
        first_clear = None
        for step_n, i in enumerate(idx_range):
            if dir_name == "boost(+)":
                a, b = emb[:, i, :], emb[:, i + 1, :]
                theta_to = theta_raw[i + 1]
            else:
                a, b = emb[:, i, :], emb[:, i - 1, :]
                theta_to = theta_raw[i - 1]
            delta = np.linalg.norm(b - a, axis=-1)
            rng = np.random.RandomState(0)
            n = len(delta)
            boots = np.array([np.mean(delta[rng.randint(0, n, n)]) for _ in range(500)])
            lo = float(np.percentile(boots, 2.5))
            if lo > null_jnd_p95 and first_clear is None:
                first_clear = float(theta_to)
                break
        results[dir_name] = first_clear
    return results


def main():
    lines = ["# 곡률(bend 각도) + 부호축 방향분리 JND — 사용자 지시 §2\n"]

    null_axis_data = {a: dr.load_concat(a) for a in dr.NULL_AXES}
    null_disp_p95, null_jnd_p95 = dr.build_null_floor(null_axis_data)
    lines.append(f"널 바닥(재사용, `11_phase2_doseresponse.md`와 동일): JND p95={null_jnd_p95:.4f}\n")

    lines.append("## (a) 곡률 bend 각도 (도, deg)\n")
    lines.append("| 축 | 중앙값 | 최댓값 | 최대 위치 | 참고: 원값 κ(구 곡률) |")
    lines.append("|---|---|---|---|---|")
    all_axes = dr.EXT_AXES + dr.NULL_AXES
    bend_summary = {}
    for axis_name in all_axes:
        emb, theta_raw, src_id = dr.load_concat(axis_name)
        bends = bend_curve(emb)
        med, mx = float(np.median(bends)), float(np.max(bends))
        max_i = int(np.argmax(bends)) + 1
        n_levels = emb.shape[1]
        third = n_levels // 3
        region = "하위1/3" if max_i < third else ("중위1/3" if max_i < 2 * third else "상위1/3")
        kap = dr.curvature_summary(emb)
        lines.append(f"| {axis_name} | {med:.1f}° | {mx:.1f}° | {region} | mean={kap['mean_kappa']:.4f} |")
        bend_summary[axis_name] = {"median_deg": med, "max_deg": mx, "max_region": region}
        print(f"완료(bend): {axis_name}")

    lines.append("\n★ bend가 크면(180°에 가까우면 완전 반전, 90°면 직각으로 꺾임) 손잡이 방향이 "
                 "구간마다 다르다는 뜻 — 8차의 방향 예측(cos 0.71~0.82)이 전역이 아니라 국소적으로만 "
                 "유효할 수 있음을 뜻한다. 위 표는 축별 대표 각도(중앙값)와 최악(최댓값) 구간을 함께 보여준다.\n")

    lines.append("## (b) 부호축(*_gain) 방향분리 JND — boost(+)/cut(-) 각각\n")
    lines.append("| 축 | boost(+) 첫 JND | cut(-) 첫 JND | 실무범위(단측 15dB) 대비 % |")
    lines.append("|---|---|---|---|")
    for axis_name in GAIN_AXES:
        emb, theta_raw, src_id = dr.load_concat(axis_name)
        sj = signed_jnd(axis_name, emb, theta_raw, null_jnd_p95)
        boost_pct = (sj["boost(+)"] / 15.0 * 100) if sj["boost(+)"] is not None else None
        cut_pct = (abs(sj["cut(-)"]) / 15.0 * 100) if sj["cut(-)"] is not None else None
        b_str = f"{sj['boost(+)']:.3g}dB ({boost_pct:.1f}%)" if sj["boost(+)"] is not None else "격자 내 미검출"
        c_str = f"{sj['cut(-)']:.3g}dB ({cut_pct:.1f}%)" if sj["cut(-)"] is not None else "격자 내 미검출"
        lines.append(f"| {axis_name} | {b_str} | {c_str} | — |")
        print(f"완료(signed JND): {axis_name}")

    out_path = RESULTS_DIR / "11_phase2_bend_signedjnd.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
