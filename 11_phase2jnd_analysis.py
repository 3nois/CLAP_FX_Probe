# -*- coding: utf-8 -*-
"""JND 정밀 측정 결과 분석 — 사용자 지시 §3 산출.

theta_min과 기존 격자 2번째 점을 캐시에서 가져와 로그 미세격자 앞뒤에 이어 붙이고,
인접쌍 변위가 널 바닥(95백분위)을 95% 신뢰수준으로 처음 넘는 지점을 찾는다.
축별 JND 값과 실무범위 대비 %를 산출한다. 부호축은 boost/cut 분리.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

dr = import_module("11_phase2_doseresponse")
jr = import_module("11_phase2jnd_render")

CACHE_DIR = dr.CACHE_DIR
RESULTS_DIR = dr.RESULTS_DIR
ONE_DIR_AXES = jr.ONE_DIR_AXES
GAIN_AXES = jr.GAIN_AXES


def bootstrap_ci_lo(x, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(x)
    boots = np.array([np.mean(x[rng.randint(0, n, n)]) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5))


def find_jnd_in_sequence(embeddings_seq, thetas_seq, null_jnd_p95):
    """embeddings_seq: (n_src, n_points, 512), thetas_seq 오름차순(절대값 기준 아님, 실제 값).
    첫 번째로 널을 넘는 (theta_from, theta_to, step_size) 반환, 없으면 None."""
    n_points = embeddings_seq.shape[1]
    for i in range(n_points - 1):
        delta = np.linalg.norm(embeddings_seq[:, i + 1, :] - embeddings_seq[:, i, :], axis=-1)
        lo = bootstrap_ci_lo(delta)
        if lo > null_jnd_p95:
            return float(thetas_seq[i]), float(thetas_seq[i + 1]), float(abs(thetas_seq[i + 1] - thetas_seq[0]))
    return None


def main():
    null_axis_data = {a: dr.load_concat(a) for a in dr.NULL_AXES}
    _, null_jnd_p95 = dr.build_null_floor(null_axis_data)

    with open(RESULTS_DIR / "11_phase2jnd_sources.json", encoding="utf-8") as f:
        jnd_src_ids = json.load(f)["src_ids"]
    jnd_src_ids = np.array(jnd_src_ids)

    lines = ["# JND 정밀 측정 — 최종 결과 (사용자 지시 §3)\n"]
    lines.append(f"널 바닥(재사용): JND p95={null_jnd_p95:.4f}. 소스 300개(30/family), "
                 f"theta_min~기존 2번째 격자점 사이 로그 20단계로 재측정.\n")
    lines.append("★ 축 간 비교는 이 값(정밀 JND)으로만 한다 — 25레벨 원 격자의 '1번째 스텝'은 "
                 "전부 '한 칸 미만'이라는 것만 알려줄 뿐 실제 문턱값이 아니었다.\n")

    lines.append("| 축 | 방향 | JND(theta 단위) | 원점~JND까지 절댓값 | 실무범위 대비 % |")
    lines.append("|---|---|---|---|---|")

    jnd_results = {}
    for axis_name in ONE_DIR_AXES:
        jnd_cache = np.load(CACHE_DIR / f"11_phase2jnd_{axis_name}.npz")
        base_cache = np.load(CACHE_DIR / f"11_phase2_{axis_name}.npz")
        theta_raw = base_cache["theta_raw"]
        idx0 = jr.theta_min_index(axis_name, theta_raw)
        base_src_id = base_cache["src_id"]
        sub_idx = np.searchsorted(base_src_id, jnd_src_ids)
        assert np.all(base_src_id[sub_idx] == jnd_src_ids)

        e_min = base_cache["embeddings"][sub_idx, idx0, :]
        e_next = base_cache["embeddings"][sub_idx, idx0 + 1, :]
        fine_theta = jnd_cache["theta_fine_pos"]
        fine_emb = jnd_cache["embeddings_pos"]

        seq_emb = np.concatenate([e_min[:, None, :], fine_emb, e_next[:, None, :]], axis=1)
        seq_theta = np.concatenate([[theta_raw[idx0]], fine_theta, [theta_raw[idx0 + 1]]])

        result = find_jnd_in_sequence(seq_emb, seq_theta, null_jnd_p95)
        practical_range = float(theta_raw.max() - theta_raw.min())
        if result:
            t_from, t_to, mag = result
            pct = mag / practical_range * 100
            lines.append(f"| {axis_name} | — | {t_to:.4g} | {mag:.4g} | {pct:.3f}% |")
            jnd_results[axis_name] = {"jnd_theta": t_to, "magnitude": mag, "pct_of_range": pct}
        else:
            lines.append(f"| {axis_name} | — | 미검출(20단계 내 못 넘음) | — | — |")
            jnd_results[axis_name] = None
        print(f"완료: {axis_name} -> {result}")

    for axis_name in GAIN_AXES:
        jnd_cache = np.load(CACHE_DIR / f"11_phase2jnd_{axis_name}.npz")
        base_cache = np.load(CACHE_DIR / f"11_phase2_{axis_name}.npz")
        theta_raw = base_cache["theta_raw"]
        idx0 = jr.theta_min_index(axis_name, theta_raw)
        base_src_id = base_cache["src_id"]
        sub_idx = np.searchsorted(base_src_id, jnd_src_ids)
        assert np.all(base_src_id[sub_idx] == jnd_src_ids)
        e_min = base_cache["embeddings"][sub_idx, idx0, :]

        for dir_name, sign, next_idx in [("boost(+)", "pos", idx0 + 1), ("cut(-)", "neg", idx0 - 1)]:
            e_next = base_cache["embeddings"][sub_idx, next_idx, :]
            fine_theta = jnd_cache[f"theta_fine_{sign}"]
            fine_emb = jnd_cache[f"embeddings_{sign}"]
            seq_emb = np.concatenate([e_min[:, None, :], fine_emb, e_next[:, None, :]], axis=1)
            seq_theta = np.concatenate([[theta_raw[idx0]], fine_theta, [theta_raw[next_idx]]])
            result = find_jnd_in_sequence(seq_emb, seq_theta, null_jnd_p95)
            practical_range_1sided = 15.0
            if result:
                t_from, t_to, mag = result
                pct = mag / practical_range_1sided * 100
                lines.append(f"| {axis_name} | {dir_name} | {t_to:.4g} | {mag:.4g} | {pct:.3f}% |")
                jnd_results[f"{axis_name}_{dir_name}"] = {"jnd_theta": t_to, "magnitude": mag, "pct_of_range": pct}
            else:
                lines.append(f"| {axis_name} | {dir_name} | 미검출 | — | — |")
                jnd_results[f"{axis_name}_{dir_name}"] = None
            print(f"완료: {axis_name} {dir_name} -> {result}")

    out_path = RESULTS_DIR / "11_phase2jnd_final.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {out_path}")

    with open(RESULTS_DIR / "11_phase2jnd_final.json", "w", encoding="utf-8") as f:
        json.dump(jnd_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
