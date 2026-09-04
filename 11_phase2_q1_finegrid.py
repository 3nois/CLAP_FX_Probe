"""CLAP FX Probe — 11_phase2_q1_finegrid.py (Q1: R² vs 윈도우 폭 — 20개 점 재계산)

`11_phase2_doseresponse.py`의 `windowed_r2_table`을 그대로 재사용하되 폭 그리드만
5개(20/40/60/80/100%)에서 20개(5%~100%, 5%씩)로 촘촘하게 바꿔 측정된 23축 전부를
다시 돌린다. CLAP 재계산 없음 — 기존 임베딩 캐시로 Ridge 프로브만 재적합.
기존 `11_phase2_doseresponse_raw.json`은 건드리지 않고 별도 파일에 저장한다.
"""
import json
import time
from importlib import import_module

import numpy as np

dr = import_module("11_phase2_doseresponse")

AXES = [
    "distortion_drive_db",
    "reverb_wet_level", "reverb_room_size", "reverb_damping", "reverb_width",
    "highshelf_gain", "lowshelf_gain", "peak_gain",
    "highshelf_cutoff_gp6", "lowshelf_cutoff_gp6", "peak_cutoff_gp6",
    "highshelf_q_gp6", "lowshelf_q_gp6", "peak_q_gp6",
    "highshelf_cutoff_gn6", "lowshelf_cutoff_gn6", "peak_cutoff_gn6",
    "highshelf_q_gn6", "lowshelf_q_gn6", "peak_q_gn6",
    "null_12k_gain", "null_15k_gain",
    "eq_cascade_intensity",
]

dr.WIDTH_FRACS = [round(w, 2) for w in np.arange(0.05, 1.0001, 0.05)]


def main():
    out = {}
    t0 = time.time()
    for axis in AXES:
        emb, theta_raw, src_id = dr.load_concat(axis)
        out[axis] = {str(w): v for w, v in dr.windowed_r2_table(emb, theta_raw, src_id).items()}
        print(f"  {axis} 완료 ({time.time() - t0:.1f}s)")

    path = "out/results/11_phase2_q1_finegrid_raw.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"저장: {path} (총 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
