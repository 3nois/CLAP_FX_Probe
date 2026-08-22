# -*- coding: utf-8 -*-
"""Phase 5-E — 곡률(bend)-회전(rotation) 통합표. 신규 계산 최소화, 기존 산출 재사용.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

dr = import_module("11_phase2_doseresponse")
bendmod = import_module("11_phase2_bend_signedjnd")

RESULTS_DIR = dr.RESULTS_DIR

# bend: Phase2 축 이름, rotation: Phase3 rotation_raw.json의 (pair, focus) 매칭
MATCH = [
    ("reverb_wet_level", "reverb_wet_room", "wet_level"),
    ("reverb_room_size", "reverb_wet_room", "room_size"),
    ("reverb_wet_level", "reverb_wet_damping", "wet_level"),
    ("reverb_damping", "reverb_wet_damping", "damping"),
    ("reverb_room_size", "reverb_room_damping", "room_size"),
    ("reverb_damping", "reverb_room_damping", "damping"),
    ("highshelf_gain", "highshelf_gain_cutoff", "gain"),
    ("highshelf_cutoff_gp6", "highshelf_gain_cutoff", "cutoff"),
    ("lowshelf_gain", "lowshelf_gain_cutoff", "gain"),
    ("lowshelf_cutoff_gp6", "lowshelf_gain_cutoff", "cutoff"),
    ("peak_gain", "peak_gain_cutoff", "gain"),
    ("peak_cutoff_gp6", "peak_gain_cutoff", "cutoff"),
]


def main():
    with open(RESULTS_DIR / "11_phase3_rotation_raw.json", encoding="utf-8") as f:
        rot_raw = json.load(f)["results"]
    rot_by_key = {(r["pair"], r["focus"]): r for r in rot_raw}

    lines = ["# Phase 5-E — 곡률(bend)-회전(rotation) 통합표\n"]
    lines.append("bend: 한 축 안에서 세게 걸수록 방향이 바뀌나(Phase 2). "
                 "rotation: 다른 파라미터가 바뀌면 방향이 바뀌나(Phase 3, 3-1). 신규 계산 없음, 기존 산출 재사용.\n")
    lines.append("| bend 축 | rotation 쌍(focus) | bend 중앙값 | bend 최댓값 | rot_context 최댓값 | rot_source | 해석 |")
    lines.append("|---|---|---|---|---|---|---|")

    seen_bend = {}
    for bend_axis, rot_pair, rot_focus in MATCH:
        if bend_axis not in seen_bend:
            emb, theta_raw, src_id = dr.load_concat(bend_axis)
            bends = bendmod.bend_curve(emb)
            seen_bend[bend_axis] = (float(np.median(bends)), float(np.max(bends)))
        bend_med, bend_max = seen_bend[bend_axis]

        r = rot_by_key.get((rot_pair, rot_focus))
        if r is None:
            continue
        rc = r["rot_context_max"]
        rs = r["rot_source"]
        both_high = bend_max > 30 and rc["mean_deg"] > rs["mean_deg"]
        interp = "축 내부·축 간 모두 회전 큼 — 국소 손잡이 필요" if both_high else (
            "축 내부는 안정, 축 간 회전이 지배적" if rc["mean_deg"] > bend_max else
            "축 내부 회전이 축 간보다 큼 — 구간 세분화가 더 중요")
        lines.append(f"| {bend_axis} | {rot_pair}({rot_focus}) | {bend_med:.1f}° | {bend_max:.1f}° | "
                     f"{rc['mean_deg']:.1f}° | {rs['mean_deg']:.1f}° | {interp} |")
        print(f"완료: {bend_axis} / {rot_pair}({rot_focus})")

    out_path = RESULTS_DIR / "11_phase5_curvature_rotation.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
