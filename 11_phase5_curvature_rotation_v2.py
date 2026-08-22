# -*- coding: utf-8 -*-
"""Phase 5-E v2 — 결함 20 수정: bend-rotation 통합표를 v1(결함18 오염) 대신 v5(unsigned)+v4(signed) 산출로 교체.

11_phase5_curvature_rotation.py(원본)는 11_phase3_rotation_raw.json(v1, 결함18 미수정
`normalize(e_max)-normalize(e_min)` 원본)을 참조하고 있었다. v2~v5를 거치며 rotation
수치가 전부 갱신됐는데도 이 파일만 한 번도 갱신되지 않았다 — 신규 계산 없음, 기존
산출 재사용이라는 원 스크립트의 취지 자체는 유지하되 재사용 대상을 최신 산출로
바꾼다. cutoff-focus(=signed context) 3쌍은 이제 gain+/gain- 두 분기로 나뉘어
있으므로, 기존 1행(EQ당 cutoff-focus) 대신 분기별 2행으로 표를 확장한다.

원본 11_phase5_curvature_rotation.md는 인용 금지로 표시(재실행 전 상태)한다.
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

# unsigned: bend 축 <-> (rotation pair, focus) 1:1 매칭 (v5 unsigned_final에서 조회)
MATCH_UNSIGNED = [
    ("reverb_wet_level", "reverb_wet_room", "wet_level"),
    ("reverb_room_size", "reverb_wet_room", "room_size"),
    ("reverb_wet_level", "reverb_wet_damping", "wet_level"),
    ("reverb_damping", "reverb_wet_damping", "damping"),
    ("reverb_room_size", "reverb_room_damping", "room_size"),
    ("reverb_damping", "reverb_room_damping", "damping"),
    ("highshelf_gain", "highshelf_gain_cutoff", "gain"),
    ("lowshelf_gain", "lowshelf_gain_cutoff", "gain"),
    ("peak_gain", "peak_gain_cutoff", "gain"),
]
# signed: bend 축(EQ cutoff) <-> (rotation pair, branch) — v4 signed_results에서 조회, +/- 각각
MATCH_SIGNED = [
    ("highshelf_cutoff_gp6", "highshelf_gain_cutoff"),
    ("lowshelf_cutoff_gp6", "lowshelf_gain_cutoff"),
    ("peak_cutoff_gp6", "peak_gain_cutoff"),
]


def main():
    with open(RESULTS_DIR / "11_phase3_rotation_v5_raw.json", encoding="utf-8") as f:
        v5 = json.load(f)
    with open(RESULTS_DIR / "11_phase3_rotation_v4_raw.json", encoding="utf-8") as f:
        v4 = json.load(f)
    unsigned_by_key = {(r["pair"], r["focus"]): r for r in v5["unsigned_final"]}
    signed_by_pair = v4["signed_results"]

    lines = ["# Phase 5-E v2 — 곡률(bend)-회전(rotation) 통합표 (결함 20 수정, 2026-08-22)\n"]
    lines.append("**결함 20**: 원본(`11_phase5_curvature_rotation.md`)은 `11_phase3_rotation_raw.json`"
                 "(v1, 결함18 미수정 — 회전 벡터를 `normalize(e_max)-normalize(e_min)`로 정의해 약한 "
                 "효과 지점에서 잡음에 압도되는 버전)을 참조하고 있었다. v2~v5로 rotation 수치가 갱신되는 "
                 "동안 이 파일만 누락됐다 — **하류 오염 사례**. 원본 파일은 재인용 금지로 표시한다. "
                 "이 파일은 v5(unsigned 9건)와 v4(signed 6건, 부호 분기별)의 최신 산출을 그대로 재사용한다"
                 "(신규 rotation 계산 없음, bend만 재사용).\n")
    lines.append("bend: 한 축 안에서 세게 걸수록 방향이 바뀌나(Phase 2). "
                 "rotation: 다른 파라미터가 바뀌면 방향이 바뀌나(Phase 3, 3-1 v4/v5). "
                 "cutoff-focus(EQ 3쌍)는 부호 분기가 있어 gain+/gain- 두 행으로 나눠 보고한다.\n")
    lines.append("| bend 축 | rotation 쌍(focus/분기) | bend 중앙값 | bend 최댓값 | rot_context 최댓값 | "
                 "rot_source | rotation 판정 | 해석 |")
    lines.append("|---|---|---|---|---|---|---|---|")

    seen_bend = {}

    def get_bend(axis_name):
        if axis_name not in seen_bend:
            emb, theta_raw, src_id = dr.load_concat(axis_name)
            bends = bendmod.bend_curve(emb)
            seen_bend[axis_name] = (float(np.median(bends)), float(np.max(bends)))
        return seen_bend[axis_name]

    def emit_row(bend_axis, label, rc, rs, verdict):
        bend_med, bend_max = get_bend(bend_axis)
        both_high = bend_max > 30 and rc["mean_deg"] > rs["mean_deg"]
        interp = "축 내부·축 간 모두 회전 큼 — 국소 손잡이 필요" if both_high else (
            "축 내부는 안정, 축 간 회전이 지배적" if rc["mean_deg"] > bend_max else
            "축 내부 회전이 축 간보다 큼 — 구간 세분화가 더 중요")
        lines.append(f"| {bend_axis} | {label} | {bend_med:.1f}° | {bend_max:.1f}° | "
                     f"{rc['mean_deg']:.1f}° | {rs['mean_deg']:.1f}° | {verdict} | {interp} |")
        print(f"완료: {bend_axis} / {label} -> {verdict}")

    for bend_axis, rot_pair, rot_focus in MATCH_UNSIGNED:
        r = unsigned_by_key.get((rot_pair, rot_focus))
        if r is None or r.get("rot_context_max") is None:
            continue
        emit_row(bend_axis, f"{rot_pair}({rot_focus})", r["rot_context_max"], r["rot_source"], r["verdict"])

    for bend_axis, rot_pair in MATCH_SIGNED:
        for branch_key, branch_label in [("pos", "gain+"), ("neg", "gain-")]:
            br = signed_by_pair[rot_pair][branch_key]
            emit_row(bend_axis, f"{rot_pair}(cutoff, {branch_label})", br["rot_context_max"], br["rot_source"], br["verdict"])

    out_path = RESULTS_DIR / "11_phase5_curvature_rotation_v2.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
