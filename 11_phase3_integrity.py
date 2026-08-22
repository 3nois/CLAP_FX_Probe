# -*- coding: utf-8 -*-
"""Phase 3분석 착수 전 무결성 검증 — 10개 캐시 파일의 shape·src_id·NaN/Inf.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "out" / "caches"
RESULTS_DIR = ROOT / "out" / "results"

PAIRS_2D = ["reverb_wet_room", "reverb_wet_damping", "reverb_room_damping",
            "highshelf_gain_cutoff", "lowshelf_gain_cutoff", "peak_gain_cutoff"]
GRIDS_3DPLUS = ["highshelf_gain_cutoff_q", "lowshelf_gain_cutoff_q", "peak_gain_cutoff_q",
                "reverb_wet_room_damping_width"]


def check_file(path, expected_shape_check):
    if not path.exists():
        return {"exists": False}
    d = np.load(path, allow_pickle=True)
    emb = d["embeddings"]
    src_id = d["src_id"]
    ok_shape = expected_shape_check(emb.shape)
    ok_src = (len(src_id) == 1200 and np.array_equal(np.sort(src_id), np.arange(1200)))
    has_nan = bool(np.isnan(emb).any())
    has_inf = bool(np.isinf(emb).any())
    return {
        "exists": True, "shape": tuple(emb.shape), "shape_ok": ok_shape,
        "src_id_ok": ok_src, "has_nan": has_nan, "has_inf": has_inf,
        "ok": ok_shape and ok_src and not has_nan and not has_inf,
    }


def main():
    lines = ["# Phase 3 무결성 검증 (Phase 3분석 착수 전)\n"]
    lines.append("| 파일 | 존재 | shape | shape 정상 | src_id 0~1199 | NaN | Inf | 판정 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    all_ok = True

    for name in PAIRS_2D:
        p = CACHE_DIR / f"11_phase3_2d_{name}.npz"
        r = check_file(p, lambda s: s == (1200, 13, 13, 512))
        if not r.get("exists") or not r.get("ok"):
            all_ok = False
        if r.get("exists"):
            lines.append(f"| 2d_{name} | O | {r['shape']} | {'OK' if r['shape_ok'] else '**FAIL**'} | "
                         f"{'OK' if r['src_id_ok'] else '**FAIL**'} | {'없음' if not r['has_nan'] else '**있음**'} | "
                         f"{'없음' if not r['has_inf'] else '**있음**'} | {'PASS' if r['ok'] else '**FAIL**'} |")
        else:
            lines.append(f"| 2d_{name} | **✗** | — | — | — | — | — | **FAIL** |")

    for name in GRIDS_3DPLUS:
        p = CACHE_DIR / f"11_phase3_3dplus_{name}.npz"
        expected = (1200, 5, 5, 5, 512) if "reverb" not in name else (1200, 5, 5, 5, 5, 512)
        r = check_file(p, lambda s, e=expected: s == e)
        if not r.get("exists") or not r.get("ok"):
            all_ok = False
        if r.get("exists"):
            lines.append(f"| 3dplus_{name} | O | {r['shape']} | {'OK' if r['shape_ok'] else '**FAIL**'} | "
                         f"{'OK' if r['src_id_ok'] else '**FAIL**'} | {'없음' if not r['has_nan'] else '**있음**'} | "
                         f"{'없음' if not r['has_inf'] else '**있음**'} | {'PASS' if r['ok'] else '**FAIL**'} |")
        else:
            lines.append(f"| 3dplus_{name} | **✗** | — | — | — | — | — | **FAIL** |")

    lines.append(f"\n## 종합 판정: {'**PASS**' if all_ok else '**★ FAIL — 진행 중단, 보고 필요**'}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "11_phase3_integrity.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {RESULTS_DIR / '11_phase3_integrity.md'}")
    print(f"종합 판정: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
