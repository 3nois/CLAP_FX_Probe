"""CLAP FX Probe — 11_phase9_listen.py (Phase 9 §6-9: 청취용 표본 추출)

pedalboard 전용 재렌더, CLAP 재계산 없음. 질의 / R0 top1 / R1 top1 을 한 벌로
묶어 out/audio/phase9_listen/ 에 저장한다. alpha는 §6-4에서 val로 선택한 값을
그대로 재사용(§6-5와 동일). 실제 청취(§5-3 tier 1, 20~30개, 결과와 무관하게
필수)는 사람이 직접 해야 하며 이 스크립트는 표본만 준비한다.
"""
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf

r0mod = import_module("11_phase9_retrieval")
r1mod = import_module("11_phase9_r1")
m2mod = import_module("11_phase9_m2m3")
r2mod = import_module("11_phase2_render")
physmod = import_module("11_phase9_physical")

unit = r0mod.unit
OUT_DIR = Path("out/audio/phase9_listen")
N_SRC_PER_LEVEL = 4
SAMPLE_SEED = 0


def render_lib_item(fname_of, axis_spec, pos, lib_src):
    """M2 라이브러리 위치(0..3599)에 해당하는 실제 오디오를 재생성한다."""
    src = int(lib_src[pos])
    y = r2mod.embed_mod.load_and_preprocess(r2mod.AUDIO_DIR / fname_of[src])
    if pos < 1200:
        return y, src, "dry"
    lvl = 18 if pos < 2400 else 24
    wet = axis_spec["board_fn"](axis_spec["levels"][lvl])(y, r2mod.SR)
    return wet, src, f"wet@{lvl}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bypass = r0mod.load_bypass()
    family_arr = r0mod.load_family_array()
    group_of = r0mod.load_dup_groups()
    train_idx, val_idx, test_idx = r0mod.stratified_split(family_arr, seed=r0mod.SEED)
    r1_prior = json.load(open("out/results/11_phase9_r1.json"))

    sources = (json.load(open("out/results/11_phase2_sources.json"))["sources"]
               + json.load(open("out/results/11_phase2_sources_ext.json"))["sources"])
    fname_of = {s["src_id"]: s["filename"] for s in sources}

    rng = np.random.RandomState(SAMPLE_SEED)
    manifest = []

    for axis in r0mod.AXES:
        emb, theta = r0mod.load_axis(axis)
        model = r1mod.train_b2(axis, emb, bypass, train_idx, val_idx)
        lib, lib_src, lib_is_dry, lib_family = m2mod.build_m2_library(axis, emb, bypass, family_arr)
        axis_spec = r2mod.AXES[axis]
        sample_srcs = rng.choice(test_idx, size=N_SRC_PER_LEVEL, replace=False)

        for lvl in r0mod.QUERY_LEVELS:
            e_wet = emb[:, lvl, :]
            v_hat = unit(r1mod.predict_direction(model, e_wet))
            alpha = r1_prior[axis][str(lvl)]["R1"]["alpha"]

            pos_r0 = physmod.top1_lib_pos(sample_srcs, e_wet[sample_srcs], lib, lib_src, group_of)
            q_r1 = unit(e_wet[sample_srcs] + alpha * v_hat[sample_srcs])
            pos_r1 = physmod.top1_lib_pos(sample_srcs, q_r1, lib, lib_src, group_of)

            for i, src in enumerate(sample_srcs):
                y_dry = r2mod.embed_mod.load_and_preprocess(r2mod.AUDIO_DIR / fname_of[int(src)])
                query_audio = axis_spec["board_fn"](axis_spec["levels"][lvl])(y_dry, r2mod.SR)
                r0_audio, r0_src, r0_tag = render_lib_item(fname_of, axis_spec, pos_r0[i], lib_src)
                r1_audio, r1_src, r1_tag = render_lib_item(fname_of, axis_spec, pos_r1[i], lib_src)

                stem = f"{axis}_lvl{lvl}_src{src}"
                sf.write(OUT_DIR / f"{stem}_query.wav", query_audio, r2mod.SR)
                sf.write(OUT_DIR / f"{stem}_R0top1.wav", r0_audio, r2mod.SR)
                sf.write(OUT_DIR / f"{stem}_R1top1.wav", r1_audio, r2mod.SR)
                manifest.append({
                    "axis": axis, "level": lvl, "alpha": alpha, "query_src": int(src),
                    "R0_top1_src": r0_src, "R0_top1_tag": r0_tag,
                    "R1_top1_src": r1_src, "R1_top1_tag": r1_tag,
                    "files": [f"{stem}_query.wav", f"{stem}_R0top1.wav", f"{stem}_R1top1.wav"],
                })

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"{len(manifest)}개 표본(triplet) 저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
