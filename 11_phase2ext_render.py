# -*- coding: utf-8 -*-
"""Phase 2 확장 렌더링 — 400 → 1,200 소스 (사용자 지시 §C).

기존 400소스 캐시(out/caches/11_phase2_*.npz)는 절대 건드리지 않는다. 새로 뽑은
800소스만 별도 파일명(out/caches/11_phase2ext_<axis>.npz)에 렌더링하고, 분석
단계에서 두 캐시를 concat해 1,200소스로 합친다.

확장분은 17축만 렌더링한다(gp6 게이트 한 벌만, gn6 6축 제외 — 부스트/컷 비대칭
점검용 보조 분석으로만 쓰고 주 수치에는 안 쓰므로 기존 400소스분으로 충분).
"""
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

base = import_module("11_phase2_render")

ROOT = base.ROOT
AUDIO_DIR = base.AUDIO_DIR
CACHE_DIR = base.CACHE_DIR
RESULTS_DIR = base.RESULTS_DIR
LOG_DIR = base.LOG_DIR
SR = base.SR
BATCH_SIZE = base.BATCH_SIZE
N_PER_FAMILY_EXT = 80
EXT_SOURCE_SEED = 1  # 기존 400(seed=0)과 달라야 함 — 반드시 기록

EXT_AXES = [
    "distortion_drive_db",
    "reverb_wet_level", "reverb_room_size", "reverb_damping", "reverb_width",
    "highshelf_gain", "lowshelf_gain", "peak_gain",
    "highshelf_cutoff_gp6", "lowshelf_cutoff_gp6", "peak_cutoff_gp6",
    "highshelf_q_gp6", "lowshelf_q_gp6", "peak_q_gp6",
    "eq_cascade_intensity",
    "null_12k_gain", "null_15k_gain",
]
assert len(EXT_AXES) == 17, len(EXT_AXES)


def select_extension_sources(n_per_family, seed, used_filenames):
    files = sorted(AUDIO_DIR.glob("*.wav"))
    by_family = collections.defaultdict(list)
    for f in files:
        if f.name in used_filenames:
            continue
        instrument, pitch, velocity = base.parse_nsynth_filename(f)
        if pitch is None:
            continue
        fam = base.parse_instrument_family(instrument)
        by_family[fam].append((f.name, instrument))
    rng = np.random.RandomState(seed)
    selected = []
    for fam in sorted(by_family.keys()):
        pool = by_family[fam]
        n_take = min(n_per_family, len(pool))
        if n_take < n_per_family:
            print(f"경고: 패밀리 {fam} 에 신규 후보가 {n_take}개뿐 (목표 {n_per_family})")
        idx = rng.choice(len(pool), size=n_take, replace=False)
        for i in sorted(idx.tolist()):
            fname, instrument = pool[i]
            selected.append((fname, instrument, fam))
    return selected


def get_or_select_ext_sources():
    ext_path = RESULTS_DIR / "11_phase2_sources_ext.json"
    if ext_path.exists():
        with open(ext_path, encoding="utf-8") as f:
            return json.load(f)["sources"]

    with open(RESULTS_DIR / "11_phase2_sources.json", encoding="utf-8") as f:
        base_sources = json.load(f)["sources"]
    used_filenames = {s["filename"] for s in base_sources}
    n_base = len(base_sources)

    selected = select_extension_sources(N_PER_FAMILY_EXT, EXT_SOURCE_SEED, used_filenames)
    fam_counts = collections.Counter(fam for _, _, fam in selected)
    sources = [{"src_id": n_base + i, "filename": fn, "instrument": inst, "family": fam}
               for i, (fn, inst, fam) in enumerate(selected)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ext_path, "w", encoding="utf-8") as f:
        json.dump({"seed": EXT_SOURCE_SEED, "n_per_family": N_PER_FAMILY_EXT,
                   "family_counts": dict(fam_counts), "n_total": len(sources),
                   "src_id_offset": n_base, "excluded_filenames_count": len(used_filenames),
                   "sources": sources}, f, indent=2, ensure_ascii=False)
    print(f"확장 소스 목록 신규 확정 및 저장: {ext_path} ({len(sources)}개, src_id {n_base}~{n_base+len(sources)-1})")

    # 중복 검증
    overlap = used_filenames & {s["filename"] for s in sources}
    assert not overlap, f"기존 소스와 파일명 중복: {overlap}"
    return sources


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t_script_start = time.time()

    sources = get_or_select_ext_sources()
    n_sources = len(sources)
    print(f"확장 소스 {n_sources}개 확정")

    print("CLAP 로딩...")
    torch_device = base.embed_mod.torch.device("cpu")
    clap = base.embed_mod.load_clap(torch_device, ROOT / "ckpts")

    print("확장 소스 오디오 로딩 (dry, 조건 A)...")
    y_dry = []
    for s in sources:
        y = base.embed_mod.load_and_preprocess(AUDIO_DIR / s["filename"])
        assert y is not None, f"무음 소스: {s['filename']}"
        y_dry.append(y)

    def embed_batch(batch):
        return base.embed_mod.embed_batch(clap, torch_device, batch)

    # bypass(진짜 dry) — 확장분도 별도 저장 (분석 시 base bypass와 concat)
    bypass_path = CACHE_DIR / "11_phase2ext_bypass.npz"
    if bypass_path.exists():
        print(f"확장 bypass 캐시 재사용: {bypass_path}")
    else:
        emb_list = [embed_batch(y_dry[i:i + BATCH_SIZE]) for i in range(0, n_sources, BATCH_SIZE)]
        bypass_emb = np.concatenate(emb_list, axis=0).astype(np.float32)
        np.savez(bypass_path, embeddings=bypass_emb,
                 src_id=np.array([s["src_id"] for s in sources], dtype=np.int64))
        print(f"저장: {bypass_path}")

    cascade_seeds_ext_path = RESULTS_DIR / "11_phase2_cascade_seeds_ext.json"
    cascade_gains_by_src = {}
    if cascade_seeds_ext_path.exists():
        with open(cascade_seeds_ext_path, encoding="utf-8") as f:
            saved = json.load(f)
        cascade_gains_by_src = {int(k): v for k, v in saved.items()}
    else:
        for s in sources:
            rng = np.random.default_rng(42 + s["src_id"])  # base와 동일 규칙 — src_id로만 결정
            gains = {b: float(rng.uniform(-15, 15)) for b in base.CASCADE_BAND_FREQS}
            cascade_gains_by_src[s["src_id"]] = gains
        with open(cascade_seeds_ext_path, "w", encoding="utf-8") as f:
            json.dump(cascade_gains_by_src, f, indent=2, ensure_ascii=False)
        print(f"확장분 cascade 시드 저장: {cascade_seeds_ext_path}")

    total_axes = len(EXT_AXES)
    for axis_i, axis_name in enumerate(EXT_AXES):
        axis_cache_path = CACHE_DIR / f"11_phase2ext_{axis_name}.npz"
        if axis_cache_path.exists():
            print(f"[{axis_i+1}/{total_axes}] {axis_name} — 캐시 존재, 건너뜀")
            continue

        t_axis_start = time.time()
        spec = base.AXES[axis_name]
        levels = spec["levels"]
        n_levels = len(levels)
        emb = np.zeros((n_sources, n_levels, 512), dtype=np.float32)
        batch_audio, batch_keys = [], []

        def flush():
            if not batch_audio:
                return
            e = embed_batch(batch_audio)
            for (si, li), v in zip(batch_keys, e):
                emb[si, li] = v
            batch_audio.clear()
            batch_keys.clear()

        n_jobs = n_sources * n_levels
        print(f"[{axis_i+1}/{total_axes}] {axis_name} — {n_jobs}회 렌더링 시작")

        for si, s in enumerate(sources):
            y = y_dry[si]
            if axis_name == "eq_cascade_intensity":
                gains = cascade_gains_by_src[s["src_id"]]
                for li, sv in enumerate(levels):
                    board = base.cascade_board(gains, float(sv))
                    wet = board(y, SR)
                    batch_audio.append(wet)
                    batch_keys.append((si, li))
                    if len(batch_audio) >= BATCH_SIZE:
                        flush()
            else:
                for li, lv in enumerate(levels):
                    board = spec["board_fn"](lv)
                    wet = board(y, SR)
                    batch_audio.append(wet)
                    batch_keys.append((si, li))
                    if len(batch_audio) >= BATCH_SIZE:
                        flush()
        flush()

        elapsed_axis = time.time() - t_axis_start
        np.savez(
            axis_cache_path, embeddings=emb, theta_raw=np.array(levels, dtype=np.float64),
            src_id=np.array([s["src_id"] for s in sources], dtype=np.int64),
            axis_name=axis_name, n_levels=n_levels, elapsed_sec=elapsed_axis,
        )
        elapsed_total = time.time() - t_script_start
        print(f"[{axis_i+1}/{total_axes}] {axis_name} 완료 — {elapsed_axis/60:.1f}분 "
              f"(누적 {elapsed_total/60:.1f}분) 저장: {axis_cache_path}")

    total_elapsed = time.time() - t_script_start
    meta = {
        "n_sources_ext": n_sources, "n_per_family_ext": N_PER_FAMILY_EXT, "ext_source_seed": EXT_SOURCE_SEED,
        "axes": EXT_AXES, "n_axes": total_axes, "total_conditions": total_axes * n_sources * base.N_LEVELS,
        "total_elapsed_sec": total_elapsed,
    }
    with open(RESULTS_DIR / "11_phase2ext_render_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n=== Phase 2 확장 렌더링 완료: {total_elapsed/60:.1f}분 ===")


if __name__ == "__main__":
    main()
