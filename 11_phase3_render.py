# -*- coding: utf-8 -*-
"""Phase 3 — 2-D 상호작용 격자 (13x13) + 3-D+ 수치전용 격자. 1,200소스(승인됨).

2-D 6쌍(13x13=169조건):
  reverb: wet_level x room_size, wet_level x damping, room_size x damping
  EQ 3타입: gain x cutoff (q는 기하평균 중앙값 0.4472 고정)

3-D+ 수치전용(그림 없음, 분산분해용 축소 격자 5레벨/축):
  EQ 3타입: gain x cutoff x q (5^3=125)
  reverb: wet_level x room_size x damping x width (5^4=625)

축별 기준점(Phase 1 사전등록 OAT 규칙 그대로): 스윕에 포함 안 된 reverb 파라미터는
Phase 2와 동일한 기본값/기준점 사용, wet_level이 스윕 축이 아닐 때만 0.3 고정.

조건 수: 2D 6쌍x169 + 3D+ EQ 3x125 + 3D+ reverb 1x625 = 1014+375+625=2014
조건 x 1,200소스 = 2,416,800회.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pedalboard as pb

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

Q_MEDIAN = float(np.sqrt(0.1 * 2.0))
N_2D = 13
N_3D = 5

RANGES = {
    "wet_level": (0.0, 0.5), "room_size": (0.05, 0.85), "damping": (0.0, 1.0), "width": (0.0, 1.0),
    "gain": (-15.0, 15.0), "q": (0.1, 2.0),
    "cutoff_highshelf": (500.0, 4000.0), "cutoff_lowshelf": (30.0, 200.0), "cutoff_peak": (200.0, 6000.0),
}
EQ_CLASS = {"highshelf": pb.HighShelfFilter, "lowshelf": pb.LowShelfFilter, "peak": pb.PeakFilter}


def lin_grid(lo, hi, n):
    return np.linspace(lo, hi, n)


def log_grid(lo, hi, n):
    return np.geomspace(lo, hi, n)


def reverb_board_full(wet_level, room_size, damping, width):
    return pb.Pedalboard([pb.Reverb(
        room_size=room_size, damping=damping, wet_level=wet_level, dry_level=1.0 - wet_level,
        width=width, freeze_mode=0.0,
    )])


def eq_board_full(eq_type, gain, cutoff, q):
    return pb.Pedalboard([EQ_CLASS[eq_type](cutoff_frequency_hz=cutoff, gain_db=gain, q=q)])


# ------------------------------------------------------------------
# 2-D 쌍 정의: (name, kind, axis1_name, axis2_name, grid1, grid2, board_fn(v1,v2))
# ------------------------------------------------------------------
def build_2d_pairs():
    pairs = {}
    g_wet = lin_grid(*RANGES["wet_level"], N_2D)
    g_room = lin_grid(*RANGES["room_size"], N_2D)
    g_damp = lin_grid(*RANGES["damping"], N_2D)
    pairs["reverb_wet_room"] = {
        "axis1": "wet_level", "axis2": "room_size", "grid1": g_wet, "grid2": g_room,
        "board_fn": lambda v1, v2: reverb_board_full(v1, v2, 0.1, 0.7),
    }
    pairs["reverb_wet_damping"] = {
        "axis1": "wet_level", "axis2": "damping", "grid1": g_wet, "grid2": g_damp,
        "board_fn": lambda v1, v2: reverb_board_full(v1, 0.5, v2, 0.7),
    }
    pairs["reverb_room_damping"] = {
        "axis1": "room_size", "axis2": "damping", "grid1": g_room, "grid2": g_damp,
        "board_fn": lambda v1, v2: reverb_board_full(0.3, v1, v2, 0.7),
    }
    g_gain = lin_grid(*RANGES["gain"], N_2D)
    for eq_type in ["highshelf", "lowshelf", "peak"]:
        g_cutoff = log_grid(*RANGES[f"cutoff_{eq_type}"], N_2D)
        pairs[f"{eq_type}_gain_cutoff"] = {
            "axis1": "gain", "axis2": "cutoff", "grid1": g_gain, "grid2": g_cutoff,
            "board_fn": (lambda v1, v2, et=eq_type: eq_board_full(et, v1, v2, Q_MEDIAN)),
        }
    return pairs


# ------------------------------------------------------------------
# 3-D+ 수치전용 격자 정의
# ------------------------------------------------------------------
def build_3dplus():
    grids = {}
    g_gain5 = lin_grid(*RANGES["gain"], N_3D)
    g_q5 = log_grid(*RANGES["q"], N_3D)
    for eq_type in ["highshelf", "lowshelf", "peak"]:
        g_cutoff5 = log_grid(*RANGES[f"cutoff_{eq_type}"], N_3D)
        grids[f"{eq_type}_gain_cutoff_q"] = {
            "axes": ["gain", "cutoff", "q"], "grids": [g_gain5, g_cutoff5, g_q5],
            "board_fn": (lambda v, et=eq_type: eq_board_full(et, v[0], v[1], v[2])),
        }
    g_wet5 = lin_grid(*RANGES["wet_level"], N_3D)
    g_room5 = lin_grid(*RANGES["room_size"], N_3D)
    g_damp5 = lin_grid(*RANGES["damping"], N_3D)
    g_width5 = lin_grid(*RANGES["width"], N_3D)
    grids["reverb_wet_room_damping_width"] = {
        "axes": ["wet_level", "room_size", "damping", "width"],
        "grids": [g_wet5, g_room5, g_damp5, g_width5],
        "board_fn": (lambda v: reverb_board_full(v[0], v[1], v[2], v[3])),
    }
    return grids


def get_sources_1200():
    with open(RESULTS_DIR / "11_phase2_sources.json", encoding="utf-8") as f:
        base_sources = json.load(f)["sources"]
    with open(RESULTS_DIR / "11_phase2_sources_ext.json", encoding="utf-8") as f:
        ext_sources = json.load(f)["sources"]
    all_sources = sorted(base_sources + ext_sources, key=lambda s: s["src_id"])
    assert [s["src_id"] for s in all_sources] == list(range(1200))
    return all_sources


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t_script_start = time.time()

    sources = get_sources_1200()
    n_sources = len(sources)
    print(f"소스 {n_sources}개 (1,200 고정 목록 재사용)")

    print("CLAP 로딩...")
    torch_device = base.embed_mod.torch.device("cpu")
    clap = base.embed_mod.load_clap(torch_device, ROOT / "ckpts")

    print("소스 오디오 로딩...")
    y_dry = [base.embed_mod.load_and_preprocess(AUDIO_DIR / s["filename"]) for s in sources]
    for y in y_dry:
        assert y is not None

    def embed_batch(batch):
        return base.embed_mod.embed_batch(clap, torch_device, batch)

    def render_grid_2d(name, spec):
        cache_path = CACHE_DIR / f"11_phase3_2d_{name}.npz"
        if cache_path.exists():
            print(f"[2D] {name} — 캐시 존재, 건너뜀")
            return
        t0 = time.time()
        g1, g2 = spec["grid1"], spec["grid2"]
        emb = np.zeros((n_sources, len(g1), len(g2), 512), dtype=np.float32)
        batch_audio, batch_keys = [], []

        def flush():
            if not batch_audio:
                return
            e = embed_batch(batch_audio)
            for (si, i1, i2), v in zip(batch_keys, e):
                emb[si, i1, i2] = v
            batch_audio.clear()
            batch_keys.clear()

        n_jobs = n_sources * len(g1) * len(g2)
        print(f"[2D] {name} — {n_jobs}회 렌더링 시작")
        for si in range(n_sources):
            y = y_dry[si]
            for i1, v1 in enumerate(g1):
                for i2, v2 in enumerate(g2):
                    wet = spec["board_fn"](v1, v2)(y, SR)
                    batch_audio.append(wet)
                    batch_keys.append((si, i1, i2))
                    if len(batch_audio) >= BATCH_SIZE:
                        flush()
        flush()
        elapsed = time.time() - t0
        np.savez(cache_path, embeddings=emb, grid1=g1, grid2=g2,
                 axis1=spec["axis1"], axis2=spec["axis2"],
                 src_id=np.array([s["src_id"] for s in sources], dtype=np.int64), elapsed_sec=elapsed)
        print(f"[2D] {name} 완료 — {elapsed/60:.1f}분 (누적 {(time.time()-t_script_start)/60:.1f}분) 저장: {cache_path}")

    def render_grid_3dplus(name, spec):
        cache_path = CACHE_DIR / f"11_phase3_3dplus_{name}.npz"
        if cache_path.exists():
            print(f"[3D+] {name} — 캐시 존재, 건너뜀")
            return
        t0 = time.time()
        grids = spec["grids"]
        shape = tuple(len(g) for g in grids)
        emb = np.zeros((n_sources,) + shape + (512,), dtype=np.float32)
        batch_audio, batch_keys = [], []

        def flush():
            if not batch_audio:
                return
            e = embed_batch(batch_audio)
            for key, v in zip(batch_keys, e):
                emb[key] = v
            batch_audio.clear()
            batch_keys.clear()

        n_jobs = n_sources * int(np.prod(shape))
        print(f"[3D+] {name} — {n_jobs}회 렌더링 시작")
        idx_grids = np.array(np.meshgrid(*[np.arange(len(g)) for g in grids], indexing="ij")).reshape(len(grids), -1).T
        val_combos = [tuple(g[idx_grids[k, d]] for d, g in enumerate(grids)) for k in range(len(idx_grids))]
        for si in range(n_sources):
            y = y_dry[si]
            for k, idxs in enumerate(idx_grids):
                v = val_combos[k]
                wet = spec["board_fn"](v)(y, SR)
                batch_audio.append(wet)
                batch_keys.append((si,) + tuple(idxs))
                if len(batch_audio) >= BATCH_SIZE:
                    flush()
        flush()
        elapsed = time.time() - t0
        np.savez(cache_path, embeddings=emb, axes=np.array(spec["axes"]),
                 **{f"grid_{i}": g for i, g in enumerate(grids)},
                 src_id=np.array([s["src_id"] for s in sources], dtype=np.int64), elapsed_sec=elapsed)
        print(f"[3D+] {name} 완료 — {elapsed/60:.1f}분 (누적 {(time.time()-t_script_start)/60:.1f}분) 저장: {cache_path}")

    pairs_2d = build_2d_pairs()
    for name, spec in pairs_2d.items():
        render_grid_2d(name, spec)

    grids_3dplus = build_3dplus()
    for name, spec in grids_3dplus.items():
        render_grid_3dplus(name, spec)

    total_elapsed = time.time() - t_script_start
    meta = {
        "n_sources": n_sources, "n_2d_levels": N_2D, "n_3dplus_levels": N_3D,
        "pairs_2d": list(pairs_2d.keys()), "grids_3dplus": list(grids_3dplus.keys()),
        "q_median_fixed": Q_MEDIAN, "total_elapsed_sec": total_elapsed,
    }
    with open(RESULTS_DIR / "11_phase3_render_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n=== Phase 3 렌더링 전체 완료: {total_elapsed/60:.1f}분 ===")


if __name__ == "__main__":
    main()
