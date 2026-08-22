# -*- coding: utf-8 -*-
"""Phase 2 JND 정밀 측정 — 사용자 지시 §3.

25레벨 격자 한 칸이 이미 널 바닥을 넘어 "JND < 한 칸"만 확인됐다. theta_min 근방을
로그 간격 20단계로 재훑어 진짜 JND를 찾는다. 부호 축(*_gain)은 양/음 방향 각각
20단계(합 40단계). theta_min과 기존 격자의 두 번째 점은 이미 캐시에 있으므로
재렌더링하지 않고, 그 사이 새 로그 격자점만 렌더링한다.

대상 15축(주축, gn6 제외) — 지시서는 "14개"라고 했으나 실제로 세면 distortion(1)
+reverb(4)+eq_gain(3)+eq_cutoff_gp6(3)+eq_q_gp6(3)+cascade(1)=15개다(gn6 6개 제외).
14는 계산 착오로 보고 15개 전부 처리한다.

소스: 기존 400 중 30/family=300 고정 서브샘플(seed=2, 기록). CI가 필요한 스칼라가
아니라 문턱값 추정이므로 1,200 불필요하다는 지시 근거를 그대로 따른다.
"""
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
N_FINE = 20
N_PER_FAMILY_JND = 30
JND_SOURCE_SEED = 2

GAIN_AXES = ["highshelf_gain", "lowshelf_gain", "peak_gain"]
ONE_DIR_AXES = [
    "distortion_drive_db", "reverb_wet_level", "reverb_room_size", "reverb_damping", "reverb_width",
    "highshelf_cutoff_gp6", "lowshelf_cutoff_gp6", "peak_cutoff_gp6",
    "highshelf_q_gp6", "lowshelf_q_gp6", "peak_q_gp6",
    "eq_cascade_intensity",
]
JND_AXES = ONE_DIR_AXES + GAIN_AXES
assert len(JND_AXES) == 15


def theta_min_index(axis_name, theta_raw):
    if axis_name in GAIN_AXES:
        return int(np.argmin(np.abs(theta_raw)))
    return 0


def select_jnd_sources():
    src_path = RESULTS_DIR / "11_phase2jnd_sources.json"
    if src_path.exists():
        with open(src_path, encoding="utf-8") as f:
            return json.load(f)["src_ids"]
    with open(RESULTS_DIR / "11_phase2_sources.json", encoding="utf-8") as f:
        all_sources = json.load(f)["sources"]
    by_family = {}
    for s in all_sources:
        by_family.setdefault(s["family"], []).append(s["src_id"])
    rng = np.random.RandomState(JND_SOURCE_SEED)
    chosen = []
    for fam in sorted(by_family.keys()):
        pool = sorted(by_family[fam])
        n_take = min(N_PER_FAMILY_JND, len(pool))
        idx = rng.choice(len(pool), size=n_take, replace=False)
        chosen.extend(int(pool[i]) for i in sorted(idx.tolist()))
    chosen = sorted(chosen)
    with open(src_path, "w", encoding="utf-8") as f:
        json.dump({"seed": JND_SOURCE_SEED, "n_per_family": N_PER_FAMILY_JND,
                   "n_total": len(chosen), "src_ids": chosen}, f, indent=2, ensure_ascii=False)
    print(f"JND 소스 목록 확정: {src_path} ({len(chosen)}개)")
    return chosen


def build_fine_grid(theta_min, existing_next, n=N_FINE):
    """theta_min과 existing_next 사이, 양쪽 다 제외한 로그 20단계 (오름차순 거리)."""
    span = existing_next - theta_min
    eps = span * 1e-3
    offsets = np.geomspace(eps, span * (1 - 1e-3), n)
    return theta_min + offsets  # 오름차순, theta_min < ... < existing_next


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t_script_start = time.time()

    src_ids = select_jnd_sources()
    n_sources = len(src_ids)
    print(f"JND 소스 {n_sources}개 확정: {src_ids[:5]}...")

    with open(RESULTS_DIR / "11_phase2_sources.json", encoding="utf-8") as f:
        all_sources = {s["src_id"]: s for s in json.load(f)["sources"]}
    filenames = [all_sources[sid]["filename"] for sid in src_ids]

    print("CLAP 로딩...")
    torch_device = base.embed_mod.torch.device("cpu")
    clap = base.embed_mod.load_clap(torch_device, ROOT / "ckpts")

    print("JND 소스 오디오 로딩...")
    y_dry = [base.embed_mod.load_and_preprocess(AUDIO_DIR / fn) for fn in filenames]
    for y in y_dry:
        assert y is not None

    def embed_batch(batch):
        return base.embed_mod.embed_batch(clap, torch_device, batch)

    cascade_gains_by_src = None

    total_axes = len(JND_AXES)
    for axis_i, axis_name in enumerate(JND_AXES):
        axis_cache_path = CACHE_DIR / f"11_phase2jnd_{axis_name}.npz"
        if axis_cache_path.exists():
            print(f"[{axis_i+1}/{total_axes}] {axis_name} — 캐시 존재, 건너뜀")
            continue

        base_cache = np.load(CACHE_DIR / f"11_phase2_{axis_name}.npz")
        theta_raw_orig = base_cache["theta_raw"]
        idx0 = theta_min_index(axis_name, theta_raw_orig)
        theta_min_val = float(theta_raw_orig[idx0])

        directions = []
        if axis_name in GAIN_AXES:
            directions.append(("pos", theta_min_val, float(theta_raw_orig[idx0 + 1])))
            directions.append(("neg", theta_min_val, float(theta_raw_orig[idx0 - 1])))
        else:
            directions.append(("pos", theta_min_val, float(theta_raw_orig[idx0 + 1])))

        t_axis_start = time.time()
        result = {}
        for dir_name, t_min, t_next in directions:
            fine_grid = build_fine_grid(t_min, t_next)
            n_levels = len(fine_grid)
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

            print(f"[{axis_i+1}/{total_axes}] {axis_name} ({dir_name}) — "
                  f"{n_sources*n_levels}회 렌더링 시작 (theta {t_min:.4g}~{t_next:.4g})")

            for si in range(n_sources):
                y = y_dry[si]
                if axis_name == "eq_cascade_intensity":
                    if cascade_gains_by_src is None:
                        with open(RESULTS_DIR / "11_phase2_cascade_seeds.json", encoding="utf-8") as f:
                            cascade_gains_by_src = {int(k): v for k, v in json.load(f).items()}
                    gains = cascade_gains_by_src[src_ids[si]]
                    for li, sv in enumerate(fine_grid):
                        board = base.cascade_board(gains, float(sv))
                        wet = board(y, SR)
                        batch_audio.append(wet)
                        batch_keys.append((si, li))
                        if len(batch_audio) >= BATCH_SIZE:
                            flush()
                else:
                    board_fn = base.AXES[axis_name]["board_fn"]
                    for li, lv in enumerate(fine_grid):
                        board = board_fn(lv)
                        wet = board(y, SR)
                        batch_audio.append(wet)
                        batch_keys.append((si, li))
                        if len(batch_audio) >= BATCH_SIZE:
                            flush()
            flush()
            result[dir_name] = {"theta_fine": fine_grid, "embeddings": emb}

        elapsed_axis = time.time() - t_axis_start
        save_kwargs = {"src_id": np.array(src_ids, dtype=np.int64), "theta_min": theta_min_val,
                        "elapsed_sec": elapsed_axis}
        for dir_name in result:
            save_kwargs[f"theta_fine_{dir_name}"] = result[dir_name]["theta_fine"]
            save_kwargs[f"embeddings_{dir_name}"] = result[dir_name]["embeddings"]
        np.savez(axis_cache_path, **save_kwargs)
        elapsed_total = time.time() - t_script_start
        print(f"[{axis_i+1}/{total_axes}] {axis_name} 완료 — {elapsed_axis/60:.1f}분 "
              f"(누적 {elapsed_total/60:.1f}분) 저장: {axis_cache_path}")

    total_elapsed = time.time() - t_script_start
    with open(RESULTS_DIR / "11_phase2jnd_render_meta.json", "w", encoding="utf-8") as f:
        json.dump({"n_sources": n_sources, "n_per_family": N_PER_FAMILY_JND, "seed": JND_SOURCE_SEED,
                   "axes": JND_AXES, "n_fine": N_FINE, "total_elapsed_sec": total_elapsed}, f, indent=2, ensure_ascii=False)
    print(f"\n=== JND 정밀 렌더링 완료: {total_elapsed/60:.1f}분 ===")


if __name__ == "__main__":
    main()
