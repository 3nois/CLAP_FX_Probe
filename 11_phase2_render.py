# -*- coding: utf-8 -*-
"""Phase 2 — 밀집 격자 렌더링 (11차 지시서, out/prereg/11_phase1.md 승인 설계).

원칙 1: 축마다 실무 범위 전체를 25레벨 밀집 격자로 렌더링하고 캐시한다. 범위별
R²/JND/windowed R² 질의는 전부 이 캐시에 대한 사후 분석(별도 스크립트)에서 낸다.

400소스(패밀리당 40, 고정 시드) x 23축(주축 21 + 널축 2) x 25레벨 = 230,000회
+ 소스당 bypass(진짜 dry) 1회 = 400회.

체크포인트: 축 단위로 out/caches/11_phase2_<axis>.npz 저장. 이미 존재하는 축 파일은
건너뛴다(재실행 시 이어서 진행). 소스 목록은 out/results/11_phase2_sources.json에
먼저 고정 저장하고, 이후 모든 축이 그 목록을 그대로 재사용한다.
"""
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import pedalboard as pb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

embed_mod = import_module("01_embed")

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "nsynth-test" / "audio"
CACHE_DIR = ROOT / "out" / "caches"
RESULTS_DIR = ROOT / "out" / "results"
LOG_DIR = ROOT / "out" / "logs"
SR = 48000
N_LEVELS = 25
N_PER_FAMILY = 40
SOURCE_SEED = 0
BATCH_SIZE = 32
Q_DEFAULT = 0.7071067811865476

NSYNTH_SOURCE_TYPES = {"acoustic", "electronic", "synthetic"}


def parse_instrument_family(instrument: str) -> str:
    tokens = instrument.split("_")
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] in NSYNTH_SOURCE_TYPES:
            fam = "_".join(tokens[:i])
            return fam if fam else instrument
    return instrument


def parse_nsynth_filename(path: Path):
    parts = path.stem.rsplit("-", 2)
    if len(parts) != 3:
        return path.stem, None, None
    instrument, pitch_str, vel_str = parts
    try:
        return instrument, int(pitch_str), int(vel_str)
    except ValueError:
        return instrument, None, None


def select_sources_stratified(n_per_family, seed):
    files = sorted(AUDIO_DIR.glob("*.wav"))
    by_family = collections.defaultdict(list)
    for f in files:
        instrument, pitch, velocity = parse_nsynth_filename(f)
        if pitch is None:
            continue
        fam = parse_instrument_family(instrument)
        by_family[fam].append((f.name, instrument))
    rng = np.random.RandomState(seed)
    selected = []
    for fam in sorted(by_family.keys()):
        pool = by_family[fam]
        n_take = min(n_per_family, len(pool))
        idx = rng.choice(len(pool), size=n_take, replace=False)
        for i in sorted(idx.tolist()):
            fname, instrument = pool[i]
            selected.append((fname, instrument, fam))
    return selected


# ----------------------------------------------------------------------
# 축 정의 (out/prereg/11_phase1.md §4.1, §5 최종안 — cutoff/q는 gain=+-6dB 두 벌)
# ----------------------------------------------------------------------
def eq_board(cls, cutoff, gain, q):
    return pb.Pedalboard([cls(cutoff_frequency_hz=cutoff, gain_db=gain, q=q)])


def reverb_board(room_size, damping, wet_level, width):
    return pb.Pedalboard([pb.Reverb(
        room_size=room_size, damping=damping, wet_level=wet_level, dry_level=1.0 - wet_level,
        width=width, freeze_mode=0.0,
    )])


AXES = {}

# 1. distortion (Koo 범위 그대로 0~20dB, theta_min=0 삽입비용은 사전등록에서 별도 보고)
AXES["distortion_drive_db"] = {
    "levels": np.linspace(0.0, 20.0, N_LEVELS),
    "board_fn": lambda v: pb.Pedalboard([pb.Distortion(drive_db=float(v))]),
}

# 2~5. reverb 4축
AXES["reverb_wet_level"] = {
    "levels": np.linspace(0.0, 0.5, N_LEVELS),
    "board_fn": lambda v: reverb_board(room_size=0.5, damping=0.1, wet_level=float(v), width=0.7),
}
AXES["reverb_room_size"] = {
    "levels": np.linspace(0.05, 0.85, N_LEVELS),
    "board_fn": lambda v: reverb_board(room_size=float(v), damping=0.1, wet_level=0.3, width=0.7),
}
AXES["reverb_damping"] = {
    "levels": np.linspace(0.0, 1.0, N_LEVELS),
    "board_fn": lambda v: reverb_board(room_size=0.5, damping=float(v), wet_level=0.3, width=0.7),
}
AXES["reverb_width"] = {
    "levels": np.linspace(0.0, 1.0, N_LEVELS),
    "board_fn": lambda v: reverb_board(room_size=0.5, damping=0.1, wet_level=0.3, width=float(v)),
}

# 6~8. EQ gain 축 (자체 스윕, 게이팅 없음, 대표 cutoff 고정)
EQ_TYPES = {
    "highshelf": (pb.HighShelfFilter, 2000.0, (500.0, 4000.0)),
    "lowshelf": (pb.LowShelfFilter, 100.0, (30.0, 200.0)),
    "peak": (pb.PeakFilter, 1000.0, (200.0, 6000.0)),
}
for name, (cls, rep_cutoff, cutoff_range) in EQ_TYPES.items():
    AXES[f"{name}_gain"] = {
        "levels": np.linspace(-15.0, 15.0, N_LEVELS),
        "board_fn": (lambda v, cls=cls, rep_cutoff=rep_cutoff: eq_board(cls, rep_cutoff, float(v), Q_DEFAULT)),
    }
    for sign, gval in [("gp6", 6.0), ("gn6", -6.0)]:
        AXES[f"{name}_cutoff_{sign}"] = {
            "levels": np.geomspace(cutoff_range[0], cutoff_range[1], N_LEVELS),
            "board_fn": (lambda v, cls=cls, gval=gval: eq_board(cls, float(v), gval, Q_DEFAULT)),
        }
        AXES[f"{name}_q_{sign}"] = {
            "levels": np.geomspace(0.1, 2.0, N_LEVELS),
            "board_fn": (lambda v, cls=cls, gval=gval, rep_cutoff=rep_cutoff: eq_board(cls, rep_cutoff, gval, float(v))),
        }

# 21. eq_cascade_intensity — 소스별 고정 시드는 렌더링 루프에서 src idx로 주입한다
CASCADE_BAND_FREQS = {"low_shelf": 100.0, "first_band": 400.0, "second_band": 2000.0,
                       "third_band": 3000.0, "high_shelf": 6500.0}
# 2026-08-13 정정(사용자 지시): 이전 값(third_band=4000, high_shelf=3500)은 high_shelf가
# third_band보다 낮아 5밴드 순서가 역전되는 결함이 있었다. Koo 범위(third_band
# 3000~8000, high_shelf 5000~10000)와 Nyquist(8000Hz) 사이에서 순서를 보존하도록
# 재조정: 100 < 400 < 2000 < 3000 < 6500 < 8000(Nyquist). out/prereg/11_phase1.md
# §3.1 addendum 참고.


def cascade_board(gains, s):
    return pb.Pedalboard([
        pb.LowShelfFilter(cutoff_frequency_hz=CASCADE_BAND_FREQS["low_shelf"], gain_db=s * gains["low_shelf"], q=Q_DEFAULT),
        pb.PeakFilter(cutoff_frequency_hz=CASCADE_BAND_FREQS["first_band"], gain_db=s * gains["first_band"], q=0.7),
        pb.PeakFilter(cutoff_frequency_hz=CASCADE_BAND_FREQS["second_band"], gain_db=s * gains["second_band"], q=0.7),
        pb.PeakFilter(cutoff_frequency_hz=CASCADE_BAND_FREQS["third_band"], gain_db=s * gains["third_band"], q=0.7),
        pb.HighShelfFilter(cutoff_frequency_hz=CASCADE_BAND_FREQS["high_shelf"], gain_db=s * gains["high_shelf"], q=Q_DEFAULT),
    ])


AXES["eq_cascade_intensity"] = {
    "levels": np.linspace(0.0, 1.0, N_LEVELS),
    "board_fn": None,  # 소스별 seed 필요 — 렌더링 루프에서 특별 처리
}

# 22~23. 널 축 (ultrasonic shelf, 단독)
AXES["null_12k_gain"] = {
    "levels": np.linspace(-15.0, 15.0, N_LEVELS),
    "board_fn": lambda v: eq_board(pb.HighShelfFilter, 12000.0, float(v), Q_DEFAULT),
}
AXES["null_15k_gain"] = {
    "levels": np.linspace(-15.0, 15.0, N_LEVELS),
    "board_fn": lambda v: eq_board(pb.HighShelfFilter, 15000.0, float(v), Q_DEFAULT),
}

AXIS_ORDER = list(AXES.keys())


def get_or_select_sources():
    sources_path = RESULTS_DIR / "11_phase2_sources.json"
    if sources_path.exists():
        with open(sources_path, encoding="utf-8") as f:
            data = json.load(f)
        return data["sources"]
    selected = select_sources_stratified(N_PER_FAMILY, SOURCE_SEED)
    fam_counts = collections.Counter(fam for _, _, fam in selected)
    sources = [{"src_id": i, "filename": fn, "instrument": inst, "family": fam}
               for i, (fn, inst, fam) in enumerate(selected)]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(sources_path, "w", encoding="utf-8") as f:
        json.dump({"seed": SOURCE_SEED, "n_per_family": N_PER_FAMILY,
                   "family_counts": dict(fam_counts), "n_total": len(sources),
                   "sources": sources}, f, indent=2, ensure_ascii=False)
    print(f"소스 목록 신규 확정 및 저장: {sources_path} ({len(sources)}개)")
    return sources


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t_script_start = time.time()

    sources = get_or_select_sources()
    n_sources = len(sources)
    print(f"소스 {n_sources}개 확정")

    print("CLAP 로딩...")
    torch_device = embed_mod.torch.device("cpu")
    clap = embed_mod.load_clap(torch_device, ROOT / "ckpts")

    print("소스 오디오 로딩 (dry, 조건 A)...")
    y_dry = []
    for s in sources:
        y = embed_mod.load_and_preprocess(AUDIO_DIR / s["filename"])
        assert y is not None, f"무음 소스 발견 — 사전 선정 단계에서 걸러졌어야 함: {s['filename']}"
        y_dry.append(y)

    def embed_batch(batch):
        return embed_mod.embed_batch(clap, torch_device, batch)

    # ------------------------------------------------------------
    # bypass 임베딩 (진짜 dry, 모든 축의 공통 기준점) — 캐시 있으면 재사용
    # ------------------------------------------------------------
    bypass_path = CACHE_DIR / "11_phase2_bypass.npz"
    if bypass_path.exists():
        print(f"bypass 캐시 재사용: {bypass_path}")
    else:
        print("bypass(진짜 dry) 임베딩 계산 중...")
        batch, emb_list = [], []
        for i in range(0, n_sources, BATCH_SIZE):
            chunk = y_dry[i:i + BATCH_SIZE]
            emb_list.append(embed_batch(chunk))
        bypass_emb = np.concatenate(emb_list, axis=0).astype(np.float32)
        np.savez(bypass_path, embeddings=bypass_emb,
                 src_id=np.array([s["src_id"] for s in sources], dtype=np.int64))
        print(f"저장: {bypass_path}")

    # ------------------------------------------------------------
    # 축별 렌더링 (체크포인트: 이미 존재하는 축 파일은 건너뜀)
    # ------------------------------------------------------------
    cascade_seeds_path = RESULTS_DIR / "11_phase2_cascade_seeds.json"
    cascade_gains_by_src = {}
    if cascade_seeds_path.exists():
        with open(cascade_seeds_path, encoding="utf-8") as f:
            saved = json.load(f)
        cascade_gains_by_src = {int(k): v for k, v in saved.items()}
    else:
        for s in sources:
            rng = np.random.default_rng(42 + s["src_id"])
            gains = {b: float(rng.uniform(-15, 15)) for b in CASCADE_BAND_FREQS}
            cascade_gains_by_src[s["src_id"]] = gains
        with open(cascade_seeds_path, "w", encoding="utf-8") as f:
            json.dump(cascade_gains_by_src, f, indent=2, ensure_ascii=False)
        print(f"cascade 소스별 시드 확정 저장: {cascade_seeds_path}")

    total_axes = len(AXIS_ORDER)
    for axis_i, axis_name in enumerate(AXIS_ORDER):
        axis_cache_path = CACHE_DIR / f"11_phase2_{axis_name}.npz"
        if axis_cache_path.exists():
            print(f"[{axis_i+1}/{total_axes}] {axis_name} — 캐시 존재, 건너뜀")
            continue

        t_axis_start = time.time()
        spec = AXES[axis_name]
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
                    board = cascade_board(gains, float(sv))
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
        "n_sources": n_sources, "n_per_family": N_PER_FAMILY, "source_seed": SOURCE_SEED,
        "n_levels": N_LEVELS, "batch_size": BATCH_SIZE, "axes": AXIS_ORDER,
        "n_axes": total_axes, "total_conditions": total_axes * n_sources * N_LEVELS,
        "total_elapsed_sec": total_elapsed,
    }
    with open(RESULTS_DIR / "11_phase2_render_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n=== Phase 2 렌더링 전체 완료: {total_elapsed/60:.1f}분 ===")


if __name__ == "__main__":
    main()
