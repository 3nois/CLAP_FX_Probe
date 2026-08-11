"""CLAP FX Probe — 19_oat_render.py (6차 후속 family cosine: OAT 재렌더링)

2차(OAT, 7단계 스윕) 임베딩은 out/embeddings.npz가 gitignore 대상이면서 3차 이후
같은 파일명으로 덮어써져 디스크에서 소실됐다 (git에도 한 번도 커밋된 적 없음 —
확인 완료). family cosine 분석(v = e(최고 레벨) − e(최저 레벨))에 실제로 쓰는 것은
극단 레벨 2개뿐이라, 2차를 그대로 복제하지 않고 이번 분석에 맞게 재설계한다.

    2차 복제안   800 소스 x 7 레벨 x 3 이펙트 = 16,800회, 패밀리당 80
    이번 설계    1,200 소스 x 3 레벨 x 3 이펙트 = 10,800회, 패밀리당 120

within/between 분해의 검정력은 소스 다양성에서 나오므로 레벨 해상도를 줄이고 소스를
늘리는 쪽을 택했다. 중간 레벨(레벨 1) 하나는 비선형성 확인용으로만 남긴다.

조건 A(피크 0.7 정규화, 1~5차와 동일)를 쓴다 — 조건 C 아님, 기존 결과와 비교 가능해야
한다. reverb는 wet_level=0.4/dry_level=0.6/freeze_mode=0.0으로 고정하고 room_size만
스윕한다(대표축 근거: 4-R8에서 room_size R²=0.520이 wet_level R²=0.271보다 높음).

out/oat_emb.npz는 gitignore 대상이 아님(out/embeddings.npz만 명시적으로 제외됨) —
2차 데이터 소실이 "재생성 가능"이라는 이유로 gitignore된 파일이 실수로 유일한 사본이
된 데서 비롯됐으므로, 이번엔 커밋 대상으로 남긴다(약 22MB, GitHub 100MB 제한 이내).
"""
import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import librosa
import torch
from huggingface_hub import hf_hub_download
from pedalboard import Distortion, HighShelfFilter, Pedalboard, Reverb
from tqdm import tqdm

SAMPLE_RATE = 48000
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
PEAK_TARGET_A = 0.7
SILENCE_PEAK_THRESHOLD = 1e-4

CLAP_REPO_ID = "lukewys/laion_clap"
CLAP_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"

NSYNTH_SOURCE_TYPES = {"acoustic", "electronic", "synthetic"}

EFFECT_NAMES = ["reverb", "distortion", "highshelf"]
LEVEL_VALUES = {
    "reverb": [0.0, 0.25, 0.5],           # room_size
    "distortion": [0.0, 7.5, 15.0],       # drive_db
    "highshelf": [-9.0, 0.0, 9.0],        # gain_db
}
REVERB_WET_LEVEL = 0.4
REVERB_DRY_LEVEL = 0.6
HIGHSHELF_CUTOFF_HZ = 4000.0
N_SOURCES_PER_FAMILY_TARGET = 120
N_SOURCES_PER_FAMILY_MIN = 60


def parse_instrument_family(instrument: str) -> str:
    tokens = instrument.split("_")
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] in NSYNTH_SOURCE_TYPES:
            family = "_".join(tokens[:i])
            return family if family else instrument
    return instrument


def parse_nsynth_filename(path: Path):
    parts = path.stem.rsplit("-", 2)
    if len(parts) != 3:
        return path.stem, None, None
    instrument, pitch_str, velocity_str = parts
    try:
        return instrument, int(pitch_str), int(velocity_str)
    except ValueError:
        return instrument, None, None


def render_reverb(y, room_size):
    board = Pedalboard([Reverb(
        room_size=room_size, damping=0.5, wet_level=REVERB_WET_LEVEL, dry_level=REVERB_DRY_LEVEL,
        width=1.0, freeze_mode=0.0,
    )])
    return board(y, SAMPLE_RATE)


def render_distortion(y, drive_db):
    board = Pedalboard([Distortion(drive_db=drive_db)])
    return board(y, SAMPLE_RATE)


def render_highshelf(y, gain_db):
    board = Pedalboard([HighShelfFilter(cutoff_frequency_hz=HIGHSHELF_CUTOFF_HZ, gain_db=gain_db)])
    return board(y, SAMPLE_RATE)


RENDER_FN = {"reverb": render_reverb, "distortion": render_distortion, "highshelf": render_highshelf}


def load_and_preprocess_A(path: Path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) < NUM_SAMPLES:
        y = np.pad(y, (0, NUM_SAMPLES - len(y)))
    else:
        y = y[:NUM_SAMPLES]
    peak = float(np.abs(y).max())
    if peak < SILENCE_PEAK_THRESHOLD:
        return None
    return (y * (PEAK_TARGET_A / peak)).astype(np.float32)


def apply_condition_A(wet):
    peak = float(np.abs(wet).max())
    if peak > 1.0:
        wet = wet * (0.99 / peak)
    return wet.astype(np.float32)


def download_clap_checkpoint(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / CLAP_FILENAME
    if not ckpt_path.exists():
        hf_hub_download(repo_id=CLAP_REPO_ID, filename=CLAP_FILENAME, local_dir=cache_dir)
    return ckpt_path


def load_clap(device: torch.device, cache_dir: Path):
    import laion_clap
    ckpt_path = download_clap_checkpoint(cache_dir)
    try:
        clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    except TypeError:
        clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        clap = clap.to(device)
    clap.load_ckpt(str(ckpt_path), verbose=False)
    clap.eval()
    return clap


def embed_batch(clap, device, batch: list) -> np.ndarray:
    tensor = torch.tensor(np.stack(batch), dtype=torch.float32, device=device)
    with torch.no_grad():
        emb = clap.get_audio_embedding_from_data(tensor, use_tensor=True)
    return emb.cpu().numpy()


def select_sources(audio_dir: Path, n_target: int, n_min: int, seed: int):
    files = sorted(audio_dir.glob("*.wav"))
    by_family = collections.defaultdict(list)
    for f in files:
        instrument, pitch, velocity = parse_nsynth_filename(f)
        if pitch is None:
            continue
        fam = parse_instrument_family(instrument)
        by_family[fam].append((f.name, instrument))

    rng = np.random.RandomState(seed)
    selected = []  # (filename, instrument, family)
    family_counts = {}
    excluded_families = []
    for fam in sorted(by_family.keys()):
        pool = by_family[fam]
        if len(pool) < n_min:
            excluded_families.append(fam)
            continue
        n_take = min(n_target, len(pool))
        idx = rng.choice(len(pool), size=n_take, replace=False)
        for i in sorted(idx.tolist()):
            fname, instrument = pool[i]
            selected.append((fname, instrument, fam))
        family_counts[fam] = n_take
    return selected, family_counts, excluded_families


def main():
    parser = argparse.ArgumentParser(description="6차 후속 family cosine — OAT 재렌더링 (신규 설계)")
    parser.add_argument("--audio-dir", type=str, default="nsynth-test/audio")
    parser.add_argument("--n-per-family", type=int, default=N_SOURCES_PER_FAMILY_TARGET)
    parser.add_argument("--n-per-family-min", type=int, default=N_SOURCES_PER_FAMILY_MIN)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.out)
    audio_dir = Path(args.audio_dir)

    print("소스 선정 중 (패밀리 균형)...")
    selected, family_counts, excluded_families = select_sources(audio_dir, args.n_per_family, args.n_per_family_min, args.seed)
    n_sources = len(selected)
    print(f"소스 {n_sources}개 선정. 패밀리별: {family_counts}")
    if excluded_families:
        print(f"제외된 패밀리(최소 {args.n_per_family_min}개 미만): {excluded_families}")

    print("오디오 로딩 및 조건 A 전처리 중...")
    y_dry_by_idx = {}
    src_id_arr, family_arr, instrument_arr = [], [], []
    for i, (fname, instrument, fam) in enumerate(tqdm(selected, desc="오디오 로딩")):
        y = load_and_preprocess_A(audio_dir / fname)
        if y is None:
            continue  # 무음 — 건너뜀 (아래에서 실제 반영된 소스만 최종 배열에 포함)
        y_dry_by_idx[i] = y
        src_id_arr.append(i)
        family_arr.append(fam)
        instrument_arr.append(instrument)
    n_silent_excluded = n_sources - len(src_id_arr)
    n_sources_final = len(src_id_arr)
    print(f"무음 제외 {n_silent_excluded}개. 최종 소스 {n_sources_final}개.")

    device = torch.device(args.device)
    if args.device == "mps":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print("CLAP 모델 로딩 중...")
    clap = load_clap(device, Path(__file__).parent / "ckpts")

    emb = np.zeros((n_sources_final, len(EFFECT_NAMES), 3, 512), dtype=np.float32)
    idx_map = {orig_i: k for k, orig_i in enumerate(src_id_arr)}

    n_jobs = n_sources_final * len(EFFECT_NAMES) * 3
    print(f"렌더링+임베딩 시작: {n_sources_final} 소스 x {len(EFFECT_NAMES)} 이펙트 x 3 레벨 = {n_jobs}회")

    batch_audio, batch_keys = [], []

    def flush():
        if not batch_audio:
            return
        e = embed_batch(clap, device, batch_audio)
        for (k, ei, li), v in zip(batch_keys, e):
            emb[k, ei, li] = v
        batch_audio.clear()
        batch_keys.clear()

    pbar = tqdm(total=n_jobs, desc="렌더링+임베딩")
    for orig_i in src_id_arr:
        k = idx_map[orig_i]
        y = y_dry_by_idx[orig_i]
        for ei, effect in enumerate(EFFECT_NAMES):
            for li, level in enumerate(LEVEL_VALUES[effect]):
                wet_raw = RENDER_FN[effect](y, level)
                wet = apply_condition_A(wet_raw)
                batch_audio.append(wet)
                batch_keys.append((k, ei, li))
                if len(batch_audio) >= args.batch_size:
                    flush()
                pbar.update(1)
    flush()
    pbar.close()

    npz_path = out_dir / "oat_emb.npz"
    np.savez(
        npz_path,
        emb=emb,
        src_id=np.array(src_id_arr, dtype=np.int64),
        instrument_family=np.array(family_arr),
        instrument_name=np.array(instrument_arr),
        effect_names=np.array(EFFECT_NAMES),
        level_values_reverb=np.array(LEVEL_VALUES["reverb"]),
        level_values_distortion=np.array(LEVEL_VALUES["distortion"]),
        level_values_highshelf=np.array(LEVEL_VALUES["highshelf"]),
    )
    print(f"저장: {npz_path}")

    elapsed = time.time() - t_start
    meta = {
        "render_spec": {
            "n_sources_target_per_family": args.n_per_family,
            "n_sources_min_per_family": args.n_per_family_min,
            "family_counts_actual": family_counts,
            "excluded_families": excluded_families,
            "n_sources_selected": n_sources,
            "n_silent_excluded": n_silent_excluded,
            "n_sources_final": n_sources_final,
            "condition": "A (피크 0.7 정규화, 이펙트 후 peak>1.0이면 0.99/peak 재정규화)",
            "sample_rate": SAMPLE_RATE, "duration_sec": DURATION_SEC, "peak_target": PEAK_TARGET_A,
            "effect_axes": {
                "reverb": {"swept_param": "room_size", "levels": LEVEL_VALUES["reverb"],
                           "fixed": {"wet_level": REVERB_WET_LEVEL, "dry_level": REVERB_DRY_LEVEL,
                                     "damping": 0.5, "width": 1.0, "freeze_mode": 0.0}},
                "distortion": {"swept_param": "drive_db", "levels": LEVEL_VALUES["distortion"], "fixed": {}},
                "highshelf": {"swept_param": "gain_db", "levels": LEVEL_VALUES["highshelf"],
                              "fixed": {"cutoff_frequency_hz": HIGHSHELF_CUTOFF_HZ, "q": "pedalboard 기본값(미지정)"}},
            },
            "n_render_jobs": n_jobs, "elapsed_sec": elapsed, "seed": args.seed,
        }
    }
    with open(out_dir / "oat_emb_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n완료: {n_jobs}회 렌더링, 소요 {elapsed/60:.1f}분")
    print(f"저장: {npz_path}, {out_dir / 'oat_emb_meta.json'}")


if __name__ == "__main__":
    main()
