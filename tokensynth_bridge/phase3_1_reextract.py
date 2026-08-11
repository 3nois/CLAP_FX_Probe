"""9차 Phase 3-1 — TokenSynth 임베딩 공간으로 재추출.

Phase 2-A에서 우리 파이프라인(48kHz 직접) vs TokenSynth `clap.encode_audio()`
(16kHz 로드 -> 48kHz 업샘플)가 같은 오디오에서 cos=0.7035로 실질적으로 다름을
확인했다. TokenSynth에 주입할 것이므로 이제부터는 TokenSynth 공간을 쓴다.

7차(19_oat_render.py)와 완전히 동일한 설계·시드·렌더링(pedalboard) 코드를 그대로
쓰고, **임베딩 추출 단계만** 교체한다: 48kHz 렌더 결과를 16kHz로 다운샘플한 뒤 다시
48kHz로 업샘플(TokenSynth clap.py의 encode_audio와 동일 연산)하고 나서 CLAP에 넣는다.
이 배치판 재현이 실제 encode_audio(파일경로) 호출과 cos=0.9999999999989999로
사실상 동일함을 사전에 확인했다(파일 I/O 없이 배치 처리해 속도만 높인 것).

out/oat_emb.npz는 건드리지 않는다 — 새 파일은 out/caches/oat_emb_ts.npz.
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
    "reverb": [0.0, 0.25, 0.5],
    "distortion": [0.0, 7.5, 15.0],
    "highshelf": [-9.0, 0.0, 9.0],
}
REVERB_WET_LEVEL = 0.4
REVERB_DRY_LEVEL = 0.6
HIGHSHELF_CUTOFF_HZ = 4000.0
N_SOURCES_PER_FAMILY_TARGET = 120
N_SOURCES_PER_FAMILY_MIN = 60

TS_ENCODE_SR = 16000  # TokenSynth clap.py encode_audio()의 중간 샘플레이트


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


def to_tokensynth_space(wet_48k: np.ndarray) -> np.ndarray:
    """TokenSynth clap.py encode_audio()의 16k->48k 왕복을 재현 (파일 I/O 없이).
    사전 검증: 실제 encode_audio(파일경로)와 cos=0.9999999999989999로 일치."""
    y16 = librosa.resample(wet_48k, orig_sr=SAMPLE_RATE, target_sr=TS_ENCODE_SR)
    y48_recon = librosa.resample(y16, orig_sr=TS_ENCODE_SR, target_sr=SAMPLE_RATE)
    return y48_recon.astype(np.float32)


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
    selected = []
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
    parser = argparse.ArgumentParser(description="9차 Phase 3-1 — TokenSynth 임베딩 공간 재추출 (7차와 동일 렌더링)")
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
    caches_dir = out_dir / "caches"
    config_dir = out_dir / "config"
    caches_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir)

    print("소스 선정 중 (7차와 동일 시드 -> 동일 1,200소스)...")
    selected, family_counts, excluded_families = select_sources(audio_dir, args.n_per_family, args.n_per_family_min, args.seed)
    n_sources = len(selected)
    print(f"소스 {n_sources}개 선정. 패밀리별: {family_counts}")

    print("오디오 로딩 및 조건 A 전처리 중...")
    y_dry_by_idx = {}
    src_id_arr, family_arr, instrument_arr = [], [], []
    for i, (fname, instrument, fam) in enumerate(tqdm(selected, desc="오디오 로딩")):
        y = load_and_preprocess_A(audio_dir / fname)
        if y is None:
            continue
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
    clap = load_clap(device, Path(__file__).resolve().parent.parent / "ckpts")

    emb = np.zeros((n_sources_final, len(EFFECT_NAMES), 3, 512), dtype=np.float32)
    idx_map = {orig_i: k for k, orig_i in enumerate(src_id_arr)}

    n_jobs = n_sources_final * len(EFFECT_NAMES) * 3
    print(f"렌더링+TokenSynth식 재샘플+임베딩: {n_sources_final} 소스 x {len(EFFECT_NAMES)} 이펙트 x 3 레벨 = {n_jobs}회")

    batch_audio, batch_keys = [], []

    def flush():
        if not batch_audio:
            return
        e = embed_batch(clap, device, batch_audio)
        for (k, ei, li), v in zip(batch_keys, e):
            emb[k, ei, li] = v
        batch_audio.clear()
        batch_keys.clear()

    pbar = tqdm(total=n_jobs, desc="렌더링+재샘플+임베딩")
    for orig_i in src_id_arr:
        k = idx_map[orig_i]
        y = y_dry_by_idx[orig_i]
        for ei, effect in enumerate(EFFECT_NAMES):
            for li, level in enumerate(LEVEL_VALUES[effect]):
                wet_raw = RENDER_FN[effect](y, level)
                wet = apply_condition_A(wet_raw)
                wet_ts = to_tokensynth_space(wet)  # ★ 유일한 변경점 — TokenSynth 임베딩 공간
                batch_audio.append(wet_ts)
                batch_keys.append((k, ei, li))
                if len(batch_audio) >= args.batch_size:
                    flush()
                pbar.update(1)
    flush()
    pbar.close()

    npz_path = caches_dir / "oat_emb_ts.npz"
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
            "note": "7차(19_oat_render.py)와 동일 렌더링·시드. 임베딩 추출만 TokenSynth 공간(16k<->48k 왕복)으로 교체.",
            "n_sources_final": n_sources_final, "n_silent_excluded": n_silent_excluded,
            "family_counts_actual": family_counts, "excluded_families": excluded_families,
            "condition": "A (피크 0.7 정규화, 이펙트 후 peak>1.0이면 0.99/peak 재정규화)",
            "embedding_space": "TokenSynth (16kHz 다운샘플 -> 48kHz 업샘플 후 CLAP, clap.py encode_audio()와 동치 확인됨 cos=0.9999999999989999)",
            "effect_axes": {
                "reverb": {"swept_param": "room_size", "levels": LEVEL_VALUES["reverb"],
                           "fixed": {"wet_level": REVERB_WET_LEVEL, "dry_level": REVERB_DRY_LEVEL, "damping": 0.5, "width": 1.0, "freeze_mode": 0.0}},
                "distortion": {"swept_param": "drive_db", "levels": LEVEL_VALUES["distortion"], "fixed": {}},
                "highshelf": {"swept_param": "gain_db", "levels": LEVEL_VALUES["highshelf"], "fixed": {"cutoff_frequency_hz": HIGHSHELF_CUTOFF_HZ}},
            },
            "n_render_jobs": n_jobs, "elapsed_sec": elapsed, "seed": args.seed,
        }
    }
    with open(config_dir / "oat_emb_ts_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n완료: {n_jobs}회, 소요 {elapsed/60:.1f}분")
    print(f"저장: {npz_path}, {config_dir / 'oat_emb_ts_meta.json'}")


if __name__ == "__main__":
    main()
