"""CLAP FX Probe — 01_embed.py

NSynth 오디오에 Reverb/Distortion/HighShelf 이펙트를 걸어가며 CLAP 임베딩을 추출한다.
오디오는 디스크에 쓰지 않고 메모리에서 처리한 뒤 임베딩만 저장한다.

이펙트별 스윕 범위(EFFECT_SPECS)는 실무에서 흔히 쓰는 세기로 설정했다 — 예를 들어
아무도 EQ를 ±60dB씩 걸지 않는다. "오디오 도메인 거리로 강도를 기계적으로 맞추는" 접근은
쓰지 않는다: 그건 "단위 음향 변화당 인코딩 효율"이라는 다른 질문에 답할 뿐, 실무에서
실제로 쓰는 세기에서 얼마나 잘 작동하는지는 알려주지 않는다.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import librosa
import torch
from huggingface_hub import hf_hub_download
from pedalboard import Distortion, HighShelfFilter, Pedalboard, Reverb
from tqdm import tqdm

SAMPLE_RATE = 48000  # LAION-CLAP이 기대하는 샘플레이트
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
PEAK_TARGET = 0.7
SILENCE_PEAK_THRESHOLD = 1e-4

# 이펙트별 파라미터 범위와 레벨 수 — 실무에서 흔히 쓰는 세기로 잡는다 (60dB EQ를 거는
# 사람은 없다). "스윕 강도를 기계적으로 맞추는" 대신, 애초에 각 이펙트의 스윕 범위를
# 실무 상식선으로 설정해 이펙트 간 성적 차이가 비현실적인 강도 차이 때문이 아니게 한다.
# reverb/distortion은 범위의 첫 값이 "무효과"에 해당해 dry와 자연스럽게 이어진다.
# highshelf는 gain_db=0이 무효과 지점이며, 대칭 범위(-9~+9)의 중간에 위치한다.
EFFECT_SPECS = {
    "reverb": {"param_range": (0.0, 0.5), "n_levels": 7},  # room_size: 무반향~중대형 룸 (카세드럴급 0.9는 제외)
    "distortion": {"param_range": (0.0, 15.0), "n_levels": 7},  # drive_db: 미세~중간 새추레이션 (헤비 퍼즈 급은 제외)
    "highshelf": {"param_range": (-9.0, 9.0), "n_levels": 7},  # gain_db: 일반적인 믹싱 EQ 부스트/컷 범위
}

CLAP_REPO_ID = "lukewys/laion_clap"
CLAP_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"

# NSynth 악기 패밀리 11종. 개별 악기(수백~천 종)는 소스 수 대비 클래스당 표본이
# 너무 적어(1차 실험: 47클래스/294샘플 ≈ 6개) 통제로 부적합 — 패밀리 단위로 바꾼다.
NSYNTH_FAMILIES = [
    "bass", "brass", "flute", "guitar", "keyboard", "mallet",
    "organ", "reed", "string", "synth_lead", "vocal",
]
NSYNTH_SOURCE_TYPES = {"acoustic", "electronic", "synthetic"}


def parse_nsynth_filename(path: Path):
    """예: bass_synthetic_033-045-100.wav -> ("bass_synthetic_033", 45, 100)"""
    parts = path.stem.rsplit("-", 2)
    if len(parts) != 3:
        return path.stem, None, None
    instrument, pitch_str, velocity_str = parts
    try:
        return instrument, int(pitch_str), int(velocity_str)
    except ValueError:
        return instrument, None, None


def parse_instrument_family(instrument: str) -> str:
    """예: "synth_lead_synthetic_006" -> "synth_lead" (패밀리명 자체에 밑줄이 있는 경우 대응).

    NSynth 악기 문자열은 "{family}_{source_type}_{id}" 형태다. source_type
    (acoustic/electronic/synthetic) 바로 앞까지가 패밀리명이므로, 알려진
    source_type 토큰을 뒤에서 찾아 그 앞부분을 패밀리로 취급한다.
    """
    tokens = instrument.split("_")
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] in NSYNTH_SOURCE_TYPES:
            family = "_".join(tokens[:i])
            return family if family else instrument
    return instrument


def load_and_preprocess(path: Path):
    """48kHz/모노/4초로 맞추고 피크 0.7로 정규화. 무음이면 None을 반환."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) < NUM_SAMPLES:
        y = np.pad(y, (0, NUM_SAMPLES - len(y)))
    else:
        y = y[:NUM_SAMPLES]

    peak = float(np.abs(y).max())
    if peak < SILENCE_PEAK_THRESHOLD:
        return None
    y = y * (PEAK_TARGET / peak)
    return y.astype(np.float32)


def apply_effect(y: np.ndarray, effect_name: str, value: float) -> np.ndarray:
    if effect_name == "reverb":
        board = Pedalboard([Reverb(room_size=value, wet_level=0.4, dry_level=0.6)])
    elif effect_name == "distortion":
        board = Pedalboard([Distortion(drive_db=value)])
    elif effect_name == "highshelf":
        board = Pedalboard([HighShelfFilter(cutoff_frequency_hz=4000.0, gain_db=value)])
    else:
        raise ValueError(f"unknown effect: {effect_name}")

    wet = board(y, SAMPLE_RATE)
    peak = float(np.abs(wet).max())
    if peak > 1.0:
        wet = wet * (0.99 / peak)
    return wet.astype(np.float32)


def build_conditions():
    """소스 1개당 적용할 (effect, level_idx, param_value) 조건 리스트. 'dry'가 맨 앞."""
    conditions = [("dry", -1, 0.0)]
    for effect_name, spec in EFFECT_SPECS.items():
        levels = np.linspace(spec["param_range"][0], spec["param_range"][1], spec["n_levels"])
        for level_idx, value in enumerate(levels):
            conditions.append((effect_name, level_idx, float(value)))
    return conditions


def download_clap_checkpoint(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / CLAP_FILENAME
    if not ckpt_path.exists():
        print(f"{CLAP_FILENAME}이 로컬에 없습니다. Hugging Face에서 다운로드합니다...")
        try:
            hf_hub_download(repo_id=CLAP_REPO_ID, filename=CLAP_FILENAME, local_dir=cache_dir)
        except Exception as e:
            print(
                f"체크포인트 다운로드 실패: {e}\n"
                f"수동으로 받아 {ckpt_path}에 두세요:\n"
                f"https://huggingface.co/{CLAP_REPO_ID}/resolve/main/{CLAP_FILENAME}",
                file=sys.stderr,
            )
            sys.exit(1)
    return ckpt_path


def load_clap(device: torch.device, cache_dir: Path):
    import laion_clap

    ckpt_path = download_clap_checkpoint(cache_dir)
    try:
        clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    except TypeError:
        # 구버전 laion_clap은 CLAP_Module 생성자가 device 인자를 받지 않는다.
        clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        clap = clap.to(device)
    clap.load_ckpt(str(ckpt_path), verbose=False)
    clap.eval()
    return clap


def embed_batch(clap, device, batch: list[np.ndarray]) -> np.ndarray:
    tensor = torch.tensor(np.stack(batch), dtype=torch.float32, device=device)
    with torch.no_grad():
        emb = clap.get_audio_embedding_from_data(tensor, use_tensor=True)
    return emb.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="이펙트 조건별 CLAP 임베딩 추출")
    parser.add_argument("--audio-dir", type=str, required=True, help="NSynth wav 폴더 (예: test split)")
    parser.add_argument("--n-sources", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    if args.device == "mps":
        import os

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        print("MPS 사용 중: 일부 연산은 PYTORCH_ENABLE_MPS_FALLBACK=1로 CPU에 폴백됩니다.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    audio_dir = Path(args.audio_dir)
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"오류: {audio_dir}에서 wav 파일을 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(wav_files)

    device = torch.device(args.device)
    ckpt_cache_dir = Path(__file__).parent / "ckpts"

    print("CLAP 모델 로딩 중...")
    clap = load_clap(device, ckpt_cache_dir)

    conditions = build_conditions()

    records: list[dict] = []
    batch_audio: list[np.ndarray] = []
    batch_meta: list[dict] = []
    embedding_chunks: list[np.ndarray] = []

    def flush():
        if not batch_audio:
            return
        emb = embed_batch(clap, device, batch_audio)
        embedding_chunks.append(emb)
        records.extend(batch_meta)
        batch_audio.clear()
        batch_meta.clear()

    n_collected = 0
    n_skipped_silent = 0

    pbar = tqdm(total=args.n_sources, desc="소스 처리")
    for wav_path in wav_files:
        if n_collected >= args.n_sources:
            break

        instrument, pitch, _velocity = parse_nsynth_filename(wav_path)
        family = parse_instrument_family(instrument)
        y = load_and_preprocess(wav_path)
        if y is None:
            n_skipped_silent += 1
            continue

        src_id = n_collected
        for effect_name, level_idx, value in conditions:
            wet = y if effect_name == "dry" else apply_effect(y, effect_name, value)
            batch_audio.append(wet)
            batch_meta.append(
                {
                    "src_id": src_id,
                    "filename": wav_path.name,
                    "instrument": instrument,
                    "instrument_family": family,
                    "pitch": pitch if pitch is not None else -1,
                    "effect": effect_name,
                    "level_idx": level_idx,
                    "param_value": value,
                }
            )
            if len(batch_audio) >= args.batch_size:
                flush()

        n_collected += 1
        pbar.update(1)
    pbar.close()
    flush()

    if n_collected < args.n_sources:
        print(
            f"경고: 요청한 {args.n_sources}개 중 {n_collected}개만 확보했습니다 "
            f"(무음 스킵 {n_skipped_silent}개, 파일 부족 가능성)."
        )

    embeddings = np.concatenate(embedding_chunks, axis=0)
    assert embeddings.shape[0] == len(records), "임베딩 개수와 메타데이터 개수가 일치하지 않습니다."

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "embeddings.npz",
        embeddings=embeddings,
        src_id=np.array([r["src_id"] for r in records], dtype=np.int64),
        filename=np.array([r["filename"] for r in records]),
        instrument=np.array([r["instrument"] for r in records]),
        instrument_family=np.array([r["instrument_family"] for r in records]),
        pitch=np.array([r["pitch"] for r in records], dtype=np.int64),
        effect=np.array([r["effect"] for r in records]),
        level_idx=np.array([r["level_idx"] for r in records], dtype=np.int64),
        param_value=np.array([r["param_value"] for r in records], dtype=np.float64),
    )

    config = {
        "audio_dir": str(audio_dir),
        "n_sources_requested": args.n_sources,
        "n_sources_collected": n_collected,
        "n_skipped_silent": n_skipped_silent,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
        "sample_rate": SAMPLE_RATE,
        "duration_sec": DURATION_SEC,
        "peak_target": PEAK_TARGET,
        "effect_specs": EFFECT_SPECS,
        "nsynth_families": NSYNTH_FAMILIES,
        "clap_repo_id": CLAP_REPO_ID,
        "clap_checkpoint": CLAP_FILENAME,
        "n_conditions_per_source": len(conditions),
        "n_embeddings": int(embeddings.shape[0]),
    }
    with open(out_dir / "embed_config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"완료: {embeddings.shape[0]}개 임베딩을 {out_dir / 'embeddings.npz'}에 저장했습니다.")


if __name__ == "__main__":
    main()
