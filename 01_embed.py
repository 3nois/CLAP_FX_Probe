"""CLAP FX Probe — 01_embed.py (3차 개정: 결합 샘플링)

2차까지는 "한 번에 하나씩 스윕(OAT)"이었다. reverb의 wet_level은 다른 모든 파라미터에
대한 곱셈 게이트다(wet_level=0이면 room_size/damping/width가 무엇이든 출력은 dry) —
그래서 OAT로 damping을 재려면 wet_level을 어딘가에 고정해야 하는데, 그 고정값에 따라
"damping 효과"가 완전히 달라진다. 격자 탐색(7^5=16,807 조합/소스)도 불가능하다.

이번 판은 파라미터 공간을 결합(joint) Latin Hypercube 샘플링하고, 미분 가능한 대리모델을
학습해 그 야코비안을 분석하는 방식으로 바꾼다 (야코비안 분석은 02_surrogate.py /
03_jacobian.py에서 진행). 이 스크립트는 그 입력이 되는 (e_dry, θ, e_wet) 데이터를 만든다.

오디오는 디스크에 쓰지 않고 메모리에서 처리한 뒤 임베딩만 저장한다.
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
from scipy.stats import qmc
from tqdm import tqdm

SAMPLE_RATE = 48000  # LAION-CLAP이 기대하는 샘플레이트
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
PEAK_TARGET = 0.7
SILENCE_PEAK_THRESHOLD = 1e-4

CLAP_REPO_ID = "lukewys/laion_clap"
CLAP_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"

# NSynth 악기 패밀리 11종. 개별 악기(수백~천 종)는 소스 수 대비 클래스당 표본이
# 너무 적어(1차 실험: 47클래스/294샘플 ≈ 6개) 통제로 부적합 — 패밀리 단위로 바꾼다.
NSYNTH_FAMILIES = [
    "bass", "brass", "flute", "guitar", "keyboard", "mallet",
    "organ", "reed", "string", "synth_lead", "vocal",
]
NSYNTH_SOURCE_TYPES = {"acoustic", "electronic", "synthetic"}

# ---------------------------------------------------------------------------
# 파라미터 공간 정의 — pedalboard 0.9.24 실측 시그니처와 대조 확인함
# (Reverb: room_size/damping/wet_level/dry_level/width/freeze_mode,
#  Distortion: drive_db, HighShelfFilter: cutoff_frequency_hz/gain_db/q)
#
# scale: "linear" | "log" (LHS를 log 공간에서 뽑은 뒤 지수화) | "bernoulli"
# is_negative_control: True면 오디오가 모노라 이 파라미터는 원리적으로 파형에
#   영향을 줄 수 없어야 한다 (width — 과제 4의 음성 통제).
# ---------------------------------------------------------------------------
PARAM_SPACE = {
    "reverb": {
        "wet_level": {"range": (0.0, 0.5), "scale": "linear"},
        "room_size": {"range": (0.0, 0.9), "scale": "linear"},
        "damping": {"range": (0.0, 1.0), "scale": "linear"},
        "width": {"range": (0.0, 1.0), "scale": "linear", "is_negative_control": True},
        "freeze_mode": {"range": (0.0, 1.0), "scale": "bernoulli"},
    },
    "distortion": {
        "drive_db": {"range": (0.0, 15.0), "scale": "linear"},
    },
    "highshelf": {
        "gain_db": {"range": (-9.0, 9.0), "scale": "linear", "signed": True},
        "cutoff_frequency_hz": {"range": (500.0, 8000.0), "scale": "log"},
        "q": {"range": (0.3, 3.0), "scale": "log"},
    },
}
EFFECTS = ["reverb", "distortion", "highshelf"]
EFFECT_SEED_ID = {"reverb": 1, "distortion": 2, "highshelf": 3}  # 고정 정수 — 파이썬 hash()는
# 프로세스마다 값이 달라져(해시 랜덤화) 시드 재현성이 깨지므로 절대 쓰지 말 것.
PARAM_ORDER = {e: list(PARAM_SPACE[e].keys()) for e in EFFECTS}
N_SAMPLES_PER_SOURCE = {"reverb": 32, "distortion": 8, "highshelf": 16}

# 9차원 패딩 θ 벡터에서 각 이펙트가 차지하는 슬롯 (효과가 아닌 파라미터 자리는 0으로 채움).
THETA_WIDTH = sum(len(PARAM_ORDER[e]) for e in EFFECTS)
_offset = 0
THETA_SLOTS = {}
for _e in EFFECTS:
    _n = len(PARAM_ORDER[_e])
    THETA_SLOTS[_e] = (_offset, _offset + _n)
    _offset += _n

NEGATIVE_CONTROL_PARAMS = [
    f"{e}.{p}" for e in EFFECTS for p, spec in PARAM_SPACE[e].items() if spec.get("is_negative_control")
]

# "무효과" 앵커 θ — 이 값일 때는 pedalboard를 아예 통과시키지 않고 dry 오디오를 그대로 쓴다.
ANCHOR_THETA = {
    "reverb": {"wet_level": 0.0, "room_size": 0.0, "damping": 0.0, "width": 0.0, "freeze_mode": 0.0},
    "distortion": {"drive_db": 0.0},
    "highshelf": {"gain_db": 0.0, "cutoff_frequency_hz": 4000.0, "q": 0.7071067811865476},
}

# 텍스트-오디오 정렬 검증(과제 5)용 캡션 템플릿. {} 에 악기명이 들어간다.
TEXT_TEMPLATES = {
    "distortion": [("distorted {}", "{}"), ("overdriven {}", "{}")],
    "reverb": [("{} in a large hall", "{}"), ("reverberant {}", "{}")],
    # highshelf에 대응하는 정확한 관용 표현이 없어 "밝다/선명하다" 계열로 근사한다.
    # 이 근사가 불완전하다는 점을 README 해석 가이드에 명시할 것.
    "highshelf": [("bright {}", "{}"), ("crisp {}", "{}")],
}
# 무작위 텍스트 방향 통제 — 오디오 이펙트와 무관한 수식어
RANDOM_TEXT_TEMPLATES = [("wooden {}", "{}"), ("square {}", "{}"), ("left-handed {}", "{}")]
TEXT_FAMILIES = [f for f in NSYNTH_FAMILIES if f != "synth_lead"]  # NSynth test split에 없음


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


def apply_effect(y: np.ndarray, effect_name: str, theta: dict) -> np.ndarray:
    if effect_name == "reverb":
        # dry_level=1.0으로 고정 — wet_level이 "얼마나 더할지"만 통제하는 순수 게이트가
        # 되도록 한다. 그렇지 않으면 dry_level이 또 다른 숨은 스케일 요인이 된다.
        board = Pedalboard(
            [
                Reverb(
                    room_size=theta["room_size"],
                    damping=theta["damping"],
                    wet_level=theta["wet_level"],
                    dry_level=1.0,
                    width=theta["width"],
                    freeze_mode=theta["freeze_mode"],
                )
            ]
        )
    elif effect_name == "distortion":
        board = Pedalboard([Distortion(drive_db=theta["drive_db"])])
    elif effect_name == "highshelf":
        board = Pedalboard(
            [
                HighShelfFilter(
                    cutoff_frequency_hz=theta["cutoff_frequency_hz"],
                    gain_db=theta["gain_db"],
                    q=theta["q"],
                )
            ]
        )
    else:
        raise ValueError(f"unknown effect: {effect_name}")

    wet = board(y, SAMPLE_RATE)
    peak = float(np.abs(wet).max())
    if peak > 1.0:
        wet = wet * (0.99 / peak)
    return wet.astype(np.float32)


def _lhs_unit_samples(n: int, dims: int, seed_seq: np.random.SeedSequence) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, dims))
    sampler = qmc.LatinHypercube(d=dims, seed=np.random.default_rng(seed_seq))
    return sampler.random(n=n)


def sample_theta_for_effect(effect_name: str, n_random: int, rng: np.random.Generator, seed_seq) -> list:
    """이 이펙트의 θ를 n_random개 결합 LHS로 뽑는다. 반환은 raw 값 dict의 리스트."""
    params = PARAM_ORDER[effect_name]
    continuous = [p for p in params if PARAM_SPACE[effect_name][p]["scale"] != "bernoulli"]
    bernoulli_params = [p for p in params if PARAM_SPACE[effect_name][p]["scale"] == "bernoulli"]

    unit = _lhs_unit_samples(n_random, len(continuous), seed_seq)
    samples = []
    for i in range(n_random):
        theta = {}
        for j, p in enumerate(continuous):
            spec = PARAM_SPACE[effect_name][p]
            lo, hi = spec["range"]
            u = unit[i, j]
            if spec["scale"] == "log":
                theta[p] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
            else:
                theta[p] = float(lo + u * (hi - lo))
        for p in bernoulli_params:
            theta[p] = float(rng.integers(0, 2))  # Bernoulli(0.5), 다른 파라미터와 별도 추출
        samples.append(theta)
    return samples


def normalize_theta(effect_name: str, theta: dict) -> dict:
    """θ를 [0,1]로 정규화 (log 파라미터는 log 공간에서, bernoulli는 그대로)."""
    norm = {}
    for p, v in theta.items():
        spec = PARAM_SPACE[effect_name][p]
        if spec["scale"] == "bernoulli":
            norm[p] = float(v)
            continue
        lo, hi = spec["range"]
        if spec["scale"] == "log":
            norm[p] = float((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)))
        else:
            norm[p] = float((v - lo) / (hi - lo))
    return norm


def theta_dict_to_padded(effect_name: str, theta: dict) -> np.ndarray:
    """9차원 패딩 벡터로 변환 (해당 이펙트 슬롯만 채움, 나머지는 0)."""
    vec = np.zeros(THETA_WIDTH, dtype=np.float64)
    start, _ = THETA_SLOTS[effect_name]
    for i, p in enumerate(PARAM_ORDER[effect_name]):
        vec[start + i] = theta[p]
    return vec


def build_conditions_for_source(src_id: int, base_seed: int):
    """소스 1개당 조건 리스트: [("dry", {}, theta_raw, theta_norm, is_anchor), (effect, ...), ...]."""
    rng = np.random.default_rng([base_seed, src_id])
    conditions = [("dry", {}, np.zeros(THETA_WIDTH), np.zeros(THETA_WIDTH), False)]

    for effect_name in EFFECTS:
        n_total = N_SAMPLES_PER_SOURCE[effect_name]
        n_random = n_total - 1  # 하나는 θ=0 앵커로 예약
        seed_seq = np.random.SeedSequence([base_seed, src_id, EFFECT_SEED_ID[effect_name]])
        random_thetas = sample_theta_for_effect(effect_name, n_random, rng, seed_seq)

        for theta_raw in random_thetas:
            theta_norm = normalize_theta(effect_name, theta_raw)
            conditions.append(
                (effect_name, theta_raw, theta_dict_to_padded(effect_name, theta_raw), theta_dict_to_padded(effect_name, theta_norm), False)
            )

        anchor_raw = ANCHOR_THETA[effect_name]
        anchor_norm = normalize_theta(effect_name, anchor_raw)
        conditions.append(
            (effect_name, anchor_raw, theta_dict_to_padded(effect_name, anchor_raw), theta_dict_to_padded(effect_name, anchor_norm), True)
        )

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


def extract_text_embeddings(clap, out_dir: Path):
    """과제 5용 텍스트 임베딩 — 이펙트별 캡션 쌍 + 무작위 통제. CLAP이 이미 로딩된 김에 여기서 처리."""
    print("텍스트 임베딩 추출 중 (텍스트-오디오 정렬 검증용)...")
    rows = []  # (group, template_pos, template_neg, family, text, role)

    def add_pair(group, tmpl_pos, tmpl_neg, family):
        rows.append({"group": group, "family": family, "text_pos": tmpl_pos.format(family), "text_neg": tmpl_neg.format(family)})

    for effect_name, templates in TEXT_TEMPLATES.items():
        for tmpl_pos, tmpl_neg in templates:
            for family in TEXT_FAMILIES:
                add_pair(effect_name, tmpl_pos, tmpl_neg, family)

    for tmpl_pos, tmpl_neg in RANDOM_TEXT_TEMPLATES:
        for family in TEXT_FAMILIES:
            add_pair("random_control", tmpl_pos, tmpl_neg, family)

    all_texts = sorted({r["text_pos"] for r in rows} | {r["text_neg"] for r in rows})
    text_to_idx = {t: i for i, t in enumerate(all_texts)}

    embeddings = []
    with torch.no_grad():
        for text in tqdm(all_texts, desc="텍스트 임베딩"):
            emb = clap.get_text_embedding([text], use_tensor=True)
            embeddings.append(emb.cpu().numpy()[0])
    embeddings = np.stack(embeddings)

    np.savez(
        out_dir / "text_embeddings.npz",
        embeddings=embeddings,
        texts=np.array(all_texts),
        pair_group=np.array([r["group"] for r in rows]),
        pair_family=np.array([r["family"] for r in rows]),
        pair_pos_idx=np.array([text_to_idx[r["text_pos"]] for r in rows], dtype=np.int64),
        pair_neg_idx=np.array([text_to_idx[r["text_neg"]] for r in rows], dtype=np.int64),
    )
    print(f"완료: 텍스트 {len(all_texts)}개, 쌍 {len(rows)}개 -> {out_dir / 'text_embeddings.npz'}")


def main():
    parser = argparse.ArgumentParser(description="결합 LHS 샘플링으로 이펙트 조건별 CLAP 임베딩 추출 (3차)")
    parser.add_argument("--audio-dir", type=str, required=True, help="NSynth wav 폴더 (예: test split)")
    parser.add_argument("--n-sources", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--skip-text", action="store_true", help="텍스트 임베딩 추출 생략")
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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_text:
        extract_text_embeddings(clap, out_dir)

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
        conditions = build_conditions_for_source(src_id, args.seed)

        for effect_name, theta_raw, theta_raw_vec, theta_norm_vec, is_anchor in conditions:
            if effect_name == "dry" or is_anchor:
                # θ=0 앵커는 이펙트 체인을 통과시키지 않고 dry 오디오를 그대로 쓴다 —
                # 이러면 neutral_check의 cos(e_dry, e_theta0)가 정의상 1.000이 된다.
                wet = y
            else:
                wet = apply_effect(y, effect_name, theta_raw)
            batch_audio.append(wet)
            batch_meta.append(
                {
                    "src_id": src_id,
                    "filename": wav_path.name,
                    "instrument": instrument,
                    "instrument_family": family,
                    "pitch": pitch if pitch is not None else -1,
                    "effect": effect_name,
                    "is_anchor": bool(is_anchor),
                    "theta_raw": theta_raw_vec,
                    "theta_norm": theta_norm_vec,
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

    np.savez(
        out_dir / "embeddings.npz",
        embeddings=embeddings,
        src_id=np.array([r["src_id"] for r in records], dtype=np.int64),
        filename=np.array([r["filename"] for r in records]),
        instrument=np.array([r["instrument"] for r in records]),
        instrument_family=np.array([r["instrument_family"] for r in records]),
        pitch=np.array([r["pitch"] for r in records], dtype=np.int64),
        effect=np.array([r["effect"] for r in records]),
        is_anchor=np.array([r["is_anchor"] for r in records], dtype=bool),
        theta_raw=np.stack([r["theta_raw"] for r in records]),
        theta_norm=np.stack([r["theta_norm"] for r in records]),
    )

    config = {
        "experiment_version": 3,
        "sampling": "latin_hypercube",
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
        "param_space": {
            e: {p: (spec["range"] if spec["scale"] != "bernoulli" else "bernoulli_0.5") for p, spec in PARAM_SPACE[e].items()}
            for e in EFFECTS
        },
        "param_order": PARAM_ORDER,
        "theta_slots": THETA_SLOTS,
        "theta_width": THETA_WIDTH,
        "negative_control_params": NEGATIVE_CONTROL_PARAMS,
        "n_samples_per_source": N_SAMPLES_PER_SOURCE,
        "nsynth_families": NSYNTH_FAMILIES,
        "clap_repo_id": CLAP_REPO_ID,
        "clap_checkpoint": CLAP_FILENAME,
        "n_conditions_per_source": 1 + sum(N_SAMPLES_PER_SOURCE.values()),
        "n_embeddings": int(embeddings.shape[0]),
    }
    with open(out_dir / "embed_config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"완료: {embeddings.shape[0]}개 임베딩을 {out_dir / 'embeddings.npz'}에 저장했습니다.")


if __name__ == "__main__":
    main()
