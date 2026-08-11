"""CLAP FX Probe — 13_fd_phase3_render.py (6차 후속 과제 2: 조건 C 렌더링 + FD 캐시)

과제 1(12_freeze_probe.py)에서 확정된 두 가지 문제를 모두 피하는 새 조건 C를 도입한다.

  조건 A(1~5차): dry 피크 0.7 정규화 → 이펙트 → wet 피크>1.0이면 0.99로 재정규화.
                 이 재정규화가 wet_level 등과 상관되는 confound다(과제 1 근거 참고).
  조건 B(5차 시도, 폐기): 하드클립. 클리핑 비선형이라는 새 교란(음성 통제 width가
                 62% 증가)이 생겨서 이번 라운드에서는 쓰지 않는다.
  조건 C(신규):  dry 피크 0.3 정규화(헤드룸 확보) → 이펙트 → 추가 정규화 없음.
                 정규화 상쇄와 클리핑 둘 다 피한다. 클리핑 발생 여부만 기록한다.

100소스(패밀리 균형 10x10) x 소스당 결합 LHS θ 5개 = 500 평가점. freeze_mode=0 고정
(Bernoulli 추출 안 함 — 과제 1에서 freeze_mode=1이 reverb 4파라미터를 완전 무효화함이
확인됨). 스윕 축 10개(조건 C 전체) + reverb 4개(조건 A, 민감도 비교용)를 h=0.02 중앙차분
(경계는 편측)으로 미분한다. h=0.02 단일값만 쓴다 — 5차 Phase 1에서 h 민감도 코사인
중앙값 ≥0.985로 이미 확인됨.

★ 점별 512차원 J_fd 벡터를 반드시 캐시한다 (out/phase3_fd_cache.npz). Phase 1은 집계
통계만 저장하고 점별 벡터를 버려서 재렌더링 없이는 후속 분석(과제 5~8)이 불가능했다 —
그 실수를 반복하지 않는다. 이후 모든 분석은 이 캐시만 읽는다.

결과 해석은 이 스크립트가 단정하지 않는다. README 6차 후속 절의 판정 기준표를 따를 것.
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
from scipy.stats import qmc
from tqdm import tqdm

SAMPLE_RATE = 48000
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
SILENCE_PEAK_THRESHOLD = 1e-4

PEAK_TARGET_A = 0.7  # 조건 A (1~5차와 동일)
DEFAULT_PEAK_TARGET_C = 0.3  # 조건 C 헤드룸

CLAP_REPO_ID = "lukewys/laion_clap"
CLAP_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"

ULTRASONIC_12K_HZ = 12000.0
ULTRASONIC_15K_HZ = 15000.0
ULTRASONIC_Q = 0.7071067811865476  # butterworth Q, phase1과 동일 관례

H = 0.02

NSYNTH_SOURCE_TYPES = {"acoustic", "electronic", "synthetic"}

# ---------------------------------------------------------------------------
# 파라미터 공간 — 이번 라운드 전용 (surrogate 좌표계와 무관, 대리모델을 안 씀)
# ---------------------------------------------------------------------------
AXIS_SPEC = {
    ("reverb", "wet_level"): {"range": (0.0, 0.5), "scale": "linear"},
    ("reverb", "room_size"): {"range": (0.0, 0.9), "scale": "linear"},
    ("reverb", "damping"): {"range": (0.0, 1.0), "scale": "linear"},
    ("reverb", "width"): {"range": (0.0, 1.0), "scale": "linear"},
    ("distortion", "drive_db"): {"range": (0.0, 15.0), "scale": "linear"},
    ("highshelf", "gain_db"): {"range": (-9.0, 9.0), "scale": "linear"},
    ("highshelf", "cutoff_frequency_hz"): {"range": (500.0, 4000.0), "scale": "log"},
    ("highshelf", "q"): {"range": (0.3, 3.0), "scale": "log"},
    ("highshelf", "ultrasonic_12k_gain_db"): {"range": (-9.0, 9.0), "scale": "linear"},
    ("highshelf", "ultrasonic_15k_gain_db"): {"range": (-9.0, 9.0), "scale": "linear"},
}
AXES_C = list(AXIS_SPEC.keys())  # 10축, 고정 순서
AXES_A = [("reverb", "wet_level"), ("reverb", "room_size"), ("reverb", "damping"), ("reverb", "width")]
GROUP_PARAMS = {
    "reverb": ["wet_level", "room_size", "damping", "width"],
    "distortion": ["drive_db"],
    "highshelf": ["gain_db", "cutoff_frequency_hz", "q", "ultrasonic_12k_gain_db", "ultrasonic_15k_gain_db"],
}
GROUPS = ["reverb", "distortion", "highshelf"]


def to_raw(lo, hi, scale, u):
    u = float(np.clip(u, 0.0, 1.0))
    if scale == "log":
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    return float(lo + u * (hi - lo))


# ---------------------------------------------------------------------------
# 렌더링
# ---------------------------------------------------------------------------
def render_reverb(y, theta_raw):
    board = Pedalboard([Reverb(
        room_size=theta_raw["room_size"], damping=theta_raw["damping"],
        wet_level=theta_raw["wet_level"], dry_level=1.0, width=theta_raw["width"],
        freeze_mode=0.0,
    )])
    return board(y, SAMPLE_RATE)


def render_distortion(y, theta_raw):
    board = Pedalboard([Distortion(drive_db=theta_raw["drive_db"])])
    return board(y, SAMPLE_RATE)


def render_highshelf(y, theta_raw):
    board = Pedalboard([
        HighShelfFilter(cutoff_frequency_hz=theta_raw["cutoff_frequency_hz"],
                         gain_db=theta_raw["gain_db"], q=theta_raw["q"]),
        HighShelfFilter(cutoff_frequency_hz=ULTRASONIC_12K_HZ,
                         gain_db=theta_raw["ultrasonic_12k_gain_db"], q=ULTRASONIC_Q),
        HighShelfFilter(cutoff_frequency_hz=ULTRASONIC_15K_HZ,
                         gain_db=theta_raw["ultrasonic_15k_gain_db"], q=ULTRASONIC_Q),
    ])
    return board(y, SAMPLE_RATE)


RENDER_FN = {"reverb": render_reverb, "distortion": render_distortion, "highshelf": render_highshelf}


def load_raw(path: Path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) < NUM_SAMPLES:
        y = np.pad(y, (0, NUM_SAMPLES - len(y)))
    else:
        y = y[:NUM_SAMPLES]
    peak = float(np.abs(y).max())
    if peak < SILENCE_PEAK_THRESHOLD:
        return None
    return y.astype(np.float32), peak


# ---------------------------------------------------------------------------
# CLAP
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 소스 선정 — 패밀리 균형 10 x 10
# ---------------------------------------------------------------------------
def select_sources(family_by_src: dict, n_per_family: int, seed: int):
    rng = np.random.RandomState(seed)
    by_family = collections.defaultdict(list)
    for s, f in family_by_src.items():
        by_family[f].append(s)
    families = sorted(by_family.keys())
    selected = []
    family_of_selected = []
    for fam in families:
        pool = sorted(by_family[fam])
        if len(pool) < n_per_family:
            raise RuntimeError(f"패밀리 {fam}에 소스가 {len(pool)}개뿐입니다 (요청 {n_per_family})")
        idx = rng.choice(len(pool), size=n_per_family, replace=False)
        for i in sorted(idx.tolist()):
            selected.append(pool[i])
            family_of_selected.append(fam)
    return selected, family_of_selected, families


# ---------------------------------------------------------------------------
# θ 샘플링 — 소스당 결합 LHS 5개, 10차원
# ---------------------------------------------------------------------------
def sample_theta_for_source(src_id: int, n_theta: int, seed: int):
    sampler = qmc.LatinHypercube(d=len(AXES_C), seed=np.random.default_rng([seed, int(src_id)]))
    unit = sampler.random(n=n_theta)  # (n_theta, 10)
    return unit


# ---------------------------------------------------------------------------
# FD job 계획
# ---------------------------------------------------------------------------
def plan_jobs_C(theta_norm_i: dict, theta_raw_i: dict):
    jobs, plan = [], {}
    for group in GROUPS:
        params = GROUP_PARAMS[group]
        center_raw = {p: theta_raw_i[(group, p)] for p in params}
        jobs.append((group, "center", dict(center_raw)))
        for (effect, param) in AXES_C:
            if effect != group:
                continue
            lo, hi, scale = AXIS_SPEC[(effect, param)]["range"][0], AXIS_SPEC[(effect, param)]["range"][1], AXIS_SPEC[(effect, param)]["scale"]
            u0 = theta_norm_i[(effect, param)]
            can_plus, can_minus = (u0 + H) <= 1.0, (u0 - H) >= 0.0
            mode = "central" if (can_plus and can_minus) else ("fwd" if can_plus else "bwd")
            plan[(effect, param)] = mode
            if mode in ("central", "fwd"):
                theta = dict(center_raw); theta[param] = to_raw(lo, hi, scale, u0 + H)
                jobs.append((group, f"{param}+", theta))
            if mode in ("central", "bwd"):
                theta = dict(center_raw); theta[param] = to_raw(lo, hi, scale, u0 - H)
                jobs.append((group, f"{param}-", theta))
    return jobs, plan


def plan_jobs_A(theta_norm_i: dict, theta_raw_i: dict):
    jobs, plan = [], {}
    group = "reverb"
    params = GROUP_PARAMS[group]
    center_raw = {p: theta_raw_i[(group, p)] for p in params}
    jobs.append((group, "center", dict(center_raw)))
    for (effect, param) in AXES_A:
        lo, hi, scale = AXIS_SPEC[(effect, param)]["range"][0], AXIS_SPEC[(effect, param)]["range"][1], AXIS_SPEC[(effect, param)]["scale"]
        u0 = theta_norm_i[(effect, param)]
        can_plus, can_minus = (u0 + H) <= 1.0, (u0 - H) >= 0.0
        mode = "central" if (can_plus and can_minus) else ("fwd" if can_plus else "bwd")
        plan[(effect, param)] = mode
        if mode in ("central", "fwd"):
            theta = dict(center_raw); theta[param] = to_raw(lo, hi, scale, u0 + H)
            jobs.append((group, f"{param}+", theta))
        if mode in ("central", "bwd"):
            theta = dict(center_raw); theta[param] = to_raw(lo, hi, scale, u0 - H)
            jobs.append((group, f"{param}-", theta))
    return jobs, plan


def main():
    parser = argparse.ArgumentParser(description="6차 후속 과제 2 — 조건 C 렌더링 + FD 캐시")
    parser.add_argument("--audio-dir", type=str, default="nsynth-test/audio")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--n-sources-per-family", type=int, default=10)
    parser.add_argument("--n-theta-per-source", type=int, default=5)
    parser.add_argument("--peak-target-c", type=float, default=DEFAULT_PEAK_TARGET_C)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.out)
    audio_dir = Path(args.audio_dir)

    print("임베딩 메타데이터 로딩 중 (소스/패밀리 목록용)...")
    d = np.load(args.embeddings, allow_pickle=False)
    dry_mask = d["effect"] == "dry"
    dry_src_ids = d["src_id"][dry_mask]
    family_by_src = dict(zip(dry_src_ids.tolist(), d["instrument_family"][dry_mask].tolist()))
    filename_by_src = dict(zip(dry_src_ids.tolist(), d["filename"][dry_mask].tolist()))

    selected_srcs, family_of_selected, families = select_sources(family_by_src, args.n_sources_per_family, args.seed)
    n_sources = len(selected_srcs)
    print(f"패밀리 {len(families)}개 x 소스 {args.n_sources_per_family}개 = {n_sources}개 소스 선정: {families}")

    n_theta = args.n_theta_per_source
    n_points = n_sources * n_theta
    print(f"평가점 {n_points}개 (소스 {n_sources} x θ {n_theta})")

    print("소스 오디오 로딩 중...")
    y_raw_by_src, peak_by_src = {}, {}
    for s in tqdm(selected_srcs, desc="오디오 로딩"):
        r = load_raw(audio_dir / filename_by_src[s])
        if r is None:
            raise RuntimeError(f"src_id={s} ({filename_by_src[s]})가 무음입니다.")
        y_raw_by_src[s], peak_by_src[s] = r

    y_dry_C_by_src = {s: (y_raw_by_src[s] * (args.peak_target_c / peak_by_src[s])).astype(np.float32) for s in selected_srcs}
    y_dry_A_by_src = {s: (y_raw_by_src[s] * (PEAK_TARGET_A / peak_by_src[s])).astype(np.float32) for s in selected_srcs}

    print("평가점 θ 샘플링 중 (소스당 결합 LHS 10차원)...")
    point_src, point_family = [], []
    theta_norm_pts, theta_raw_pts = [], []  # list of dict keyed (effect,param)
    for si, s in enumerate(selected_srcs):
        unit = sample_theta_for_source(s, n_theta, args.seed)  # (n_theta, 10)
        for ti in range(n_theta):
            u_dict = {ax: float(unit[ti, k]) for k, ax in enumerate(AXES_C)}
            raw_dict = {ax: to_raw(AXIS_SPEC[ax]["range"][0], AXIS_SPEC[ax]["range"][1], AXIS_SPEC[ax]["scale"], u_dict[ax]) for ax in AXES_C}
            theta_norm_pts.append(u_dict)
            theta_raw_pts.append(raw_dict)
            point_src.append(s)
            point_family.append(family_of_selected[si])

    assert len(theta_norm_pts) == n_points

    print("job 계획 수립 중...")
    all_plans_C, all_plans_A = [], []
    all_jobs = []  # (point_idx, cond, group, tag, theta_raw)
    for i in range(n_points):
        jobs_c, plan_c = plan_jobs_C(theta_norm_pts[i], theta_raw_pts[i])
        jobs_a, plan_a = plan_jobs_A(theta_norm_pts[i], theta_raw_pts[i])
        all_plans_C.append(plan_c)
        all_plans_A.append(plan_a)
        for group, tag, theta_raw in jobs_c:
            all_jobs.append((i, "C", group, tag, theta_raw))
        for group, tag, theta_raw in jobs_a:
            all_jobs.append((i, "A", group, tag, theta_raw))

    print(f"렌더링 작업 {len(all_jobs)}개 (평가점 {n_points}개, h={H})")

    device = torch.device(args.device)
    if args.device == "mps":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print("CLAP 모델 로딩 중...")
    clap = load_clap(device, Path(__file__).parent / "ckpts")

    embeddings_by_job = {}
    peak_record = {}  # (point_idx, cond, group) -> peak of center render
    batch_audio, batch_keys = [], []

    def flush():
        if not batch_audio:
            return
        emb = embed_batch(clap, device, batch_audio)
        for k, e in zip(batch_keys, emb):
            embeddings_by_job[k] = e
        batch_audio.clear()
        batch_keys.clear()

    for pi, cond, group, tag, theta_raw in tqdm(all_jobs, desc="렌더링+임베딩"):
        y_dry = y_dry_C_by_src[point_src[pi]] if cond == "C" else y_dry_A_by_src[point_src[pi]]
        wet = RENDER_FN[group](y_dry, theta_raw)
        peak = float(np.abs(wet).max())
        if tag == "center":
            peak_record[(pi, cond, group)] = peak
        if cond == "A" and peak > 1.0:
            wet = wet * (0.99 / peak)
        batch_audio.append(wet.astype(np.float32))
        batch_keys.append((pi, cond, group, tag))
        if len(batch_audio) >= args.batch_size:
            flush()
    flush()

    print("FD 미분 계산 중...")
    jac_C = np.zeros((n_points, len(AXES_C), 512), dtype=np.float32)
    onesided_mask_C = np.zeros((n_points, len(AXES_C)), dtype=bool)
    for i in range(n_points):
        plan = all_plans_C[i]
        for k, (effect, param) in enumerate(AXES_C):
            mode = plan[(effect, param)]
            group = effect
            if mode == "central":
                ep = embeddings_by_job[(i, "C", group, f"{param}+")]
                em = embeddings_by_job[(i, "C", group, f"{param}-")]
                deriv = (ep - em) / (2 * H)
            elif mode == "fwd":
                ep = embeddings_by_job[(i, "C", group, f"{param}+")]
                ec = embeddings_by_job[(i, "C", group, "center")]
                deriv = (ep - ec) / H
            else:
                ec = embeddings_by_job[(i, "C", group, "center")]
                em = embeddings_by_job[(i, "C", group, f"{param}-")]
                deriv = (ec - em) / H
            jac_C[i, k] = deriv
            onesided_mask_C[i, k] = mode != "central"

    jac_A = np.zeros((n_points, len(AXES_A), 512), dtype=np.float32)
    onesided_mask_A = np.zeros((n_points, len(AXES_A)), dtype=bool)
    for i in range(n_points):
        plan = all_plans_A[i]
        for k, (effect, param) in enumerate(AXES_A):
            mode = plan[(effect, param)]
            group = "reverb"
            if mode == "central":
                ep = embeddings_by_job[(i, "A", group, f"{param}+")]
                em = embeddings_by_job[(i, "A", group, f"{param}-")]
                deriv = (ep - em) / (2 * H)
            elif mode == "fwd":
                ep = embeddings_by_job[(i, "A", group, f"{param}+")]
                ec = embeddings_by_job[(i, "A", group, "center")]
                deriv = (ep - ec) / H
            else:
                ec = embeddings_by_job[(i, "A", group, "center")]
                em = embeddings_by_job[(i, "A", group, f"{param}-")]
                deriv = (ec - em) / H
            jac_A[i, k] = deriv
            onesided_mask_A[i, k] = mode != "central"

    peak_reverb = np.array([peak_record[(i, "C", "reverb")] for i in range(n_points)], dtype=np.float32)
    peak_distortion = np.array([peak_record[(i, "C", "distortion")] for i in range(n_points)], dtype=np.float32)
    peak_highshelf = np.array([peak_record[(i, "C", "highshelf")] for i in range(n_points)], dtype=np.float32)
    peak_after_effect_C = np.maximum(np.maximum(peak_reverb, peak_distortion), peak_highshelf)
    clipping_occurred_C = peak_after_effect_C > 1.0
    clip_rate = float(clipping_occurred_C.mean())

    theta_arr = np.array([[theta_raw_pts[i][ax] for ax in AXES_C] for i in range(n_points)], dtype=np.float64)
    theta_norm_arr = np.array([[theta_norm_pts[i][ax] for ax in AXES_C] for i in range(n_points)], dtype=np.float64)
    src_id_arr = np.array(point_src, dtype=np.int64)
    family_arr = np.array(point_family)
    filename_arr = np.array([filename_by_src[s] for s in point_src])
    theta_group_id_arr = src_id_arr.copy()  # 같은 소스 = 같은 그룹

    axis_names_C = np.array([f"{e}.{p}" for e, p in AXES_C])
    axis_names_A = np.array([f"{e}.{p}" for e, p in AXES_A])

    onesided_ratio_C = {f"{e}.{p}": float(onesided_mask_C[:, k].mean()) for k, (e, p) in enumerate(AXES_C)}
    onesided_ratio_A = {f"{e}.{p}": float(onesided_mask_A[:, k].mean()) for k, (e, p) in enumerate(AXES_A)}

    cache_path = out_dir / "phase3_fd_cache.npz"
    np.savez(
        cache_path,
        jac_C=jac_C, jac_C_axis_names=axis_names_C,
        jac_A=jac_A, jac_A_axis_names=axis_names_A,
        theta=theta_arr, theta_norm=theta_norm_arr, theta_axis_names=axis_names_C,
        src_id=src_id_arr, instrument_family=family_arr, filename=filename_arr,
        theta_group_id=theta_group_id_arr,
        clipping_occurred_C=clipping_occurred_C,
        peak_after_effect_C=peak_after_effect_C,
        peak_after_effect_C_reverb=peak_reverb,
        peak_after_effect_C_distortion=peak_distortion,
        peak_after_effect_C_highshelf=peak_highshelf,
        h_used=np.float64(H),
        onesided_mask_C=onesided_mask_C, onesided_mask_A=onesided_mask_A,
        peak_target_c=np.float64(args.peak_target_c), peak_target_a=np.float64(PEAK_TARGET_A),
    )
    print(f"캐시 저장: {cache_path}")

    elapsed = time.time() - t_start
    family_dist = collections.Counter(point_family)

    results6_path = out_dir / "results_6.json"
    results6 = {}
    if results6_path.exists():
        with open(results6_path) as f:
            results6 = json.load(f)
    results6.setdefault("meta", {})
    results6["meta"].update({
        "task2_seed": args.seed, "task2_n_sources": n_sources, "task2_n_theta_per_source": n_theta,
        "task2_n_points": n_points, "task2_peak_target_c": args.peak_target_c, "task2_peak_target_a": PEAK_TARGET_A,
        "task2_h_used": H, "task2_n_render_jobs": len(all_jobs), "task2_elapsed_sec": elapsed,
        "task2_family_point_distribution": dict(family_dist),
        "task2_families": families,
        "task2_selected_src_ids": [int(s) for s in selected_srcs],
        "task2_axes_C": [f"{e}.{p}" for e, p in AXES_C],
        "task2_axes_A": [f"{e}.{p}" for e, p in AXES_A],
    })
    results6["task2_render"] = {
        "cache_path": str(cache_path),
        "clipping_rate_C": clip_rate,
        "clipping_headroom_warning": clip_rate > 0.01,
        "peak_after_effect_C_stats": {
            "reverb": {"mean": float(peak_reverb.mean()), "max": float(peak_reverb.max())},
            "distortion": {"mean": float(peak_distortion.mean()), "max": float(peak_distortion.max())},
            "highshelf": {"mean": float(peak_highshelf.mean()), "max": float(peak_highshelf.max())},
            "overall_max": float(peak_after_effect_C.max()),
        },
        "onesided_ratio_C": onesided_ratio_C,
        "onesided_ratio_A": onesided_ratio_A,
    }
    with open(results6_path, "w") as f:
        json.dump(results6, f, indent=2, ensure_ascii=False)

    print("\n=== 과제 2 결과 요약 ===")
    print(f"평가점 {n_points}개, 렌더링 작업 {len(all_jobs)}개, 소요 {elapsed/60:.1f}분")
    print(f"조건 C 클리핑 발생률: {clip_rate*100:.2f}% (헤드룸 {args.peak_target_c})")
    if clip_rate > 0.01:
        print("★ 클리핑 발생률이 1%를 초과했습니다 — 헤드룸을 0.2로 낮춰 재실행을 검토하세요.")
    print(f"peak_after_effect_C: reverb mean={peak_reverb.mean():.3f} max={peak_reverb.max():.3f} | "
          f"distortion mean={peak_distortion.mean():.3f} max={peak_distortion.max():.3f} | "
          f"highshelf mean={peak_highshelf.mean():.3f} max={peak_highshelf.max():.3f}")
    print(f"편측차분 비율(C): {onesided_ratio_C}")
    print(f"편측차분 비율(A): {onesided_ratio_A}")
    print(f"\n저장: {cache_path}, {results6_path}")


if __name__ == "__main__":
    main()
