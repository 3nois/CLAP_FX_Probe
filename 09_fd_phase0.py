"""CLAP FX Probe — 09_fd_phase0.py (5차 Phase 0: 유한차분 자체를 검증하는 널 축 테스트)

5차의 목적은 대리모델(02/03)의 autodiff 야코비안을 유한차분(J_fd)이라는 "정답"과
대조해 신뢰할 수 있는지 가르는 것이다. 그 정답 자체가 맞는지부터 확인해야 한다.

highshelf에 더미 파라미터 ultrasonic_gain_db(12kHz 하이셸프, -9~+9dB)를 추가한다.
NSynth는 16kHz 소스라 8kHz 위에 원본 내용이 없다 — 48kHz로 리샘플해도 8kHz 위는
보간이 채운 것뿐이라 실제 에너지가 없어야 한다. 따라서 이 축을 아무리 움직여도
오디오가(따라서 CLAP 임베딩이) 거의 바뀌지 않아야 하고, J_fd도 0에 가까워야 한다.

이 스크립트는 highshelf의 실제 3개 파라미터(gain_db, cutoff_frequency_hz, q)와
널 축(ultrasonic_gain_db)을 같은 200개 평가점·같은 렌더링 파이프라인에서 동시에
중앙차분으로 미분해, 널 축이 실제 축들 대비 무시할 수준인지 직접 비교한다.

★ 이 널 축 정의는 이 5차 FD 스크립트들에만 국한된다. 01_embed.py의 PARAM_SPACE나
  3·4차 embeddings.npz/results.json에는 손대지 않는다.

★ 판정 게이트 0: null_axis_norm / min(실제 축 norm) 비율로 통과/중단을 가른다.
  기준은 아래 VERDICT_* 상수와 README에 표로 명시한다. 실패하면 유한차분 구현이나
  리샘플 파이프라인에 문제가 있다는 뜻이므로 Phase 1 이후를 진행하지 않는다.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
from huggingface_hub import hf_hub_download
from pedalboard import HighShelfFilter, Pedalboard
from scipy.stats import qmc
from tqdm import tqdm

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

# 01_embed.py와 동일한 전처리 상수 — 반드시 일치해야 e_dry/e_wet이 3·4차와 같은 조건이 된다.
SAMPLE_RATE = 48000
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
PEAK_TARGET = 0.7
SILENCE_PEAK_THRESHOLD = 1e-4

CLAP_REPO_ID = "lukewys/laion_clap"
CLAP_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"

# highshelf 실제 3축 + 널 축. gain_db/cutoff/q range는 01_embed.py PARAM_SPACE["highshelf"]와
# 동일 — 다른 값이면 같은 조건에서 비교가 안 된다.
ULTRASONIC_CUTOFF_HZ = 12000.0
ULTRASONIC_Q = 0.7071067811865476  # butterworth 기본값, ANCHOR_THETA의 q와 동일
FD_PARAM_SPACE = {
    "gain_db": {"range": (-9.0, 9.0), "scale": "linear", "is_null_axis": False},
    "cutoff_frequency_hz": {"range": (500.0, 8000.0), "scale": "log", "is_null_axis": False},
    "q": {"range": (0.3, 3.0), "scale": "log", "is_null_axis": False},
    "ultrasonic_gain_db": {"range": (-9.0, 9.0), "scale": "linear", "is_null_axis": True},
}
PARAM_ORDER = ["gain_db", "cutoff_frequency_hz", "q", "ultrasonic_gain_db"]
NULL_AXIS = "ultrasonic_gain_db"

# 판정 게이트 0 임계값 — null_axis_norm_mean / min(실제 축 norm_mean) 비율 기준.
# 코드가 결론을 단정하지 않도록, 이 임계값 자체를 결과와 함께 그대로 출력·기록한다.
VERDICT_PASS_RATIO = 0.10   # 널 축이 가장 작은 실제 축의 10% 미만 -> 통과
VERDICT_BORDERLINE_RATIO = 0.30  # 10~30% -> 경계 (해석 주의, 계속 진행은 가능)
# 30% 이상 -> 실패


def load_and_preprocess(path: Path):
    """01_embed.py와 동일: 48kHz/모노/4초, 피크 0.7 정규화. 무음이면 None."""
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


def render_highshelf(y: np.ndarray, theta_raw: dict) -> np.ndarray:
    """base highshelf(gain_db, cutoff_frequency_hz, q) + 널 축(ultrasonic_gain_db, 12kHz)을
    직렬로 적용. 01_embed.py의 apply_effect와 동일한 클리핑 방지 후처리를 쓴다."""
    board = Pedalboard(
        [
            HighShelfFilter(
                cutoff_frequency_hz=theta_raw["cutoff_frequency_hz"],
                gain_db=theta_raw["gain_db"],
                q=theta_raw["q"],
            ),
            HighShelfFilter(
                cutoff_frequency_hz=ULTRASONIC_CUTOFF_HZ,
                gain_db=theta_raw["ultrasonic_gain_db"],
                q=ULTRASONIC_Q,
            ),
        ]
    )
    wet = board(y, SAMPLE_RATE)
    peak = float(np.abs(wet).max())
    if peak > 1.0:
        wet = wet * (0.99 / peak)
    return wet.astype(np.float32)


def to_raw(param: str, u: float) -> float:
    spec = FD_PARAM_SPACE[param]
    lo, hi = spec["range"]
    u = float(np.clip(u, 0.0, 1.0))
    if spec["scale"] == "log":
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    return float(lo + u * (hi - lo))


def download_clap_checkpoint(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / CLAP_FILENAME
    if not ckpt_path.exists():
        print(f"{CLAP_FILENAME}이 로컬에 없습니다. Hugging Face에서 다운로드합니다...")
        try:
            hf_hub_download(repo_id=CLAP_REPO_ID, filename=CLAP_FILENAME, local_dir=cache_dir)
        except Exception as e:
            print(
                f"체크포인트 다운로드 실패: {e}\n수동으로 받아 {ckpt_path}에 두세요:\n"
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


def pick_eval_sources(audio_dir: Path, n_points: int, seed: int) -> list[Path]:
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"오류: {audio_dir}에서 wav 파일을 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(wav_files))
    chosen = []
    for idx in order:
        if len(chosen) >= n_points:
            break
        p = wav_files[idx]
        chosen.append(p)
    return chosen


def build_eval_points(sources: list[Path], seed: int):
    """소스마다 (gain_db, cutoff, q, ultrasonic_gain_db) 정규화 좌표를 결합 LHS로 1개씩 뽑는다."""
    n = len(sources)
    sampler = qmc.LatinHypercube(d=4, seed=np.random.default_rng([seed, 12345]))
    unit = sampler.random(n=n)
    points = []
    for i, src in enumerate(sources):
        u = {p: float(unit[i, j]) for j, p in enumerate(PARAM_ORDER)}
        points.append({"source": src, "u": u})
    return points


def main():
    parser = argparse.ArgumentParser(description="5차 Phase 0 — 유한차분 널 축 검증 (highshelf ultrasonic_gain_db)")
    parser.add_argument("--audio-dir", type=str, default="nsynth-test/audio")
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--h", type=float, default=0.05, help="정규화 [0,1] 좌표에서의 중앙차분 스텝")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    if args.device == "mps":
        import os

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print("CLAP 모델 로딩 중...")
    clap = load_clap(device, Path(__file__).parent / "ckpts")

    print(f"평가점 {args.n_points}개 소스 선택 중...")
    sources = pick_eval_sources(audio_dir, args.n_points, args.seed)

    # 무음 소스는 건너뛰고 채운다.
    loaded = []
    n_skipped = 0
    all_candidates = pick_eval_sources(audio_dir, args.n_points * 2, args.seed)
    for p in all_candidates:
        if len(loaded) >= args.n_points:
            break
        y = load_and_preprocess(p)
        if y is None:
            n_skipped += 1
            continue
        loaded.append((p, y))
    if len(loaded) < args.n_points:
        print(f"경고: 무음 소스 {n_skipped}개 제외 후 {len(loaded)}개만 확보 (요청 {args.n_points}).")

    eval_points = build_eval_points([p for p, _ in loaded], args.seed)
    y_by_source = {str(p): y for p, y in loaded}

    h = args.h
    # 각 평가점마다: center 1회 + (4축 x 2방향, 단 경계에서는 편측이라 한쪽만) 렌더링을 준비한다.
    render_jobs = []  # (point_idx, tag) tag in {"center", f"{param}+", f"{param}-"}
    job_conditions = []  # theta_raw dict per job

    onesided_flags = {p: [] for p in PARAM_ORDER}  # point별 편측 여부 기록용
    plan_per_point = []  # point_idx -> {"center": True, param: {"plus": bool, "minus": bool, "mode": "central"/"fwd"/"bwd"}}

    for pi, pt in enumerate(eval_points):
        u = pt["u"]
        plan = {}
        for param in PARAM_ORDER:
            u0 = u[param]
            can_plus = (u0 + h) <= 1.0
            can_minus = (u0 - h) >= 0.0
            if can_plus and can_minus:
                mode = "central"
            elif can_plus:
                mode = "fwd"  # (e(u+h) - e(u)) / h
            elif can_minus:
                mode = "bwd"  # (e(u) - e(u-h)) / h
            else:
                mode = "central"  # 범위가 h보다 좁은 축은 없음(0..1, h<=0.5 가정)이지만 방어적으로.
            plan[param] = mode
            onesided_flags[param].append(mode != "central")
        plan_per_point.append(plan)

        theta_center = {p: to_raw(p, u[p]) for p in PARAM_ORDER}
        job_conditions.append(("center", pi, theta_center))
        render_jobs.append((pi, "center"))

        for param in PARAM_ORDER:
            mode = plan[param]
            u0 = u[param]
            if mode == "central":
                sides = [("+", u0 + h), ("-", u0 - h)]
            elif mode == "fwd":
                sides = [("+", u0 + h)]
            else:
                sides = [("-", u0 - h)]
            for sign, u_shift in sides:
                theta = dict(theta_center)
                theta[param] = to_raw(param, u_shift)
                tag = f"{param}{sign}"
                job_conditions.append((tag, pi, theta))
                render_jobs.append((pi, tag))

    print(f"렌더링 작업 {len(job_conditions)}개 (평가점 {len(eval_points)}개 x 축 {len(PARAM_ORDER)}개, 중앙/편측 혼합)")

    # 렌더 + 배치 CLAP 임베딩
    embeddings_by_job = {}  # (pi, tag) -> 512-dim np.array
    batch_audio, batch_keys = [], []

    def flush():
        if not batch_audio:
            return
        emb = embed_batch(clap, device, batch_audio)
        for k, e in zip(batch_keys, emb):
            embeddings_by_job[k] = e
        batch_audio.clear()
        batch_keys.clear()

    for tag, pi, theta in tqdm(job_conditions, desc="렌더링+임베딩"):
        src_path, y = loaded[pi]
        wet = render_highshelf(y, theta)
        batch_audio.append(wet)
        batch_keys.append((pi, tag))
        if len(batch_audio) >= args.batch_size:
            flush()
    flush()

    # 중앙/편측 차분 계산
    jac_norms = {p: [] for p in PARAM_ORDER}
    for pi in range(len(eval_points)):
        for param in PARAM_ORDER:
            mode = plan_per_point[pi][param]
            if mode == "central":
                e_plus = embeddings_by_job[(pi, f"{param}+")]
                e_minus = embeddings_by_job[(pi, f"{param}-")]
                deriv = (e_plus - e_minus) / (2 * h)
            elif mode == "fwd":
                e_plus = embeddings_by_job[(pi, f"{param}+")]
                e_center = embeddings_by_job[(pi, "center")]
                deriv = (e_plus - e_center) / h
            else:
                e_center = embeddings_by_job[(pi, "center")]
                e_minus = embeddings_by_job[(pi, f"{param}-")]
                deriv = (e_center - e_minus) / h
            jac_norms[param].append(float(np.linalg.norm(deriv)))

    stats = {}
    for param in PARAM_ORDER:
        arr = np.array(jac_norms[param])
        stats[param] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(len(arr)),
            "onesided_ratio": float(np.mean(onesided_flags[param])),
        }

    real_axes = [p for p in PARAM_ORDER if p != NULL_AXIS]
    null_mean = stats[NULL_AXIS]["mean"]
    min_real_mean = min(stats[p]["mean"] for p in real_axes)
    ratio = null_mean / min_real_mean if min_real_mean > 0 else float("inf")

    if ratio < VERDICT_PASS_RATIO:
        verdict = "pass"
    elif ratio < VERDICT_BORDERLINE_RATIO:
        verdict = "borderline"
    else:
        verdict = "fail"

    print("\n=== Phase 0 결과: J_fd 노름 (축별) ===")
    print(f"{'param':<24}{'mean':>10}{'median':>10}{'std':>10}{'onesided%':>12}")
    for param in PARAM_ORDER:
        s = stats[param]
        tag = " (NULL)" if param == NULL_AXIS else ""
        print(f"{param+tag:<24}{s['mean']:>10.5f}{s['median']:>10.5f}{s['std']:>10.5f}{s['onesided_ratio']*100:>11.1f}%")
    print(f"\nnull_axis_mean / min(real_axis_mean) = {ratio:.4f}  ->  verdict = {verdict}")
    print(f"(기준: <{VERDICT_PASS_RATIO}=pass, <{VERDICT_BORDERLINE_RATIO}=borderline, else fail)")

    # --- 플롯 ---
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    colors = ["#2a78d6", "#2a78d6", "#2a78d6", "#e34948"]
    data = [jac_norms[p] for p in PARAM_ORDER]
    labels = [p + ("\n(널 축)" if p == NULL_AXIS else "") for p in PARAM_ORDER]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor(c)
    for median in bp["medians"]:
        median.set_color("#1a1a1a")
    ax.set_yscale("log")
    ax.set_ylabel("‖J_fd[:, param]‖  (log scale)")
    # 한글 폰트(AppleGothic)에 유니코드 마이너스(U+2212) 글리프가 없어 로그축 지수 표기가
    # 깨진다 — 눈금 숫자만 DejaVu Sans로 강제한다.
    for label in ax.get_yticklabels():
        label.set_fontfamily("DejaVu Sans")
    ax.set_title(f"Phase 0 — 널 축 검증 (h={h}, n={len(eval_points)}점)\nnull/min(real) = {ratio:.4f} -> {verdict}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "phase0_null_axis.png")
    plt.close(fig)

    # --- results_5.json ---
    results5_path = out_dir / "results_5.json"
    results5 = {}
    if results5_path.exists():
        with open(results5_path) as f:
            results5 = json.load(f)

    results5.setdefault("meta", {})
    results5["meta"].update(
        {
            "experiment_version": 5,
            "phase0_fd_points": len(eval_points),
            "phase0_h": h,
            "phase0_seed": args.seed,
            "phase0_audio_dir": str(audio_dir),
            "phase0_n_skipped_silent": n_skipped,
            "phase0_null_axis_param_space": FD_PARAM_SPACE,
            "phase0_ultrasonic_cutoff_hz": ULTRASONIC_CUTOFF_HZ,
            "phase0_ultrasonic_q": ULTRASONIC_Q,
            "phase0_verdict_thresholds": {
                "pass_ratio_lt": VERDICT_PASS_RATIO,
                "borderline_ratio_lt": VERDICT_BORDERLINE_RATIO,
                "definition": "null_axis_jacobian_norm_mean / min(real_axis_jacobian_norm_mean)",
            },
            "onesided_ratio": {p: stats[p]["onesided_ratio"] for p in PARAM_ORDER},
        }
    )
    results5["phase0"] = {
        "null_axis_norm": stats[NULL_AXIS],
        "other_axis_norms": {p: stats[p] for p in real_axes},
        "ratio_null_to_min_real": ratio,
        "verdict": verdict,
    }

    with open(results5_path, "w") as f:
        json.dump(results5, f, indent=2, ensure_ascii=False)

    print(f"\n완료: {results5_path}, {out_dir / 'phase0_null_axis.png'}")
    print("★ 여기서 멈춥니다. 판정 게이트 0 결과를 확인한 뒤 Phase 1 진행 여부를 결정하세요.")


if __name__ == "__main__":
    main()
