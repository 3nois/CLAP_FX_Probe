"""CLAP FX Probe — 10_fd_phase1.py (5차 Phase 1: 유한차분 J 계산과 대리모델 판정)

Phase 0(09_fd_phase0.py)에서 널 축이 실제 축 대비 무시할 수준임을 확인했다(사용자
판정: 통과로 간주). 이 스크립트는 그 유한차분(FD) 인프라를 reverb/distortion/highshelf
전체 파라미터로 확장해 "정답" J_fd를 만들고, 02/03이 학습한 대리모델의 autodiff
J_surrogate와 같은 (src_id, θ) 점에서 직접 대조한다.

  ① 게이트가 실제로 없다        ← 수학적으로 불가능 (wet_level=0이면 damping 등은
                                   출력에 영향을 줄 수 없으므로, d(f)/d(damping)이
                                   wet_level에 비례하는 것은 항등식이다)
  ② 대리모델이 못 배웠다        ← 유력
  ③ J 계산 코드에 버그가 있다   ← 배제 못 함

J_fd로 4차 과제 5(게이트)를 다시 하면 ①②③을 가를 수 있다: rho가 높으면 게이트는
실재하고 대리모델이 그 구조를 학습하지 못한 것(②, Phase 2로), rho가 낮으면 J 계산이나
게이트 전제 자체를 재검토해야 한다(③, Phase 2를 해도 소용없다).

★ cutoff_frequency_hz 정규화는 01_embed.py 원래 범위(500~8000, log)를 그대로 쓴다 —
  안 그러면 대리모델의 학습된 좌표계와 어긋나 비교 자체가 성립하지 않는다. "500~4000으로
  축소"는 이 스크립트가 평가점을 뽑을 때 원본 좌표계 안에서 어느 구간을 표본으로 삼을지의
  문제이지, 좌표계 정의를 바꾸는 게 아니다 (아래 sample_cutoff_center_u 참고).

★ 3·4차 embeddings.npz/results.json은 건드리지 않는다. 출력은 out/results_5.json의
  "phase1" 섹션과 새 PNG들뿐이다.

결과 해석은 이 스크립트가 단정하지 않는다. README의 판정 기준표를 따를 것.
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
import torch.nn as nn
from huggingface_hub import hf_hub_download
from pedalboard import Distortion, HighShelfFilter, Pedalboard, Reverb
from scipy.stats import qmc, spearmanr
from torch.func import jacrev, vmap
from tqdm import tqdm

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
COLORS = {"reverb": "#2a78d6", "distortion": "#eb6834", "highshelf": "#1baf7a", "baseline": "#898781", "null": "#e34948"}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def dejavu_yticks(ax):
    """한글 폰트에 유니코드 마이너스 글리프가 없어 로그축 지수표기가 깨진다 — 숫자만 강제."""
    for label in ax.get_yticklabels():
        label.set_fontfamily("DejaVu Sans")


# ---------------------------------------------------------------------------
# 렌더링/전처리 상수 — 01_embed.py와 반드시 동일해야 e_dry/e_wet이 같은 조건이 된다.
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48000
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
PEAK_TARGET = 0.7
SILENCE_PEAK_THRESHOLD = 1e-4

CLAP_REPO_ID = "lukewys/laion_clap"
CLAP_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"

# 01_embed.py PARAM_SPACE와 완전히 동일 — 정규화 좌표계가 어긋나면 대리모델과의 비교가
# 무효가 된다. main()에서 embed_config.json과 range를 대조 검증한다.
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
        "gain_db": {"range": (-9.0, 9.0), "scale": "linear"},
        "cutoff_frequency_hz": {"range": (500.0, 8000.0), "scale": "log"},
        "q": {"range": (0.3, 3.0), "scale": "log"},
    },
}
EFFECTS = ["reverb", "distortion", "highshelf"]
PARAM_ORDER = {e: list(PARAM_SPACE[e].keys()) for e in EFFECTS}

# ★ 5차 신규: highshelf 널 축. 대리모델은 이 축을 모른다 — surrogate 비교에서는 제외하고
# FD 자체 검증(h-민감도, 노름)에만 포함시킨다.
NULL_AXIS_PARAM = "ultrasonic_gain_db"
ULTRASONIC_CUTOFF_HZ = 12000.0
ULTRASONIC_Q = 0.7071067811865476
PARAM_SPACE["highshelf"][NULL_AXIS_PARAM] = {"range": (-9.0, 9.0), "scale": "linear", "is_null_axis": True}
PARAM_ORDER["highshelf"] = PARAM_ORDER["highshelf"] + [NULL_AXIS_PARAM]

CONTINUOUS_PARAMS = []  # [(effect, param), ...] — freeze_mode 제외, h 기반 FD 대상
for _e in EFFECTS:
    for _p in PARAM_ORDER[_e]:
        if PARAM_SPACE[_e][_p]["scale"] != "bernoulli":
            CONTINUOUS_PARAMS.append((_e, _p))

# ★ 1-A: cutoff_frequency_hz 평가점 "표본 추출" 범위만 500~4000으로 제한한다 (정규화
# 좌표계 range=(500,8000)은 유지). 사유: highshelf.cutoff_frequency_hz.range 500-8000 중
# 8000 부근은 NSynth Nyquist(8kHz)와 거의 겹쳐 셸빙할 대역이 거의 남지 않는다 — J_fd와
# J_surrogate 둘 다 0 근처가 되어 코사인 비교가 노이즈가 된다.
CUTOFF_SAMPLING_RANGE = (500.0, 4000.0)
CUTOFF_RANGE_NOTE = (
    "평가점 샘플링 범위만 500~4000Hz로 축소 (정규화 좌표계는 01_embed.py 원본 500~8000 유지). "
    "사유: NSynth Nyquist가 8kHz라 cutoff가 8kHz 근방이면 셸빙 대역이 거의 없어 "
    "J_fd/J_surrogate 둘 다 0 근처가 되고 코사인 비교가 순수 노이즈가 됨."
)

H_VALUES = [0.01, 0.02, 0.05]
GATE_TARGET_PARAMS = ["room_size", "damping", "width"]  # reverb, wet_level 대비 게이트 재검정 대상


# ---------------------------------------------------------------------------
# 정규화 유틸 — 01_embed.py normalize_theta / 역변환과 동일한 공식
# ---------------------------------------------------------------------------
def to_raw(effect: str, param: str, u: float) -> float:
    spec = PARAM_SPACE[effect][param]
    lo, hi = spec["range"]
    u = float(np.clip(u, 0.0, 1.0))
    if spec["scale"] == "log":
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    return float(lo + u * (hi - lo))


def to_norm(effect: str, param: str, raw: float) -> float:
    spec = PARAM_SPACE[effect][param]
    lo, hi = spec["range"]
    if spec["scale"] == "log":
        return float((np.log(raw) - np.log(lo)) / (np.log(hi) - np.log(lo)))
    return float((raw - lo) / (hi - lo))


# ---------------------------------------------------------------------------
# 렌더링 — 01_embed.py apply_effect과 동일 로직 + highshelf 널 축 추가
# ---------------------------------------------------------------------------
def load_and_preprocess(path: Path):
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


def render(y: np.ndarray, effect_name: str, theta_raw: dict) -> np.ndarray:
    if effect_name == "reverb":
        board = Pedalboard(
            [
                Reverb(
                    room_size=theta_raw["room_size"],
                    damping=theta_raw["damping"],
                    wet_level=theta_raw["wet_level"],
                    dry_level=1.0,
                    width=theta_raw["width"],
                    freeze_mode=theta_raw["freeze_mode"],
                )
            ]
        )
    elif effect_name == "distortion":
        board = Pedalboard([Distortion(drive_db=theta_raw["drive_db"])])
    elif effect_name == "highshelf":
        board = Pedalboard(
            [
                HighShelfFilter(
                    cutoff_frequency_hz=theta_raw["cutoff_frequency_hz"],
                    gain_db=theta_raw["gain_db"],
                    q=theta_raw["q"],
                ),
                HighShelfFilter(
                    cutoff_frequency_hz=ULTRASONIC_CUTOFF_HZ,
                    gain_db=theta_raw[NULL_AXIS_PARAM],
                    q=ULTRASONIC_Q,
                ),
            ]
        )
    else:
        raise ValueError(effect_name)
    wet = board(y, SAMPLE_RATE)
    peak = float(np.abs(wet).max())
    if peak > 1.0:
        wet = wet * (0.99 / peak)
    return wet.astype(np.float32)


# ---------------------------------------------------------------------------
# CLAP
# ---------------------------------------------------------------------------
def download_clap_checkpoint(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / CLAP_FILENAME
    if not ckpt_path.exists():
        print(f"{CLAP_FILENAME}이 로컬에 없습니다. Hugging Face에서 다운로드합니다...")
        try:
            hf_hub_download(repo_id=CLAP_REPO_ID, filename=CLAP_FILENAME, local_dir=cache_dir)
        except Exception as e:
            print(f"체크포인트 다운로드 실패: {e}", file=sys.stderr)
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


# ---------------------------------------------------------------------------
# 대리모델 — 02/03과 동일 구조, 동일 인터페이스로 로딩
# ---------------------------------------------------------------------------
class SurrogateMLP(nn.Module):
    def __init__(self, in_dim, hidden=1024, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_surrogate(out_dir: Path, device):
    ckpt = torch.load(out_dir / "surrogate_model.pt", map_location=device, weights_only=False)
    model = SurrogateMLP(in_dim=ckpt["in_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt["theta_slots"]


def build_forward_fn(model, effect_name, theta_slots, theta_width, device):
    onehot = torch.zeros(len(EFFECTS), device=device)
    onehot[EFFECTS.index(effect_name)] = 1.0
    start, end = theta_slots[effect_name]
    pre, post = start, theta_width - end

    def f(theta_active, dry):
        theta_full = torch.cat([torch.zeros(pre, device=device), theta_active, torch.zeros(post, device=device)])
        x = torch.cat([dry, onehot, theta_full])
        delta = model(x.unsqueeze(0)).squeeze(0)
        return dry + delta

    return f


def batched_jacobian(f, theta_batch: torch.Tensor, dry_batch: torch.Tensor) -> torch.Tensor:
    return vmap(jacrev(f, argnums=0), in_dims=(0, 0))(theta_batch, dry_batch)


def batched_forward(f, theta_batch: torch.Tensor, dry_batch: torch.Tensor) -> torch.Tensor:
    return vmap(f, in_dims=(0, 0))(theta_batch, dry_batch)


# ---------------------------------------------------------------------------
# 소스 분할 — 02_surrogate.py / 03_jacobian.py와 동일 (테스트셋에서만 평가점을 뽑아야
# 대리모델이 학습 때 본 적 없는 데이터로 공정하게 비교된다).
# ---------------------------------------------------------------------------
def split_sources(unique_src_ids: np.ndarray, seed: int, test_size: float = 0.3):
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_src_ids)
    n_test = max(1, int(round(len(shuffled) * test_size)))
    return set(shuffled[:n_test].tolist())


def stratified_sample_by_family(src_ids: list, family_by_src: dict, n_target: int, seed: int):
    """패밀리별로 셔플한 뒤 라운드로빈으로 뽑는다 — 표본이 적은 패밀리는 가진 만큼만,
    나머지는 표본이 많은 패밀리끼리 균등하게 채운다."""
    rng = np.random.RandomState(seed)
    by_family = {}
    for s in src_ids:
        by_family.setdefault(family_by_src[s], []).append(s)
    for fam in by_family:
        rng.shuffle(by_family[fam])
    families_sorted = sorted(by_family.keys())
    pointers = {f: 0 for f in families_sorted}
    selected = []
    while len(selected) < n_target:
        progressed = False
        for fam in families_sorted:
            if len(selected) >= n_target:
                break
            pool = by_family[fam]
            p = pointers[fam]
            if p < len(pool):
                selected.append(pool[p])
                pointers[fam] = p + 1
                progressed = True
        if not progressed:
            break  # 전체 풀 소진
    return selected


# ---------------------------------------------------------------------------
# 평가점 θ 샘플링 (결합 LHS, 소스당 이펙트별로 1개)
# ---------------------------------------------------------------------------
def sample_eval_thetas(n_points: int, seed: int):
    """반환: {effect: [{param: u_norm, ...}, ...]} — 소스 인덱스 순서와 정렬됨."""
    thetas = {}
    for effect in EFFECTS:
        continuous = [p for p in PARAM_ORDER[effect] if PARAM_SPACE[effect][p]["scale"] != "bernoulli"]
        sampler = qmc.LatinHypercube(d=len(continuous), seed=np.random.default_rng([seed, hash(effect) % (2**31)]))
        unit = sampler.random(n=n_points)
        rng = np.random.RandomState([seed, 999])
        effect_thetas = []
        for i in range(n_points):
            u = {}
            for j, p in enumerate(continuous):
                if effect == "highshelf" and p == "cutoff_frequency_hz":
                    # ★ 표본 추출만 500~4000Hz 제한, 정규화는 원본 500~8000 좌표계.
                    lo, hi = CUTOFF_SAMPLING_RANGE
                    raw = float(np.exp(np.log(lo) + unit[i, j] * (np.log(hi) - np.log(lo))))
                    u[p] = to_norm(effect, p, raw)
                else:
                    u[p] = float(unit[i, j])
            for p in PARAM_ORDER[effect]:
                if PARAM_SPACE[effect][p]["scale"] == "bernoulli":
                    u[p] = float(rng.randint(0, 2))
            effect_thetas.append(u)
        thetas[effect] = effect_thetas
    return thetas


# ---------------------------------------------------------------------------
# FD job 계획: 평가점 1개(소스 1개) x 이펙트 3개 x (연속 파라미터 x h 3값 x 중앙/편측)
# + freeze_mode 0/1 차분
# ---------------------------------------------------------------------------
def plan_jobs_for_point(theta_center: dict):
    """theta_center: {effect: {param: u}}. 반환: jobs 리스트[(effect, tag, theta_raw_dict)],
    plan: {(effect,param): {h: mode}} mode in central/fwd/bwd."""
    jobs = []
    plan = {}

    for effect in EFFECTS:
        theta_c_raw = {p: to_raw(effect, p, theta_center[effect][p]) for p in PARAM_ORDER[effect]}
        jobs.append((effect, "center", dict(theta_c_raw)))

        for p in PARAM_ORDER[effect]:
            if PARAM_SPACE[effect][p]["scale"] == "bernoulli":
                continue
            u0 = theta_center[effect][p]
            for h in H_VALUES:
                can_plus = (u0 + h) <= 1.0
                can_minus = (u0 - h) >= 0.0
                if can_plus and can_minus:
                    mode = "central"
                elif can_plus:
                    mode = "fwd"
                elif can_minus:
                    mode = "bwd"
                else:
                    mode = "central"
                plan[(effect, p, h)] = mode
                if mode in ("central", "fwd"):
                    theta = dict(theta_c_raw)
                    theta[p] = to_raw(effect, p, u0 + h)
                    jobs.append((effect, f"{p}+{h}", theta))
                if mode in ("central", "bwd"):
                    theta = dict(theta_c_raw)
                    theta[p] = to_raw(effect, p, u0 - h)
                    jobs.append((effect, f"{p}-{h}", theta))

        # freeze_mode: 0->1 "차분" (미분 아님). center의 freeze 값과 겹치면 재사용.
        if "freeze_mode" in PARAM_ORDER[effect]:
            center_freeze = theta_center[effect]["freeze_mode"]
            other_freeze = 1.0 - center_freeze
            theta = dict(theta_c_raw)
            theta["freeze_mode"] = other_freeze
            jobs.append((effect, f"freeze_alt", theta))

    return jobs, plan


def main():
    parser = argparse.ArgumentParser(description="5차 Phase 1 — 유한차분 J 계산과 대리모델 판정")
    parser.add_argument("--audio-dir", type=str, default="nsynth-test/audio")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    if args.device == "mps":
        import os

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    audio_dir = Path(args.audio_dir)
    emb_path = Path(args.embeddings)
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path}가 없습니다.")
    if not (out_dir / "surrogate_model.pt").exists():
        raise FileNotFoundError(f"{out_dir / 'surrogate_model.pt'}가 없습니다. 먼저 02_surrogate.py를 실행하세요.")

    with open(Path(args.embeddings).parent / "embed_config.json") as f:
        embed_config = json.load(f)
    # 정규화 좌표계 대조 검증 — 하드코딩한 PARAM_SPACE range가 3차 학습 때와 어긋나면
    # 대리모델과의 비교 전체가 무효가 된다.
    for e in EFFECTS:
        for p in embed_config["param_order"][e]:
            expected = embed_config["param_space"][e][p]
            if expected == "bernoulli_0.5":
                assert PARAM_SPACE[e][p]["scale"] == "bernoulli"
            else:
                assert tuple(PARAM_SPACE[e][p]["range"]) == tuple(expected), f"{e}.{p} range mismatch: {PARAM_SPACE[e][p]['range']} vs {expected}"

    device = torch.device(args.device)
    print("CLAP 모델 로딩 중...")
    clap = load_clap(device, Path(__file__).parent / "ckpts")

    print("대리모델 로딩 중...")
    surrogate, theta_slots = load_surrogate(out_dir, device)
    theta_slots = {e: tuple(v) for e, v in theta_slots.items()}
    theta_width = embed_config["theta_width"]

    print("임베딩 메타데이터 로딩 중 (소스 분할 + 패밀리 층화용)...")
    d = np.load(emb_path, allow_pickle=False)
    dry_mask = d["effect"] == "dry"
    dry_src_ids = d["src_id"][dry_mask]
    dry_by_src = dict(zip(dry_src_ids.tolist(), d["embeddings"][dry_mask]))
    family_by_src = dict(zip(dry_src_ids.tolist(), d["instrument_family"][dry_mask].tolist()))
    filename_by_src = dict(zip(dry_src_ids.tolist(), d["filename"][dry_mask].tolist()))

    unique_srcs = np.unique(dry_src_ids)
    test_srcs = split_sources(unique_srcs, args.seed, args.test_size)
    print(f"테스트셋 소스 {len(test_srcs)}개 중 {args.n_points}개를 패밀리 층화로 추출합니다...")
    selected_srcs = stratified_sample_by_family(sorted(test_srcs), family_by_src, args.n_points, args.seed)
    n_points = len(selected_srcs)
    if n_points < args.n_points:
        print(f"경고: 테스트셋 풀이 부족해 {n_points}개만 확보했습니다 (요청 {args.n_points}).")

    import collections

    family_dist = collections.Counter(family_by_src[s] for s in selected_srcs)
    print(f"패밀리 분포: {dict(family_dist)}")

    print("평가점 θ 샘플링 중 (결합 LHS, 이펙트별)...")
    theta_u = sample_eval_thetas(n_points, args.seed)  # {effect: [dict, ...]} len n_points

    # 소스별 오디오 로딩
    print("소스 오디오 로딩 중...")
    y_by_idx = {}
    for i, src in enumerate(tqdm(selected_srcs, desc="오디오 로딩")):
        fname = filename_by_src[src]
        y = load_and_preprocess(audio_dir / fname)
        if y is None:
            raise RuntimeError(f"src_id={src} ({fname})가 무음입니다 — 3차 때는 통과했는데 이상합니다.")
        y_by_idx[i] = y

    # ---- job 계획 ----
    all_jobs = []  # (point_idx, effect, tag, theta_raw)
    all_plans = []  # point_idx -> {(effect,param,h): mode}
    for i in range(n_points):
        theta_center = {e: theta_u[e][i] for e in EFFECTS}
        jobs, plan = plan_jobs_for_point(theta_center)
        all_plans.append(plan)
        for effect, tag, theta_raw in jobs:
            all_jobs.append((i, effect, tag, theta_raw))

    print(f"렌더링 작업 {len(all_jobs)}개 (평가점 {n_points}개, h={H_VALUES})")

    embeddings_by_job = {}
    batch_audio, batch_keys = [], []

    def flush():
        if not batch_audio:
            return
        emb = embed_batch(clap, device, batch_audio)
        for k, e in zip(batch_keys, emb):
            embeddings_by_job[k] = e
        batch_audio.clear()
        batch_keys.clear()

    for pi, effect, tag, theta_raw in tqdm(all_jobs, desc="렌더링+임베딩"):
        wet = render(y_by_idx[pi], effect, theta_raw)
        batch_audio.append(wet)
        batch_keys.append((pi, effect, tag))
        if len(batch_audio) >= args.batch_size:
            flush()
    flush()

    # ---- FD 미분/차분 계산 ----
    # jac[h][(effect,param)] -> list of 512-d np arrays (len n_points), 순서는 selected_srcs와 정렬
    jac = {h: {} for h in H_VALUES}
    onesided_count = {}
    freeze_diff = {e: [] for e in EFFECTS if "freeze_mode" in PARAM_ORDER[e]}

    for i in range(n_points):
        plan = all_plans[i]
        for effect in EFFECTS:
            for p in PARAM_ORDER[effect]:
                if PARAM_SPACE[effect][p]["scale"] == "bernoulli":
                    continue
                for h in H_VALUES:
                    mode = plan[(effect, p, h)]
                    key = (effect, p)
                    jac[h].setdefault(key, [])
                    onesided_count.setdefault(key, [])
                    if mode == "central":
                        ep = embeddings_by_job[(i, effect, f"{p}+{h}")]
                        em = embeddings_by_job[(i, effect, f"{p}-{h}")]
                        deriv = (ep - em) / (2 * h)
                    elif mode == "fwd":
                        ep = embeddings_by_job[(i, effect, f"{p}+{h}")]
                        ec = embeddings_by_job[(i, effect, "center")]
                        deriv = (ep - ec) / h
                    else:
                        ec = embeddings_by_job[(i, effect, "center")]
                        em = embeddings_by_job[(i, effect, f"{p}-{h}")]
                        deriv = (ec - em) / h
                    jac[h][key].append(deriv)
                    if h == H_VALUES[0]:
                        onesided_count[key].append(mode != "central")

        for effect in freeze_diff:
            ec = embeddings_by_job[(i, effect, "center")]
            ealt = embeddings_by_job[(i, effect, "freeze_alt")]
            # freeze_alt = 1-center_freeze 값. 방향을 "0->1"로 통일한다.
            cf = theta_u[effect][i]["freeze_mode"]
            if cf == 0.0:
                diff = ealt - ec  # 0->1
            else:
                diff = ec - ealt  # 0->1 (center가 이미 1이므로 center - alt(=0))
            freeze_diff[effect].append(diff)

    onesided_ratio = {f"{e}.{p}": float(np.mean(v)) for (e, p), v in onesided_count.items()}

    # ---- h 민감도 ----
    h_sensitivity_cosine = {}  # "effect.param" -> {"h1_vs_h2":..., "h1_vs_h3":..., "h2_vs_h3":..., "median_all": ...}
    h_avg_agreement = {h: [] for h in H_VALUES}  # 각 h가 나머지 두 h와 평균적으로 얼마나 일치하는지 (전체 파라미터 평균)
    for effect, p in CONTINUOUS_PARAMS:
        key = (effect, p)
        pair_results = {}
        all_pairs_median = []
        for hi in range(len(H_VALUES)):
            for hj in range(hi + 1, len(H_VALUES)):
                h1, h2 = H_VALUES[hi], H_VALUES[hj]
                cos_vals = []
                for a, b in zip(jac[h1][key], jac[h2][key]):
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    if na < 1e-10 or nb < 1e-10:
                        continue
                    cos_vals.append(float(np.dot(a, b) / (na * nb)))
                med = float(np.median(cos_vals)) if cos_vals else None
                pair_results[f"h{h1}_vs_h{h2}"] = {"median_cosine": med, "n": len(cos_vals)}
                if med is not None:
                    all_pairs_median.append(med)
                    h_avg_agreement[h1].append(med)
                    h_avg_agreement[h2].append(med)
        pair_results["median_all_pairs"] = float(np.median(all_pairs_median)) if all_pairs_median else None
        h_sensitivity_cosine[f"{effect}.{p}"] = pair_results

    h_agreement_mean = {h: float(np.mean(v)) if v else None for h, v in h_avg_agreement.items()}
    h_selected = max(H_VALUES, key=lambda h: (h_agreement_mean[h] if h_agreement_mean[h] is not None else -1))
    unstable_params = [k for k, v in h_sensitivity_cosine.items() if v["median_all_pairs"] is not None and v["median_all_pairs"] < 0.95]

    print(f"h 민감도: 평균 상호일치도 {h_agreement_mean} -> h_selected={h_selected}")
    if unstable_params:
        print(f"경고: 다음 파라미터는 h간 코사인 중앙값이 0.95 미만입니다: {unstable_params}")

    # ---- fd.jacobian_norm_by_param (h_selected 기준) ----
    jacobian_norm_by_param = {}
    for effect, p in CONTINUOUS_PARAMS:
        key = (effect, p)
        norms = np.array([np.linalg.norm(v) for v in jac[h_selected][key]])
        jacobian_norm_by_param[f"{effect}.{p}"] = {
            "mean": float(norms.mean()), "median": float(np.median(norms)), "std": float(norms.std()),
            "n": int(len(norms)), "is_null_axis": p == NULL_AXIS_PARAM,
        }
    for effect in freeze_diff:
        norms = np.array([np.linalg.norm(v) for v in freeze_diff[effect]])
        jacobian_norm_by_param[f"{effect}.freeze_mode"] = {
            "mean": float(norms.mean()), "median": float(np.median(norms)), "std": float(norms.std()),
            "n": int(len(norms)), "note": "0->1 차분이며 미분이 아님 (이산 변수)",
        }

    # ---- 1-D: 게이트 재검정 (h_selected, 원시 점 단위) ----
    wet_level_arr = np.array([theta_u["reverb"][i]["wet_level"] for i in range(n_points)])
    gate_spearman_by_param = {}
    gate_regression_by_param = {}
    for p in GATE_TARGET_PARAMS:
        norms = np.array([np.linalg.norm(v) for v in jac[h_selected][("reverb", p)]])
        rho, pvalue = spearmanr(wet_level_arr, norms)
        slope, intercept = np.polyfit(wet_level_arr, norms, 1)
        pred = slope * wet_level_arr + intercept
        ss_res = float(np.sum((norms - pred) ** 2))
        ss_tot = float(np.sum((norms - norms.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else None
        gate_spearman_by_param[p] = {"spearman_rho": float(rho), "spearman_pvalue": float(pvalue)}
        gate_regression_by_param[p] = {"slope": float(slope), "r2": r2}

    # ---- 1-C: 대리모델 J와 비교 (같은 (src_id, θ) 점) ----
    print("대리모델 J 계산 중 (같은 평가점)...")
    comparison_by_param = {}
    all_cos_pooled = []
    random_baseline_cos_pooled = []
    rng_rb = np.random.RandomState(args.seed)

    for effect in EFFECTS:
        real_params = [
            p for p in PARAM_ORDER[effect]
            if p != NULL_AXIS_PARAM and PARAM_SPACE[effect][p]["scale"] != "bernoulli"
        ]
        f = build_forward_fn(surrogate, effect, theta_slots, theta_width, device)
        theta_dim = theta_slots[effect][1] - theta_slots[effect][0]

        theta_batch = np.zeros((n_points, theta_dim), dtype=np.float32)
        for i in range(n_points):
            for j, p in enumerate(PARAM_ORDER[effect]):
                if p == NULL_AXIS_PARAM:
                    continue
                theta_batch[i, j] = theta_u[effect][i][p]
        dry_batch = np.stack([dry_by_src[selected_srcs[i]] for i in range(n_points)]).astype(np.float32)
        theta_t = torch.tensor(theta_batch, device=device)
        dry_t = torch.tensor(dry_batch, device=device)
        J_sur = batched_jacobian(f, theta_t, dry_t).detach().cpu().numpy()  # (n_points, 512, theta_dim)

        for pj, p in enumerate(real_params):
            key = (effect, p)
            fd_cols = jac[h_selected][key]  # list of (512,)
            sur_cols = J_sur[:, :, PARAM_ORDER[effect].index(p)]  # (n_points, 512)
            cos_vals, mag_ratios = [], []
            for i in range(n_points):
                a, b = fd_cols[i], sur_cols[i]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na < 1e-10:
                    continue
                cos = float(np.dot(a, b) / (na * nb + 1e-12))
                cos_vals.append(cos)
                mag_ratios.append(float(nb / na))
                # 무작위 기준선: a와 같은 노름 위치에서 무작위 단위벡터 vs a
                rv = rng_rb.normal(size=512)
                rv /= np.linalg.norm(rv)
                random_baseline_cos_pooled.append(float(np.dot(a / na, rv)))
            cos_arr = np.array(cos_vals)
            comparison_by_param[f"{effect}.{p}"] = {
                "cos_mean": float(cos_arr.mean()), "cos_median": float(np.median(cos_arr)),
                "cos_q1": float(np.percentile(cos_arr, 25)), "cos_q3": float(np.percentile(cos_arr, 75)),
                "cos_std": float(cos_arr.std()), "n": int(len(cos_arr)),
                "magnitude_ratio_median": float(np.median(mag_ratios)),
                "magnitude_ratio_mean": float(np.mean(mag_ratios)),
            }
            all_cos_pooled.extend(cos_vals)

        # freeze_mode: autodiff가 아니라 surrogate forward 직접 평가로 0->1 차분 비교
        if "freeze_mode" in PARAM_ORDER[effect]:
            theta_batch0 = theta_batch.copy()
            theta_batch1 = theta_batch.copy()
            fm_idx = PARAM_ORDER[effect].index("freeze_mode")
            theta_batch0[:, fm_idx] = 0.0
            theta_batch1[:, fm_idx] = 1.0
            with torch.no_grad():
                out0 = batched_forward(f, torch.tensor(theta_batch0, device=device), dry_t).cpu().numpy()
                out1 = batched_forward(f, torch.tensor(theta_batch1, device=device), dry_t).cpu().numpy()
            sur_diff = out1 - out0  # 0->1
            fd_diff = freeze_diff[effect]
            cos_vals = []
            for i in range(n_points):
                a, b = fd_diff[i], sur_diff[i]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na < 1e-10:
                    continue
                cos_vals.append(float(np.dot(a, b) / (na * nb + 1e-12)))
            cos_arr = np.array(cos_vals)
            comparison_by_param[f"{effect}.freeze_mode"] = {
                "cos_mean": float(cos_arr.mean()), "cos_median": float(np.median(cos_arr)),
                "cos_q1": float(np.percentile(cos_arr, 25)), "cos_q3": float(np.percentile(cos_arr, 75)),
                "cos_std": float(cos_arr.std()), "n": int(len(cos_arr)),
                "note": "0->1 차분 비교이며 미분이 아님 (surrogate도 forward 직접 평가로 비교)",
            }
            all_cos_pooled.extend(cos_vals)

    all_cos_arr = np.array(all_cos_pooled)
    random_baseline_arr = np.array(random_baseline_cos_pooled)

    # ---- 그림 ----
    print("그림 저장 중...")

    # fd_h_sensitivity.png
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    labels = [f"{e}.{p}" for e, p in CONTINUOUS_PARAMS]
    medians = [h_sensitivity_cosine[f"{e}.{p}"]["median_all_pairs"] for e, p in CONTINUOUS_PARAMS]
    colors = [COLORS["null"] if p == NULL_AXIS_PARAM else COLORS[e] for e, p in CONTINUOUS_PARAMS]
    x = np.arange(len(labels))
    ax.bar(x, [m if m is not None else 0 for m in medians], color=colors, zorder=3)
    ax.axhline(0.95, color="#e34948", linestyle="--", linewidth=1, label="0.95 (경고 기준)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"h값 3쌍 코사인 중앙값 (전체 중앙값)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"h 민감도 (h={H_VALUES}) — h_selected={h_selected}")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fd_h_sensitivity.png")
    plt.close(fig)

    # fd_vs_surrogate.png — 파라미터별 cos(J_fd, J_surrogate) 요약치(median, Q1-Q3)를 막대로.
    # 점별 원 분포는 results_5.json의 comparison.by_param에 median/Q1/Q3로 기록되어 있다.
    keys = list(comparison_by_param.keys())
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    meds = [comparison_by_param[k]["cos_median"] for k in keys]
    q1s = [comparison_by_param[k]["cos_q1"] for k in keys]
    q3s = [comparison_by_param[k]["cos_q3"] for k in keys]
    x = np.arange(len(keys))
    yerr = np.array([[m - q1 for m, q1 in zip(meds, q1s)], [q3 - m for m, q3 in zip(meds, q3s)]])
    bar_colors = [COLORS[k.split(".")[0]] for k in keys]
    ax.bar(x, meds, yerr=yerr, color=bar_colors, capsize=3, zorder=3)
    ax.axhline(float(np.median(random_baseline_arr)), color="#898781", linestyle="--", linewidth=1.2,
               label=f"무작위 기준선 (중앙값 {np.median(random_baseline_arr):.3f})")
    ax.axhline(0.8, color="#2a9d5c", linestyle=":", linewidth=1, label="0.8 (게이트1 상단)")
    ax.axhline(0.5, color="#e34948", linestyle=":", linewidth=1, label="0.5 (게이트1 하단)")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("cos(J_fd, J_surrogate)  (median, Q1-Q3)")
    ax.set_title(f"FD vs 대리모델 야코비안 방향 일치도 (h={h_selected}, n={n_points})")
    ax.legend(frameon=True, fontsize=8, loc="upper right")
    dejavu_yticks(ax)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fd_vs_surrogate.png")
    plt.close(fig)

    # fd_gate.png
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
    for ax, p in zip(axes, GATE_TARGET_PARAMS):
        norms = np.array([np.linalg.norm(v) for v in jac[h_selected][("reverb", p)]])
        ax.scatter(wet_level_arr, norms, s=10, alpha=0.5, color=COLORS["reverb"] if p != "width" else "#e34948")
        slope = gate_regression_by_param[p]["slope"]
        r2 = gate_regression_by_param[p]["r2"]
        rho = gate_spearman_by_param[p]["spearman_rho"]
        pval = gate_spearman_by_param[p]["spearman_pvalue"]
        xs = np.linspace(wet_level_arr.min(), wet_level_arr.max(), 50)
        ax.plot(xs, slope * xs + (norms.mean() - slope * wet_level_arr.mean()), color="black", linestyle="--", linewidth=1)
        note = " (음성 통제)" if p == "width" else ""
        ax.set_title(f"‖J_fd[:,{p}]‖{note}\nSpearman ρ={rho:.3f} (p={pval:.3f}), R²={r2:.3f}", fontsize=9)
        ax.set_xlabel("wet_level (정규화)")
        style_axis(ax)
    axes[0].set_ylabel("‖J_fd‖ (원시 점)")
    fig.suptitle(f"FD 게이트 재검정 (h={h_selected}, n={n_points}) — 대리모델 없이 계산한 정답")
    fig.tight_layout()
    fig.savefig(out_dir / "fd_gate.png")
    plt.close(fig)

    # ---- 1-E-(1): d_int 비율 재보고 (기존 results.json 재학습 없이 재표기만) ----
    ablation_ratio = {}
    results_path = out_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            r4 = json.load(f)
        for e, dec in r4.get("ablation", {}).get("decomposition", {}).items():
            d_total, d_th, d_e, d_int = dec["d_total"], dec["d_th"], dec["d_e"], dec["d_int"]
            ablation_ratio[e] = {
                "d_total": d_total, "d_th": d_th, "d_e": d_e, "d_int": d_int,
                "d_th_ratio": d_th / d_total if abs(d_total) > 1e-12 else None,
                "d_e_ratio": d_e / d_total if abs(d_total) > 1e-12 else None,
                "d_int_ratio": d_int / d_total if abs(d_total) > 1e-12 else None,
            }
    else:
        print("경고: out/results.json이 없어 d_int 비율을 재표기하지 못했습니다.")

    # ---- 1-E-(2): depends_on_surrogate 플래그 ----
    depends_on_surrogate = {
        "probe_r2": False, "instrument_family_control": False, "reverse_model": False,
        "injectivity": False, "resolution_floor": False,
        "family_cosine": True, "theta_dependence": True, "gate_structure": True, "subspace_projection": True,
        "text_alignment": True,  # 05_text_alignment.py는 jacrev 기반 오디오 방향(audio_direction_for_param)을 씀 — 확인됨
        "ablation_d_int": False,  # held-out 코사인(실측 e_wet 대비) 기반 예측 정확도이며 미분/야코비안이 아님
        "fd_jacobian_norm": False,  # 5차 신규: 유한차분 자체 (대리모델 미사용)
        "fd_vs_surrogate_comparison": "N/A",  # 정의상 둘 다에 의존 (비교 자체가 목적)
    }

    # ---- results_5.json 병합 저장 ----
    results5_path = out_dir / "results_5.json"
    results5 = {}
    if results5_path.exists():
        with open(results5_path) as f:
            results5 = json.load(f)

    results5.setdefault("meta", {})
    results5["meta"].update({
        "phase1_fd_points": n_points,
        "phase1_h_values": H_VALUES,
        "phase1_h_selected": h_selected,
        "phase1_h_agreement_mean": h_agreement_mean,
        "phase1_h_unstable_params": unstable_params,
        "phase1_cutoff_range_note": CUTOFF_RANGE_NOTE,
        "phase1_cutoff_sampling_range": list(CUTOFF_SAMPLING_RANGE),
        "phase1_onesided_ratio": onesided_ratio,
        "phase1_seed": args.seed,
        "phase1_test_size": args.test_size,
        "phase1_n_test_sources": len(test_srcs),
        "phase1_family_distribution": dict(family_dist),
        "phase1_selected_src_ids": [int(s) for s in selected_srcs],
        "phase1_theta_norm_by_effect": {
            e: [{p: float(v) for p, v in theta_u[e][i].items()} for i in range(n_points)] for e in EFFECTS
        },
        "phase1_audio_dir": str(audio_dir),
    })

    results5["phase1"] = {
        "fd": {
            "jacobian_norm_by_param": jacobian_norm_by_param,
            "h_sensitivity_cosine": h_sensitivity_cosine,
            "gate_spearman_by_param": gate_spearman_by_param,
            "gate_regression_by_param": gate_regression_by_param,
        },
        "comparison": {
            "by_param": comparison_by_param,
            "cos_fd_vs_surrogate_pooled": {
                "mean": float(all_cos_arr.mean()), "median": float(np.median(all_cos_arr)),
                "q1": float(np.percentile(all_cos_arr, 25)), "q3": float(np.percentile(all_cos_arr, 75)),
                "n": int(len(all_cos_arr)),
            },
            "random_baseline_cosine": {
                "mean": float(random_baseline_arr.mean()), "median": float(np.median(random_baseline_arr)),
                "std": float(random_baseline_arr.std()), "n": int(len(random_baseline_arr)),
            },
        },
        "ablation_ratio": ablation_ratio,
        "depends_on_surrogate": depends_on_surrogate,
        "text_alignment_basis_note": (
            "05_text_alignment.py의 audio_direction_for_param()은 vmap(jacrev(...))로 계산한 "
            "대리모델 야코비안의 평균 방향을 오디오 방향 v로 쓴다 (실측 임베딩의 평균차 벡터가 아님). "
            "따라서 4차 텍스트 정렬 결과(P24 포함)는 depends_on_surrogate=true이며, "
            "게이트1 판정(FD vs 대리모델 코사인)에 따라 재검토 대상이 된다."
        ),
    }

    with open(results5_path, "w") as f:
        json.dump(results5, f, indent=2, ensure_ascii=False)

    # ---- 콘솔 요약 ----
    median_cos = float(np.median(all_cos_arr))
    if median_cos > 0.8:
        gate1_verdict = "신뢰 (>0.8) — 3·4차 야코비안 결론 유지 -> Phase 3"
    elif median_cos > 0.5:
        gate1_verdict = "부분 신뢰 (0.5~0.8) — 방향만 쓰고 크기는 버림 -> Phase 3 (단서 명시)"
    else:
        gate1_verdict = "불신 (<0.5) -> Phase 2"

    print("\n=== Phase 1 결과 요약 ===")
    print(f"h_selected = {h_selected} (평균 상호일치도 {h_agreement_mean})")
    if unstable_params:
        print(f"⚠ h 불안정 파라미터 (median cos < 0.95): {unstable_params}")
    print(f"\ncos(J_fd, J_surrogate) pooled: mean={all_cos_arr.mean():.4f} median={median_cos:.4f} "
          f"Q1={np.percentile(all_cos_arr,25):.4f} Q3={np.percentile(all_cos_arr,75):.4f}")
    print(f"무작위 기준선: mean={random_baseline_arr.mean():.4f} median={np.median(random_baseline_arr):.4f}")
    print(f"\n판정 게이트 1: {gate1_verdict}")
    print("\n파라미터별 cos(J_fd, J_surrogate):")
    for k, v in comparison_by_param.items():
        print(f"  {k:<30} median={v['cos_median']:.4f}  mag_ratio={v.get('magnitude_ratio_median', float('nan')):.3f}")
    print("\n게이트 재검정 (h_selected, 원시 점):")
    for p in GATE_TARGET_PARAMS:
        print(f"  {p:<12} rho={gate_spearman_by_param[p]['spearman_rho']:.4f} "
              f"(p={gate_spearman_by_param[p]['spearman_pvalue']:.4g})  "
              f"slope={gate_regression_by_param[p]['slope']:.4f} R²={gate_regression_by_param[p]['r2']:.4f}")

    print(f"\n완료: {results5_path}")
    print(f"그림: {out_dir}/fd_h_sensitivity.png, {out_dir}/fd_vs_surrogate.png, {out_dir}/fd_gate.png")
    print("★ 여기까지 하고 멈춥니다. 판정 게이트 1을 확인한 뒤 다음 단계를 결정하세요.")


if __name__ == "__main__":
    main()
