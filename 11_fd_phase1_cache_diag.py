"""CLAP FX Probe — 11_fd_phase1_cache_diag.py (5차 후속: 캐시 생성 + 피크 정규화 진단)

Phase 1(10_fd_phase1.py)은 파라미터별 집계 통계만 results_5.json에 남기고 점별
512차원 J_fd 벡터를 저장하지 않았다. freeze_mode 층화 게이트 재검정(과제 1),
부분공간 투영(2-B) 등 후속 분석은 점별 데이터가 있어야 하므로, Phase 1과 완전히
동일한 (src_id, θ) 200점·h=0.02에서 다시 렌더링해 이번엔 캐시(out/phase1_fd_cache.npz)로
남긴다. 새 표본을 뽑지 않으므로 "새 실험"이 아니라 "누락된 캐시 재구성"이다.

★ 이 재렌더링에 얹어 더 급한 진단을 먼저 한다: Phase 1의 jacobian_norm_by_param에서
  reverb.wet_level의 median이 0.0으로 나왔다 — wet_level을 흔들어도 임베딩이 거의
  안 움직인다는 뜻이고, 사실이면 게이트 가설(‖∂f/∂damping‖ ∝ wet_level) 자체가
  성립할 토대가 없다. 유력 용의자는 render()의 클리핑 방지 단계다:

    if peak > 1.0: wet = wet * (0.99 / peak)

  이건 "0.7로 정규화"가 아니라 peak>1일 때만 발동하는 비례 재조정이지만, reverb가
  거의 항상 peak>1을 만든다면 사실상 매번 발동해 wet_level이 커질수록 커지는 분자를
  같이 커지는 peak가 상쇄해버릴 수 있다. 조건 (A)=현재 파이프라인 그대로, 조건
  (B)=이 비례 재조정 대신 진짜 하드클리핑(np.clip(-1,1))으로 바꿔 비교한다 — 하드클리핑은
  범위를 벗어난 개별 샘플만 자르지 개러 전체 게인을 재조정하지 않으므로, wet_level이
  실제로 만드는 게인 변화를 보존한다.

★ 재렌더링 없음(이 스크립트 실행 이후): out/phase1_fd_cache.npz,
  out/phase1_fd_theta_cache.npz 두 캐시만 있으면 과제 1~4 후속 분석은 전부 이 파일들만
  읽는다.

★ 여기서는 진단만 하고 멈춘다. 과제 1(freeze_mode 층화 게이트) 판정은 이 진단 결과를
  본 뒤 확정한다 — 피크 정규화가 원인이면 조건 (A)의 게이트 검정 자체가 무의미하다.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
from huggingface_hub import hf_hub_download
from pedalboard import Distortion, HighShelfFilter, Pedalboard, Reverb
from scipy.stats import qmc, wilcoxon
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
COLORS = {"reverb": "#2a78d6", "distortion": "#eb6834", "highshelf": "#1baf7a", "A": "#2a78d6", "B": "#e34948"}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def dejavu_yticks(ax):
    for label in ax.get_yticklabels():
        label.set_fontfamily("DejaVu Sans")


# ---------------------------------------------------------------------------
# 렌더링/전처리 상수 — 01_embed.py / 10_fd_phase1.py와 동일
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48000
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
PEAK_TARGET = 0.7
SILENCE_PEAK_THRESHOLD = 1e-4

CLAP_REPO_ID = "lukewys/laion_clap"
CLAP_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"

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

NULL_AXIS_PARAM = "ultrasonic_gain_db"
ULTRASONIC_CUTOFF_HZ = 12000.0
ULTRASONIC_Q = 0.7071067811865476
PARAM_SPACE["highshelf"][NULL_AXIS_PARAM] = {"range": (-9.0, 9.0), "scale": "linear", "is_null_axis": True}
PARAM_ORDER["highshelf"] = PARAM_ORDER["highshelf"] + [NULL_AXIS_PARAM]

CONTINUOUS_PARAMS = []
for _e in EFFECTS:
    for _p in PARAM_ORDER[_e]:
        if PARAM_SPACE[_e][_p]["scale"] != "bernoulli":
            CONTINUOUS_PARAMS.append((_e, _p))

REVERB_CONTINUOUS = [p for p in PARAM_ORDER["reverb"] if PARAM_SPACE["reverb"][p]["scale"] != "bernoulli"]


def to_raw(effect, param, u):
    spec = PARAM_SPACE[effect][param]
    lo, hi = spec["range"]
    u = float(np.clip(u, 0.0, 1.0))
    if spec["scale"] == "log":
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    return float(lo + u * (hi - lo))


def to_norm(effect, param, raw):
    spec = PARAM_SPACE[effect][param]
    lo, hi = spec["range"]
    if spec["scale"] == "log":
        return float((np.log(raw) - np.log(lo)) / (np.log(hi) - np.log(lo)))
    return float((raw - lo) / (hi - lo))


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


def render_raw(y, effect_name, theta_raw):
    """이펙트 체인만 통과시킨 원시 출력 — 후처리(peak 재조정 vs 하드클리핑) 이전.
    조건 A/B는 같은 theta에 대해 이 원시 출력을 공유하므로, pedalboard 렌더링을
    반으로 줄이려면 이 함수를 조건당 한 번씩이 아니라 theta당 한 번만 불러야 한다."""
    if effect_name == "reverb":
        board = Pedalboard([Reverb(
            room_size=theta_raw["room_size"], damping=theta_raw["damping"],
            wet_level=theta_raw["wet_level"], dry_level=1.0,
            width=theta_raw["width"], freeze_mode=theta_raw["freeze_mode"],
        )])
    elif effect_name == "distortion":
        board = Pedalboard([Distortion(drive_db=theta_raw["drive_db"])])
    elif effect_name == "highshelf":
        board = Pedalboard([
            HighShelfFilter(cutoff_frequency_hz=theta_raw["cutoff_frequency_hz"], gain_db=theta_raw["gain_db"], q=theta_raw["q"]),
            HighShelfFilter(cutoff_frequency_hz=ULTRASONIC_CUTOFF_HZ, gain_db=theta_raw[NULL_AXIS_PARAM], q=ULTRASONIC_Q),
        ])
    else:
        raise ValueError(effect_name)
    return board(y, SAMPLE_RATE)


def postprocess_A(wet_raw):
    """Phase 1과 완전히 동일 — peak>1이면 0.99/peak로 비례 재조정."""
    peak = float(np.abs(wet_raw).max())
    scale_factor = 1.0
    wet = wet_raw
    if peak > 1.0:
        scale_factor = 0.99 / peak
        wet = wet_raw * scale_factor
    return wet.astype(np.float32), peak, scale_factor


def postprocess_B(wet_raw):
    """peak>1 비례 재조정 대신 하드클리핑([-1,1]) — 전체 게인은 건드리지 않는다."""
    peak = float(np.abs(wet_raw).max())
    clipped = peak > 1.0
    wet = np.clip(wet_raw, -1.0, 1.0)
    return wet.astype(np.float32), clipped


# ---------------------------------------------------------------------------
# CLAP
# ---------------------------------------------------------------------------
def download_clap_checkpoint(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / CLAP_FILENAME
    if not ckpt_path.exists():
        hf_hub_download(repo_id=CLAP_REPO_ID, filename=CLAP_FILENAME, local_dir=cache_dir)
    return ckpt_path


def load_clap(device, cache_dir: Path):
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


def embed_batch(clap, device, batch):
    tensor = torch.tensor(np.stack(batch), dtype=torch.float32, device=device)
    with torch.no_grad():
        emb = clap.get_audio_embedding_from_data(tensor, use_tensor=True)
    return emb.cpu().numpy()


def stratified_sample_by_family(src_ids: list, family_by_src: dict, n_target: int, seed: int):
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
            break
    return selected


def plan_sides(u0, h):
    can_plus = (u0 + h) <= 1.0
    can_minus = (u0 - h) >= 0.0
    if can_plus and can_minus:
        return "central"
    elif can_plus:
        return "fwd"
    elif can_minus:
        return "bwd"
    return "central"


def main():
    parser = argparse.ArgumentParser(description="5차 후속 — 캐시 생성 + 피크 정규화 진단 (재렌더링은 이번 1회, Phase 1과 동일 200점)")
    parser.add_argument("--audio-dir", type=str, default="nsynth-test/audio")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--n-theta-dep-sources", type=int, default=50)
    parser.add_argument("--n-theta-per-source", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--limit-points", type=int, default=None, help="테스트용: Phase 1 200점 중 앞 N개만 사용")
    args = parser.parse_args()

    if args.device == "mps":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    out_dir = Path(args.out)
    audio_dir = Path(args.audio_dir)
    emb_path = Path(args.embeddings)

    results5_path = out_dir / "results_5.json"
    with open(results5_path) as f:
        results5 = json.load(f)
    assert "phase1" in results5, "phase1 섹션이 없습니다 — 10_fd_phase1.py를 먼저 실행하세요."

    h = results5["meta"]["phase1_h_selected"]
    selected_src_ids = results5["meta"]["phase1_selected_src_ids"]
    theta_by_effect = results5["meta"]["phase1_theta_norm_by_effect"]  # {effect: [dict,...]} len 200, Phase1과 순서 동일
    if args.limit_points is not None:
        selected_src_ids = selected_src_ids[: args.limit_points]
        theta_by_effect = {e: v[: args.limit_points] for e, v in theta_by_effect.items()}
    n_points = len(selected_src_ids)
    print(f"Phase 1과 동일한 {n_points}점, h={h} 재사용 (새 샘플링 없음)")

    d = np.load(emb_path, allow_pickle=False)
    dry_mask = d["effect"] == "dry"
    family_by_src = dict(zip(d["src_id"][dry_mask].tolist(), d["instrument_family"][dry_mask].tolist()))
    filename_by_src = dict(zip(d["src_id"][dry_mask].tolist(), d["filename"][dry_mask].tolist()))

    device = torch.device(args.device)
    print("CLAP 모델 로딩 중...")
    clap = load_clap(device, Path(__file__).parent / "ckpts")

    print("소스 오디오 로딩 중 (Phase 1과 동일 200개)...")
    y_by_idx = {}
    for i, src in enumerate(tqdm(selected_src_ids, desc="오디오 로딩")):
        y = load_and_preprocess(audio_dir / filename_by_src[src])
        assert y is not None
        y_by_idx[i] = y

    # =========================================================================
    # 캐시 1: Phase 1과 동일 200점 — 조건 A(전체 파라미터) + 조건 B(reverb만)
    #
    # reverb는 theta가 같으면 A/B가 같은 원시 렌더(render_raw)를 공유한다 — 후처리
    # (비례 재조정 vs 하드클리핑)만 다르다. pedalboard 호출을 반으로 줄이기 위해
    # reverb 태그는 한 번만 렌더링하고 postprocess_A/B를 둘 다 적용한다.
    # =========================================================================
    raw_jobs = []  # (point_idx, effect, tag, theta_raw) — effect별 1회만 렌더
    plan_A = {}  # (point_idx, effect, param) -> mode
    for i in range(n_points):
        theta_c_raw = {e: {p: to_raw(e, p, theta_by_effect[e][i][p]) for p in PARAM_ORDER[e]} for e in EFFECTS}

        for e in EFFECTS:
            raw_jobs.append((i, e, "center", dict(theta_c_raw[e])))
            for p in PARAM_ORDER[e]:
                if PARAM_SPACE[e][p]["scale"] == "bernoulli":
                    continue
                u0 = theta_by_effect[e][i][p]
                mode = plan_sides(u0, h)
                plan_A[(i, e, p)] = mode
                if mode in ("central", "fwd"):
                    th = dict(theta_c_raw[e]); th[p] = to_raw(e, p, u0 + h)
                    raw_jobs.append((i, e, f"{p}+", th))
                if mode in ("central", "bwd"):
                    th = dict(theta_c_raw[e]); th[p] = to_raw(e, p, u0 - h)
                    raw_jobs.append((i, e, f"{p}-", th))
            if "freeze_mode" in PARAM_ORDER[e]:
                th = dict(theta_c_raw[e]); th["freeze_mode"] = 1.0 - theta_c_raw[e]["freeze_mode"]
                raw_jobs.append((i, e, "freeze_alt", th))

    print(f"캐시 1 렌더링 작업 {len(raw_jobs)}개 (reverb는 A/B 공유 — pedalboard 호출 1회, 임베딩만 2회)...")
    emb_A, emb_B = {}, {}
    peak_A_center = {}  # point_idx -> (peak, scale_factor) at reverb center
    clipped_B = {i: False for i in range(n_points)}
    batch_audio, batch_keys = [], []

    def flush():
        if not batch_audio:
            return
        out = embed_batch(clap, device, batch_audio)
        for k, e in zip(batch_keys, out):
            i, cond, effect, tag = k
            if cond == "A":
                emb_A[(i, effect, tag)] = e
            else:
                emb_B[(i, effect, tag)] = e
        batch_audio.clear(); batch_keys.clear()

    for i, effect, tag, theta_raw in tqdm(raw_jobs, desc="캐시1 렌더링+임베딩"):
        wet_raw = render_raw(y_by_idx[i], effect, theta_raw)
        wet_A, peak, scale = postprocess_A(wet_raw)
        if effect == "reverb" and tag == "center":
            peak_A_center[i] = (peak, scale)
        batch_audio.append(wet_A)
        batch_keys.append((i, "A", effect, tag))
        if len(batch_audio) >= args.batch_size:
            flush()

        if effect == "reverb":
            wet_B, clipped = postprocess_B(wet_raw)
            if clipped:
                clipped_B[i] = True
            batch_audio.append(wet_B)
            batch_keys.append((i, "B", effect, tag))
            if len(batch_audio) >= args.batch_size:
                flush()
    flush()

    def central_diff(emb_dict, i, effect, p, mode, h):
        if mode == "central":
            return (emb_dict[(i, effect, f"{p}+")] - emb_dict[(i, effect, f"{p}-")]) / (2 * h)
        elif mode == "fwd":
            return (emb_dict[(i, effect, f"{p}+")] - emb_dict[(i, effect, "center")]) / h
        else:
            return (emb_dict[(i, effect, "center")] - emb_dict[(i, effect, f"{p}-")]) / h

    jac_A = {f"{e}.{p}": np.zeros((n_points, 512), dtype=np.float32) for e, p in CONTINUOUS_PARAMS}
    jac_A["reverb.freeze_mode"] = np.zeros((n_points, 512), dtype=np.float32)
    onesided_mask = {f"{e}.{p}": np.zeros(n_points, dtype=bool) for e, p in CONTINUOUS_PARAMS}
    for i in range(n_points):
        for e, p in CONTINUOUS_PARAMS:
            mode = plan_A[(i, e, p)]
            jac_A[f"{e}.{p}"][i] = central_diff(emb_A, i, e, p, mode, h)
            onesided_mask[f"{e}.{p}"][i] = mode != "central"
        cf = theta_by_effect["reverb"][i]["freeze_mode"]
        ec, ealt = emb_A[(i, "reverb", "center")], emb_A[(i, "reverb", "freeze_alt")]
        jac_A["reverb.freeze_mode"][i] = (ealt - ec) if cf == 0.0 else (ec - ealt)

    jac_B = {f"reverb.{p}": np.zeros((n_points, 512), dtype=np.float32) for p in REVERB_CONTINUOUS}
    jac_B["reverb.freeze_mode"] = np.zeros((n_points, 512), dtype=np.float32)
    for i in range(n_points):
        for p in REVERB_CONTINUOUS:
            mode = plan_A[(i, "reverb", p)]
            jac_B[f"reverb.{p}"][i] = central_diff(emb_B, i, "reverb", p, mode, h)
        cf = theta_by_effect["reverb"][i]["freeze_mode"]
        ec, ealt = emb_B[(i, "reverb", "center")], emb_B[(i, "reverb", "freeze_alt")]
        jac_B["reverb.freeze_mode"][i] = (ealt - ec) if cf == 0.0 else (ec - ealt)

    peak_scale_factor = np.array([peak_A_center[i][1] for i in range(n_points)])
    peak_raw = np.array([peak_A_center[i][0] for i in range(n_points)])
    clipping_occurred_B = np.array([clipped_B[i] for i in range(n_points)])
    wet_level_arr = np.array([theta_by_effect["reverb"][i]["wet_level"] for i in range(n_points)])
    family_arr = np.array([family_by_src[s] for s in selected_src_ids])

    # npz 저장 (문자열 키는 numpy object 배열로, 로딩부에서 그대로 dict 복원 가능)
    save_dict = {"h_used": h, "src_id": np.array(selected_src_ids), "instrument_family": family_arr,
                 "peak_scale_factor": peak_scale_factor, "peak_raw": peak_raw,
                 "clipping_occurred_B": clipping_occurred_B, "wet_level": wet_level_arr}
    for k, v in jac_A.items():
        save_dict[f"jacA__{k}"] = v
    for k, v in jac_B.items():
        save_dict[f"jacB__{k}"] = v
    for k, v in onesided_mask.items():
        save_dict[f"onesided__{k}"] = v
    for e in EFFECTS:
        for p in PARAM_ORDER[e]:
            save_dict[f"theta_norm__{e}.{p}"] = np.array([theta_by_effect[e][i][p] for i in range(n_points)])
    np.savez(out_dir / "phase1_fd_cache.npz", **save_dict)
    print(f"저장: {out_dir / 'phase1_fd_cache.npz'}")

    # =========================================================================
    # 캐시 2: 2-C용 — 50소스 x θ4개 = 200점, freeze_mode=0 고정, 별도 집합
    # =========================================================================
    print("\n2-C용 θ-의존성 점 집합 생성 중 (50소스 x 4θ, freeze=0 고정)...")
    unique_srcs = np.unique(d["src_id"][dry_mask])
    rng_split = np.random.RandomState(args.seed)
    shuffled = rng_split.permutation(unique_srcs)
    n_test = max(1, int(round(len(shuffled) * 0.3)))
    test_srcs = set(shuffled[:n_test].tolist())
    td_sources = stratified_sample_by_family(sorted(test_srcs), family_by_src, args.n_theta_dep_sources, args.seed + 777)
    n_td_sources = len(td_sources)
    n_theta_per_source = args.n_theta_per_source

    y_by_td_idx = {}
    for si, src in enumerate(td_sources):
        y = load_and_preprocess(audio_dir / filename_by_src[src])
        assert y is not None
        y_by_td_idx[si] = y

    reverb_dim = len(REVERB_CONTINUOUS)  # freeze=0 고정이라 4차원만
    distortion_dim = 1
    highshelf_dim = len(PARAM_ORDER["highshelf"])  # 4 (널 축 포함)

    td_theta = {"reverb": [], "distortion": [], "highshelf": [], "src_idx": []}
    for si in range(n_td_sources):
        sampler_r = qmc.LatinHypercube(d=reverb_dim, seed=np.random.default_rng([args.seed, 111, si]))
        sampler_d = qmc.LatinHypercube(d=distortion_dim, seed=np.random.default_rng([args.seed, 222, si]))
        sampler_h = qmc.LatinHypercube(d=highshelf_dim, seed=np.random.default_rng([args.seed, 333, si]))
        ur = sampler_r.random(n=n_theta_per_source)
        ud = sampler_d.random(n=n_theta_per_source)
        uh = sampler_h.random(n=n_theta_per_source)
        for k in range(n_theta_per_source):
            rtheta = {p: float(ur[k, j]) for j, p in enumerate(REVERB_CONTINUOUS)}
            rtheta["freeze_mode"] = 0.0
            dtheta = {"drive_db": float(ud[k, 0])}
            htheta = {p: float(uh[k, j]) for j, p in enumerate(PARAM_ORDER["highshelf"])}
            td_theta["reverb"].append(rtheta)
            td_theta["distortion"].append(dtheta)
            td_theta["highshelf"].append(htheta)
            td_theta["src_idx"].append(si)

    n_td_points = len(td_theta["src_idx"])
    print(f"θ-의존성 점 {n_td_points}개 (소스 {n_td_sources}개 x θ{n_theta_per_source}개)")

    td_jobs = []
    td_plan = {}
    for i in range(n_td_points):
        si = td_theta["src_idx"][i]
        theta_c_raw = {
            "reverb": {p: to_raw("reverb", p, td_theta["reverb"][i][p]) for p in PARAM_ORDER["reverb"]},
            "distortion": {p: to_raw("distortion", p, td_theta["distortion"][i][p]) for p in PARAM_ORDER["distortion"]},
            "highshelf": {p: to_raw("highshelf", p, td_theta["highshelf"][i][p]) for p in PARAM_ORDER["highshelf"]},
        }
        for e in EFFECTS:
            td_jobs.append((i, si, e, "center", dict(theta_c_raw[e])))
            for p in PARAM_ORDER[e]:
                if PARAM_SPACE[e][p]["scale"] == "bernoulli":
                    continue
                u0 = td_theta[e][i][p]
                mode = plan_sides(u0, h)
                td_plan[(i, e, p)] = mode
                if mode in ("central", "fwd"):
                    th = dict(theta_c_raw[e]); th[p] = to_raw(e, p, u0 + h)
                    td_jobs.append((i, si, e, f"{p}+", th))
                if mode in ("central", "bwd"):
                    th = dict(theta_c_raw[e]); th[p] = to_raw(e, p, u0 - h)
                    td_jobs.append((i, si, e, f"{p}-", th))
            # freeze_mode=0 고정이므로 freeze diff 불필요

    print(f"캐시 2 렌더링 작업 {len(td_jobs)}개...")
    emb_td = {}
    batch_audio, batch_keys = [], []

    def flush_td():
        if not batch_audio:
            return
        out = embed_batch(clap, device, batch_audio)
        for k, e in zip(batch_keys, out):
            emb_td[k] = e
        batch_audio.clear(); batch_keys.clear()

    for i, si, effect, tag, theta_raw in tqdm(td_jobs, desc="캐시2 렌더링+임베딩"):
        wet, _peak, _scale = postprocess_A(render_raw(y_by_td_idx[si], effect, theta_raw))
        batch_audio.append(wet)
        batch_keys.append((i, effect, tag))
        if len(batch_audio) >= args.batch_size:
            flush_td()
    flush_td()

    jac_td = {f"{e}.{p}": np.zeros((n_td_points, 512), dtype=np.float32) for e, p in CONTINUOUS_PARAMS if e != "reverb" or p != "freeze_mode"}
    # reverb 연속 파라미터만 (freeze=0 고정)
    jac_td = {f"{e}.{p}": np.zeros((n_td_points, 512), dtype=np.float32)
              for e in EFFECTS for p in PARAM_ORDER[e] if PARAM_SPACE[e][p]["scale"] != "bernoulli"}
    for i in range(n_td_points):
        for e in EFFECTS:
            for p in PARAM_ORDER[e]:
                if PARAM_SPACE[e][p]["scale"] == "bernoulli":
                    continue
                mode = td_plan[(i, e, p)]
                jac_td[f"{e}.{p}"][i] = central_diff(emb_td, i, e, p, mode, h)

    td_save = {"h_used": h, "src_idx": np.array(td_theta["src_idx"]),
               "td_source_ids": np.array(td_sources), "n_theta_per_source": n_theta_per_source,
               "instrument_family": np.array([family_by_src[td_sources[si]] for si in td_theta["src_idx"]])}
    for k, v in jac_td.items():
        td_save[f"jac__{k}"] = v
    for e in EFFECTS:
        for p in PARAM_ORDER[e]:
            if PARAM_SPACE[e][p]["scale"] == "bernoulli":
                continue
            td_save[f"theta_norm__{e}.{p}"] = np.array([td_theta[e][i][p] for i in range(n_td_points)])
    np.savez(out_dir / "phase1_fd_theta_cache.npz", **td_save)
    print(f"저장: {out_dir / 'phase1_fd_theta_cache.npz'}")

    # =========================================================================
    # 피크 정규화 진단
    # =========================================================================
    print("\n피크 정규화 진단 계산 중...")
    diag = {}
    for p in REVERB_CONTINUOUS:
        na = np.linalg.norm(jac_A[f"reverb.{p}"], axis=1)
        nb = np.linalg.norm(jac_B[f"reverb.{p}"], axis=1)
        cos_ab = np.array([
            float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            for a, b in zip(jac_A[f"reverb.{p}"], jac_B[f"reverb.{p}"])
        ])
        try:
            wstat, wp = wilcoxon(na, nb)
        except ValueError:
            wstat, wp = None, None
        diag[p] = {
            "A_mean": float(na.mean()), "A_median": float(np.median(na)), "A_std": float(na.std()),
            "B_mean": float(nb.mean()), "B_median": float(np.median(nb)), "B_std": float(nb.std()),
            "cos_A_vs_B_mean": float(cos_ab.mean()), "cos_A_vs_B_median": float(np.median(cos_ab)),
            "wilcoxon_stat": (float(wstat) if wstat is not None else None),
            "wilcoxon_pvalue": (float(wp) if wp is not None else None),
        }
    # freeze diff도 참고용으로 포함
    na = np.linalg.norm(jac_A["reverb.freeze_mode"], axis=1)
    nb = np.linalg.norm(jac_B["reverb.freeze_mode"], axis=1)
    diag["freeze_mode"] = {"A_mean": float(na.mean()), "A_median": float(np.median(na)),
                            "B_mean": float(nb.mean()), "B_median": float(np.median(nb)),
                            "note": "0->1 차분, 미분 아님"}

    n_clipped_B = int(clipping_occurred_B.sum())
    n_scaled_A = int((peak_scale_factor < 0.999).sum())

    # ---- 그림 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    ax = axes[0]
    x = np.arange(len(REVERB_CONTINUOUS))
    width = 0.35
    a_means = [diag[p]["A_mean"] for p in REVERB_CONTINUOUS]
    b_means = [diag[p]["B_mean"] for p in REVERB_CONTINUOUS]
    ax.bar(x - width / 2, a_means, width, label="(A) 비례 재조정 (현재)", color=COLORS["A"], zorder=3)
    ax.bar(x + width / 2, b_means, width, label="(B) 하드클리핑", color=COLORS["B"], zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(REVERB_CONTINUOUS, rotation=30, ha="right")
    ax.set_ylabel("‖J_fd‖ mean")
    ax.set_title("조건 A vs B — 축별 평균 노름")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    ax = axes[1]
    ax.scatter(wet_level_arr, peak_scale_factor, s=10, alpha=0.5, color=COLORS["A"])
    ax.set_xlabel("wet_level (정규화)")
    ax.set_ylabel("A조건 재조정 계수 (0.99/peak, peak≤1이면 1.0)")
    ax.set_title(f"재조정 발동 비율: {n_scaled_A}/{n_points} ({100*n_scaled_A/n_points:.0f}%)")
    style_axis(ax)

    ax = axes[2]
    cos_medians = [diag[p]["cos_A_vs_B_median"] for p in REVERB_CONTINUOUS]
    ax.bar(np.arange(len(REVERB_CONTINUOUS)), cos_medians, color="#898781", zorder=3)
    ax.set_xticks(np.arange(len(REVERB_CONTINUOUS))); ax.set_xticklabels(REVERB_CONTINUOUS, rotation=30, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("cos(J_fd_A, J_fd_B)  median")
    ax.set_title(f"방향 변화 (B조건 클리핑 발생 {n_clipped_B}/{n_points})")
    style_axis(ax)

    fig.suptitle(f"피크 정규화 진단 (h={h}, n={n_points}, Phase 1과 동일 200점)")
    fig.tight_layout()
    fig.savefig(out_dir / "peak_norm_diagnostic.png")
    plt.close(fig)

    # ---- results_5.json에 followup 섹션 추가 (phase1은 건드리지 않음) ----
    results5.setdefault("results_5_followup", {})
    results5["results_5_followup"]["peak_normalization_diagnostic"] = {
        "meta": {
            "h_used": h, "n_points": n_points,
            "n_scaled_A": n_scaled_A, "n_scaled_A_ratio": n_scaled_A / n_points,
            "n_clipped_B": n_clipped_B, "n_clipped_B_ratio": n_clipped_B / n_points,
            "n_theta_dep_points": n_td_points, "n_theta_dep_sources": n_td_sources,
        },
        "by_param": diag,
    }
    with open(results5_path, "w") as f:
        json.dump(results5, f, indent=2, ensure_ascii=False)

    print("\n=== 피크 정규화 진단 결과 ===")
    print(f"{'param':<12}{'A_mean':>10}{'A_median':>10}{'B_mean':>10}{'B_median':>10}{'cos(A,B)':>10}{'wilcoxon_p':>12}")
    for p in REVERB_CONTINUOUS:
        v = diag[p]
        print(f"{p:<12}{v['A_mean']:>10.4f}{v['A_median']:>10.4f}{v['B_mean']:>10.4f}{v['B_median']:>10.4f}"
              f"{v['cos_A_vs_B_median']:>10.4f}{(v['wilcoxon_pvalue'] if v['wilcoxon_pvalue'] is not None else float('nan')):>12.4g}")
    print(f"\nA조건 재조정 발동: {n_scaled_A}/{n_points} ({100*n_scaled_A/n_points:.1f}%)")
    print(f"B조건 클리핑 발생: {n_clipped_B}/{n_points} ({100*n_clipped_B/n_points:.1f}%)")
    print(f"\n저장: {out_dir / 'phase1_fd_cache.npz'}, {out_dir / 'phase1_fd_theta_cache.npz'}")
    print(f"그림: {out_dir / 'peak_norm_diagnostic.png'}")
    print("★ 여기서 멈춥니다. 피크 정규화 진단 결과를 판정한 뒤 과제 1~4 진행 여부를 결정하세요.")


if __name__ == "__main__":
    main()
