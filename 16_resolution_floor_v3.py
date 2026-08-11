"""CLAP FX Probe — 16_resolution_floor_v3.py (6차 후속 과제 4-R: N/조건 불일치 해결 + 재판정)

과제 4(15_resolution_floor.py)의 문제: 새 바닥(-0.00172)은 조건 C·N=500(highshelf 센터
임베딩만)에서 나왔는데, 재분류 대상(reverb/distortion/damping/width 등)은 조건 A·
N=6,400~25,600(3·4차/과제 1)이었다. 널 축의 held-out R²는 N이 작을수록 더 음수로
쏠리므로, 작은 N의 바닥을 큰 N 값에 적용하면 기준이 과도하게 느슨해진다 — 실제로 이전
재판정이 전부 "통과"로 쏠렸다.

이번 판은 초음파 축 설계(전이대역 누출 없음, 12k/15k 차이 0.003)는 그대로 인정하고
"적용 방식"만 고친다.

  4-R1: 500 기준점(과제 2 캐시와 완전히 동일한 θ)을 조건 C로 "한 번에" 렌더링한다 —
        reverb → distortion → highshelf(main+12k+15k)를 하나의 Pedalboard 체인으로
        묶어 점당 임베딩 1개(e(θ), 512차원)를 낸다. (과제 2는 이펙트별로 따로
        렌더링해 J_fd를 냈다 — 그건 미분 목적이라 맞는 설계였고, 이번엔 10축 전체를
        "같은 N, 같은 조건, 같은 절차"로 프로브하는 게 목적이라 체인을 합친다.)
  4-R2: 이 임베딩 500개 → θ 10축 전체를 다변량 Ridge로 한 번에 프로브한다. 실제 축
        8개와 초음파 통제 축 2개가 정확히 같은 N(500)·같은 조건(C)·같은 절차(다변량
        Ridge, source-level GroupShuffleSplit, source-level 부트스트랩 CI)로 나온다.
  4-R3: 스칼라 바닥(참고용) 대신 CI 중첩 검정을 주 판정으로 쓴다 — 초음파 두 축의
        부트스트랩 표본을 하나의 널 분포로 합치고, 각 실제 축의 CI가 이 널 분포의
        CI와 겹치는지로 "신호 있음 / 측정 불가"를 가른다.
  4-R4: 3·4차/과제 1(조건 A) 값은 N을 500으로 맞춰 재추정한다(src_id 단위 100회
        서브샘플). 조건은 여전히 다르므로(A vs C) 그 사실을 명시하고 참고용으로만
        조건 C 널과 비교한다.
  4-R5: 4-R2/4-R3(조건·N 완전 일치)를 주 근거로 재판정한다.

결과 해석은 이 스크립트가 단정하지 않는다. README 6차 후속 절의 판정 기준표를 따를 것.
"""
import argparse
import json
import time
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
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
COLORS = {"reverb": "#2a78d6", "distortion": "#eb6834", "highshelf": "#1baf7a", "null": "#e34948", "baseline": "#898781"}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


SAMPLE_RATE = 48000
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
SILENCE_PEAK_THRESHOLD = 1e-4

CLAP_REPO_ID = "lukewys/laion_clap"
CLAP_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"

ULTRASONIC_12K_HZ = 12000.0
ULTRASONIC_15K_HZ = 15000.0
ULTRASONIC_Q = 0.7071067811865476

AXES_C = [
    ("reverb", "wet_level"), ("reverb", "room_size"), ("reverb", "damping"), ("reverb", "width"),
    ("distortion", "drive_db"),
    ("highshelf", "gain_db"), ("highshelf", "cutoff_frequency_hz"), ("highshelf", "q"),
    ("highshelf", "ultrasonic_12k_gain_db"), ("highshelf", "ultrasonic_15k_gain_db"),
]
NULL_AXES = ["highshelf.ultrasonic_12k_gain_db", "highshelf.ultrasonic_15k_gain_db"]


def render_combined(y, theta_raw):
    """reverb -> distortion -> highshelf(main+12k+15k) 단일 체인. 4-R1 전용 —
    과제 2(13_fd_phase3_render.py)의 이펙트별 독립 렌더링과는 목적이 다르다."""
    board = Pedalboard([
        Reverb(room_size=theta_raw["room_size"], damping=theta_raw["damping"],
               wet_level=theta_raw["wet_level"], dry_level=1.0, width=theta_raw["width"],
               freeze_mode=0.0),
        Distortion(drive_db=theta_raw["drive_db"]),
        HighShelfFilter(cutoff_frequency_hz=theta_raw["cutoff_frequency_hz"],
                         gain_db=theta_raw["gain_db"], q=theta_raw["q"]),
        HighShelfFilter(cutoff_frequency_hz=ULTRASONIC_12K_HZ,
                         gain_db=theta_raw["ultrasonic_12k_gain_db"], q=ULTRASONIC_Q),
        HighShelfFilter(cutoff_frequency_hz=ULTRASONIC_15K_HZ,
                         gain_db=theta_raw["ultrasonic_15k_gain_db"], q=ULTRASONIC_Q),
    ])
    return board(y, SAMPLE_RATE)


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
# 프로브 방법론 (04_probe.py/12_freeze_probe.py와 동일) — bootstrap은 원시 점수도 반환
# ---------------------------------------------------------------------------
def held_out_r2_multi(X, Y, groups, seed, n_splits=5, test_size=0.3):
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    scores = []
    for train_idx, test_idx in gss.split(X, Y, groups):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], Y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(r2_score(Y[test_idx], pred, multioutput="raw_values"))
    scores = np.array(scores)
    return scores.mean(axis=0), scores.std(axis=0)


def bootstrap_r2_multi_raw(X, Y, groups, seed, n_boot=1000):
    unique_srcs = np.unique(groups)
    n = len(unique_srcs)
    src_to_rows = {s: np.where(groups == s)[0] for s in unique_srcs}
    rng = np.random.RandomState(seed)
    all_scores = []
    for _ in range(n_boot):
        boot_srcs = rng.choice(unique_srcs, size=n, replace=True)
        oob_srcs = np.setdiff1d(unique_srcs, boot_srcs)
        if len(oob_srcs) < 3:
            continue
        train_idx = np.concatenate([src_to_rows[s] for s in boot_srcs])
        test_idx = np.concatenate([src_to_rows[s] for s in oob_srcs])
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], Y[train_idx])
        pred = model.predict(X[test_idx])
        all_scores.append(r2_score(Y[test_idx], pred, multioutput="raw_values"))
    return np.array(all_scores)  # (n_boot_used, n_params)


def ci_from_raw(raw_scores_col, ci=0.95):
    lo = float(np.percentile(raw_scores_col, (1 - ci) / 2 * 100))
    hi = float(np.percentile(raw_scores_col, (1 + ci) / 2 * 100))
    return lo, hi


def intervals_overlap(a_lo, a_hi, b_lo, b_hi):
    return not (a_hi < b_lo or b_hi < a_lo)


def probe_full(X, Y, groups, param_names, seed, n_boot):
    r2_mean, r2_std = held_out_r2_multi(X, Y, groups, seed)
    raw = bootstrap_r2_multi_raw(X, Y, groups, seed, n_boot=n_boot)
    result = {}
    for i, pname in enumerate(param_names):
        lo, hi = ci_from_raw(raw[:, i])
        result[pname] = {
            "probe_r2": float(r2_mean[i]), "probe_r2_std": float(r2_std[i]),
            "probe_r2_ci_low": lo, "probe_r2_ci_high": hi,
            "boot_mean": float(raw[:, i].mean()),
            "n_rows": int(len(Y)), "n_sources": int(len(np.unique(groups))), "n_boot_used": int(raw.shape[0]),
        }
    return result, raw


# ---------------------------------------------------------------------------
# 4-R4: 조건 A(3·4차/과제 1) N=500 매칭 서브샘플
# ---------------------------------------------------------------------------
def subsample_n_matched_repeated(X, Y, groups, param_names, n_reps, n_src_target, n_rows_per_src, seed):
    """src_id 단위로 n_src_target개 소스를 뽑고, 소스당 최대 n_rows_per_src행을 뽑아
    N을 조건 C(100소스x5θ=500)와 최대한 맞춘 뒤, 매 반복마다 held-out R² 점추정(3-fold
    평균)을 낸다. n_reps회 반복한 분포를 반환한다."""
    unique_srcs = np.unique(groups)
    rows_by_src = {s: np.where(groups == s)[0] for s in unique_srcs}
    rng = np.random.RandomState(seed)
    reps = {p: [] for p in param_names}
    n_rows_used = []
    for r in range(n_reps):
        n_src = min(n_src_target, len(unique_srcs))
        chosen_srcs = rng.choice(unique_srcs, size=n_src, replace=False)
        idx_parts = []
        for s in chosen_srcs:
            rows = rows_by_src[s]
            take = min(n_rows_per_src, len(rows))
            idx_parts.append(rng.choice(rows, size=take, replace=False))
        idx = np.concatenate(idx_parts)
        n_rows_used.append(len(idx))
        Xs, Ys, gs = X[idx], Y[idx], groups[idx]
        r2_mean, _ = held_out_r2_multi(Xs, Ys, gs, seed + r, n_splits=3, test_size=0.3)
        for i, p in enumerate(param_names):
            reps[p].append(float(r2_mean[i]))
    summary = {}
    for p in param_names:
        arr = np.array(reps[p])
        summary[p] = {
            "n_reps": n_reps, "mean": float(arr.mean()), "median": float(np.median(arr)),
            "pct2_5": float(np.percentile(arr, 2.5)), "pct97_5": float(np.percentile(arr, 97.5)),
            "n_rows_target": n_src_target * n_rows_per_src, "n_rows_actual_mean": float(np.mean(n_rows_used)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="6차 후속 과제 4-R — N/조건 불일치 해결 + 재판정")
    parser.add_argument("--audio-dir", type=str, default="nsynth-test/audio")
    parser.add_argument("--cache", type=str, default="out/phase3_fd_cache.npz")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--results4", type=str, default="out/results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-reps-subsample", type=int, default=100)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.out)
    audio_dir = Path(args.audio_dir)

    # ---- 4-R1 ----
    print(f"캐시 로딩 중: {args.cache}")
    c = np.load(args.cache, allow_pickle=False)
    theta_cache = c["theta"]
    theta_norm_cache = c["theta_norm"]
    axis_names = [str(x) for x in c["theta_axis_names"]]
    src_id = c["src_id"]
    family = c["instrument_family"]
    filename = c["filename"]
    theta_group_id = c["theta_group_id"]
    peak_target_c = float(c["peak_target_c"])
    n_points = theta_cache.shape[0]
    axis_idx = {ax: k for k, ax in enumerate(axis_names)}
    assert axis_names == [f"{e}.{p}" for e, p in AXES_C], "축 순서가 과제 2 캐시와 다릅니다."

    unique_srcs = sorted(set(int(s) for s in src_id))
    print(f"소스 {len(unique_srcs)}개 오디오 로딩 중...")
    y_dry_C_by_src = {}
    for s in tqdm(unique_srcs, desc="오디오 로딩"):
        fname = str(filename[list(src_id).index(s)])
        r = load_raw(audio_dir / fname)
        if r is None:
            raise RuntimeError(f"src_id={s}가 무음입니다.")
        y_raw, peak = r
        y_dry_C_by_src[s] = (y_raw * (peak_target_c / peak)).astype(np.float32)

    print("통합 체인 렌더링(조건 C, reverb->distortion->highshelf) + 임베딩 중 (500회)...")
    device = torch.device(args.device)
    if args.device == "mps":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    clap = load_clap(device, Path(__file__).parent / "ckpts")

    emb = np.zeros((n_points, 512), dtype=np.float32)
    batch_audio, batch_idx = [], []

    def flush():
        if not batch_audio:
            return
        e = embed_batch(clap, device, batch_audio)
        for k, v in zip(batch_idx, e):
            emb[k] = v
        batch_audio.clear()
        batch_idx.clear()

    for i in tqdm(range(n_points), desc="렌더링+임베딩"):
        s = int(src_id[i])
        theta_raw = {}
        for e_, p in AXES_C:
            theta_raw[p] = float(theta_cache[i, axis_idx[f"{e_}.{p}"]])
        wet = render_combined(y_dry_C_by_src[s], theta_raw)
        batch_audio.append(wet.astype(np.float32))
        batch_idx.append(i)
        if len(batch_audio) >= args.batch_size:
            flush()
    flush()

    base_emb_path = out_dir / "phase3_base_emb.npz"
    np.savez(
        base_emb_path, emb=emb, theta=theta_cache, theta_norm=theta_norm_cache,
        theta_axis_names=np.array(axis_names), src_id=src_id, instrument_family=family,
        theta_group_id=theta_group_id,
    )
    assert np.allclose(theta_cache, np.load(base_emb_path)["theta"]), "저장된 theta가 캐시와 불일치합니다."
    print(f"기준점 임베딩 저장: {base_emb_path} (theta 일치 확인됨)")

    # ---- 4-R2 ----
    print("10축 통합 다변량 Ridge 프로브 계산 중 (조건 C, N=500)...")
    param_names_full = [f"{e}.{p}" for e, p in AXES_C]
    probe_result, raw_scores = probe_full(emb, theta_norm_cache, src_id, param_names_full, args.seed, args.n_boot)

    # ---- 4-R3: CI 중첩 검정 ----
    null_col_idx = [param_names_full.index(k) for k in NULL_AXES]
    null_pool = np.concatenate([raw_scores[:, k] for k in null_col_idx])
    null_lo, null_hi = ci_from_raw(null_pool)
    scalar_floor_reference = max(probe_result[NULL_AXES[0]]["probe_r2_ci_high"], probe_result[NULL_AXES[1]]["probe_r2_ci_high"])

    ci_overlap_verdict = {}
    for k in param_names_full:
        if k in NULL_AXES:
            continue
        lo, hi = probe_result[k]["probe_r2_ci_low"], probe_result[k]["probe_r2_ci_high"]
        overlaps = intervals_overlap(lo, hi, null_lo, null_hi)
        ci_overlap_verdict[k] = {
            "ci_low": lo, "ci_high": hi, "overlaps_null": overlaps,
            "verdict": "널과 구분 안 됨 (측정 불가)" if overlaps else "널과 유의하게 다름 (신호 있음)",
        }

    print(f"널 풀(초음파 12k+15k, n={len(null_pool)}) CI = [{null_lo:.4f}, {null_hi:.4f}]")

    # ---- 4-R4: 조건 A N매칭 서브샘플 ----
    print(f"조건 A(3·4차) N=500 매칭 서브샘플 재산출 중 ({args.n_reps_subsample}회 반복)...")
    d = np.load(args.embeddings, allow_pickle=False)
    embed_config = json.load(open(Path(args.embeddings).parent / "embed_config.json"))
    theta_slots = {e: tuple(v) for e, v in embed_config["theta_slots"].items()}
    param_order = embed_config["param_order"]

    reverb_mask = d["effect"] == "reverb"
    s0, s1 = theta_slots["reverb"]
    reverb_theta = d["theta_norm"][reverb_mask][:, s0:s1]
    freeze_idx = param_order["reverb"].index("freeze_mode")
    freeze = reverb_theta[:, freeze_idx]
    cont_params = [p for p in param_order["reverb"] if p != "freeze_mode"]
    cont_idx = [param_order["reverb"].index(p) for p in cont_params]
    m0 = freeze == 0.0
    X_rev = d["embeddings"][reverb_mask][m0]
    Y_rev = reverb_theta[m0][:, cont_idx]
    g_rev = d["src_id"][reverb_mask][m0]

    def get_effect_full(effect_name):
        mask = d["effect"] == effect_name
        s, e = theta_slots[effect_name]
        return d["embeddings"][mask], d["theta_norm"][mask][:, s:e], d["src_id"][mask]

    X_dist, Y_dist, g_dist = get_effect_full("distortion")
    X_hs, Y_hs, g_hs = get_effect_full("highshelf")

    subsample_rev = subsample_n_matched_repeated(X_rev, Y_rev, g_rev, cont_params, args.n_reps_subsample, 100, 5, args.seed)
    subsample_dist = subsample_n_matched_repeated(X_dist, Y_dist, g_dist, param_order["distortion"], args.n_reps_subsample, 100, 5, args.seed)
    subsample_hs = subsample_n_matched_repeated(X_hs, Y_hs, g_hs, param_order["highshelf"], args.n_reps_subsample, 100, 5, args.seed)

    condition_A_matched = {}
    for p in cont_params:
        condition_A_matched[f"reverb.{p}"] = subsample_rev[p]
    for p in param_order["distortion"]:
        condition_A_matched[f"distortion.{p}"] = subsample_dist[p]
    for p in param_order["highshelf"]:
        condition_A_matched[f"highshelf.{p}"] = subsample_hs[p]

    condition_A_vs_C_null_overlap = {}
    for k, v in condition_A_matched.items():
        overlaps = intervals_overlap(v["pct2_5"], v["pct97_5"], null_lo, null_hi)
        condition_A_vs_C_null_overlap[k] = {
            "overlaps_C_null": overlaps,
            "note": "조건 A(3·4차 원본, range도 4차와 동일) N=500 매칭 분포 vs 조건 C 널 — 조건 자체는 다름, 참고용",
        }

    # 순위 일치 확인 (이펙트 평균)
    def mean_of(d_, keys):
        return float(np.mean([d_[k]["probe_r2" if "probe_r2" in d_[k] else "mean"] for k in keys]))

    order_C = sorted(
        [("reverb", np.mean([probe_result[f"reverb.{p}"]["probe_r2"] for p in ["wet_level", "room_size", "damping", "width"]])),
         ("distortion", probe_result["distortion.drive_db"]["probe_r2"]),
         ("highshelf", np.mean([probe_result[f"highshelf.{p}"]["probe_r2"] for p in ["gain_db", "cutoff_frequency_hz", "q"]]))],
        key=lambda x: -x[1])
    order_A_matched = sorted(
        [("reverb", np.mean([subsample_rev[p]["mean"] for p in cont_params])),
         ("distortion", np.mean([subsample_dist[p]["mean"] for p in param_order["distortion"]])),
         ("highshelf", np.mean([subsample_hs[p]["mean"] for p in param_order["highshelf"]]))],
        key=lambda x: -x[1])
    rank_consistent = [o[0] for o in order_C] == [o[0] for o in order_A_matched]

    # ---- 4-R5: 재판정 ----
    reclass_v2 = {}
    for k in ["reverb.damping", "reverb.width"]:
        v = ci_overlap_verdict[k]
        reclass_v2[k] = {"condition": "C", "N": 500, "probe_r2": probe_result[k]["probe_r2"], **v}
    for k in ["highshelf.cutoff_frequency_hz", "highshelf.q"]:
        v = ci_overlap_verdict[k]
        reclass_v2[k] = {
            "condition": "C", "N": 500, "range_note": "cutoff 500~4000 (4차 원본 500~8000과 다름 — 직접 비교 금지)",
            "probe_r2": probe_result[k]["probe_r2"], **v,
        }
    # 4차 "측정 불가" 2건은 조건·N·range가 전부 달라 직접 재분류 불가 — 재측정 필요로 표시
    old_below_floor_items = {
        "reverb.width (4차, 조건A, N=25600, range 0~1)": {"status": "재측정 필요",
            "reason": "조건·N·range가 조건C(4-R2)와 달라 직접 재분류 불가",
            "reference_condition_A_matched_N500": condition_A_matched["reverb.width"],
            "reference_condition_C_N500_range_current": ci_overlap_verdict["reverb.width"]},
        "highshelf.cutoff_frequency_hz (4차, 조건A, N=12800, range 500~8000)": {"status": "재측정 필요",
            "reason": "range가 조건C(500~4000)와 달라 직접 재분류 불가",
            "reference_condition_A_matched_N500": condition_A_matched["highshelf.cutoff_frequency_hz"],
            "reference_condition_C_N500_range_500_4000": ci_overlap_verdict["highshelf.cutoff_frequency_hz"]},
    }

    # ---- 그림 ----
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)
    labels = param_names_full
    r2s = [probe_result[k]["probe_r2"] for k in labels]
    lo = [probe_result[k]["probe_r2_ci_low"] for k in labels]
    hi = [probe_result[k]["probe_r2_ci_high"] for k in labels]
    x = np.arange(len(labels))
    yerr = np.array([[r - l for r, l in zip(r2s, lo)], [h - r for r, h in zip(r2s, hi)]])
    yerr = np.clip(yerr, 0, None)
    bar_colors = [COLORS["null"] if k in NULL_AXES else COLORS[k.split(".")[0]] for k in labels]
    overlap_hatch = ["//" if (k not in NULL_AXES and ci_overlap_verdict[k]["overlaps_null"]) else None for k in labels]
    bars = ax.bar(x, r2s, yerr=yerr, capsize=3, color=bar_colors, zorder=3)
    for b, h in zip(bars, overlap_hatch):
        if h:
            b.set_hatch(h)
            b.set_edgecolor("black")
    ax.axhspan(null_lo, null_hi, color=COLORS["null"], alpha=0.15, zorder=1, label=f"널 CI(초음파 풀) [{null_lo:.3f},{null_hi:.3f}]")
    ax.axhline(scalar_floor_reference, color="black", linestyle=":", linewidth=1, label=f"참고용 스칼라 바닥={scalar_floor_reference:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Held-out R² (조건 C, N=500, 통합 체인)")
    ax.set_title("10축 통합 프로브 — 빗금=널과 CI 중첩(측정 불가), 빨강=초음파 통제축")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "resolution_floor_v3.png")
    plt.close(fig)

    # ---- 저장 ----
    elapsed = time.time() - t_start
    results6_path = out_dir / "results_6.json"
    with open(results6_path) as f:
        r6 = json.load(f)
    r6.setdefault("meta", {})
    r6["meta"].update({
        "task4r_seed": args.seed, "task4r_elapsed_sec": elapsed, "task4r_n_boot": args.n_boot,
        "task4r_n_reps_subsample": args.n_reps_subsample, "task4r_base_emb_path": str(base_emb_path),
    })
    r6["task4_resolution_floor_v2"] = {
        "method_note": "4-R1~4-R5: reverb->distortion->highshelf 통합 체인, 조건C, N=500, 10축 동일 절차 프로브. "
                        "판정 근거는 CI 중첩(4-R3), 스칼라 바닥은 참고용.",
        "condition_C_probe_N500": probe_result,
        "null_pool": {"columns": NULL_AXES, "n_pooled": int(len(null_pool)), "ci_low": null_lo, "ci_high": null_hi},
        "scalar_floor_reference_not_for_verdict": scalar_floor_reference,
        "ci_overlap_verdict": ci_overlap_verdict,
        "condition_A_n_matched_subsample": condition_A_matched,
        "condition_A_vs_C_null_overlap_reference_only": condition_A_vs_C_null_overlap,
        "effect_order": {
            "condition_C_N500": [o[0] for o in order_C], "condition_C_N500_means": dict(order_C),
            "condition_A_matched_N500": [o[0] for o in order_A_matched], "condition_A_matched_N500_means": dict(order_A_matched),
            "rank_consistent_across_conditions": rank_consistent,
        },
        "old_4th_round_below_floor_items_reassessment": old_below_floor_items,
    }
    with open(results6_path, "w") as f:
        json.dump(r6, f, indent=2, ensure_ascii=False)

    print("\n=== 과제 4-R 결과 ===")
    print(f"\n조건 C·N=500 통합 프로브 R² (10축):")
    for k in labels:
        v = probe_result[k]
        tag = " [널]" if k in NULL_AXES else ""
        print(f"  {k:<35} R²={v['probe_r2']:.4f}  CI=[{v['probe_r2_ci_low']:.4f},{v['probe_r2_ci_high']:.4f}]{tag}")
    print(f"\n널 풀 CI = [{null_lo:.4f}, {null_hi:.4f}] (참고용 스칼라 바닥={scalar_floor_reference:.4f})")
    print("\nCI 중첩 판정:")
    for k, v in ci_overlap_verdict.items():
        print(f"  {k:<35} {v['verdict']}")
    print(f"\n이펙트 순서 — 조건C(N=500): {[o[0] for o in order_C]} {dict(order_C)}")
    print(f"이펙트 순서 — 조건A매칭(N=500): {[o[0] for o in order_A_matched]} {dict(order_A_matched)}")
    print(f"순위 일치: {rank_consistent}")
    print(f"\n저장: {results6_path}, {out_dir / 'resolution_floor_v3.png'}, {base_emb_path}")
    print("★ 여기서 멈춥니다. 재판정을 확인한 뒤 과제 5 진행 여부를 결정하세요.")


if __name__ == "__main__":
    main()
