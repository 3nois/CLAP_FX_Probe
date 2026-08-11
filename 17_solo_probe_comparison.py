"""CLAP FX Probe — 17_solo_probe_comparison.py (6차 후속 과제 4-R6/R7: 단독 렌더 + 4방향 비교)

4-R2(16_resolution_floor_v3.py)는 reverb→distortion→highshelf를 하나의 체인으로 묶어
임베딩 1개에서 10축을 다변량 회귀했다. 이건 조건(A→C)과 N(수천→500) 뿐 아니라 **구성
자체**(단독 적용 → 체인 적용)까지 동시에 바꾼 것이었다 — 1~4차는 항상 이펙트를 각각
단독으로 적용했다. 점 추정치가 체계적으로 하락한 정황(wet_level 0.271→0.080,
damping 0.046→−0.021 등, N만으로는 설명 안 되는 하락폭)이 있어 원인이 조건인지 N인지
체인인지 뒤섞여 있었다.

이 스크립트는 같은 500 기준점(과제 2 캐시와 theta 일치를 assert로 확인)에서 이펙트를
**각각 단독으로** 조건 C 렌더링해(1,500회) 이펙트별로 독립 프로브한다(1~4차와 같은
구조 — 10축을 한 회귀에 넣지 않는다). 그 결과("단독·C·500")를 아래 넷과 나란히 놓고
조건 효과/N 효과/체인 효과를 분리한다.

    구성   조건   N        출처
    단독    A    25,600   과제 1 (res_pooled — freeze 미분리, 48.7% 무효 포함 주의)
    단독    A       500   4-R4(a) 서브샘플 (task4_resolution_floor_v2)
    단독    C       500   4-R6 (신규) ← 주 근거
    체인    C       500   4-R2 (task4_resolution_floor_v2.condition_C_probe_N500)

주 판정 근거는 [단독·C·500]로 바꾼다 — 구성이 1~4차와 일치하고 조건·N도 이번 라운드와
일치하는 유일한 지점이다. 결과 해석은 이 스크립트가 단정하지 않는다. README 6차 후속
절의 판정 기준표를 따를 것.
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
GROUP_PARAMS = {
    "reverb": ["wet_level", "room_size", "damping", "width"],
    "distortion": ["drive_db"],
    "highshelf": ["gain_db", "cutoff_frequency_hz", "q", "ultrasonic_12k_gain_db", "ultrasonic_15k_gain_db"],
}
NULL_AXES = ["highshelf.ultrasonic_12k_gain_db", "highshelf.ultrasonic_15k_gain_db"]


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
# 프로브 방법론 (16_resolution_floor_v3.py와 동일)
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
    return np.array(all_scores)


def ci_from_raw(col, ci=0.95):
    lo = float(np.percentile(col, (1 - ci) / 2 * 100))
    hi = float(np.percentile(col, (1 + ci) / 2 * 100))
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
            "n_rows": int(len(Y)), "n_sources": int(len(np.unique(groups))), "n_boot_used": int(raw.shape[0]),
        }
    return result, raw


def main():
    parser = argparse.ArgumentParser(description="6차 후속 과제 4-R6/R7 — 단독 렌더 + 4방향 비교")
    parser.add_argument("--audio-dir", type=str, default="nsynth-test/audio")
    parser.add_argument("--cache", type=str, default="out/phase3_fd_cache.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.out)
    audio_dir = Path(args.audio_dir)

    # ---- 4-R6: 단독 렌더링 ----
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

    print("CLAP 모델 로딩 중...")
    device = torch.device(args.device)
    if args.device == "mps":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    clap = load_clap(device, Path(__file__).parent / "ckpts")

    emb_by_group = {g: np.zeros((n_points, 512), dtype=np.float32) for g in GROUP_PARAMS}

    for group in GROUP_PARAMS:
        print(f"{group} 단독 렌더링(조건 C) + 임베딩 중 (500회)...")
        batch_audio, batch_idx = [], []

        def flush():
            if not batch_audio:
                return
            e = embed_batch(clap, device, batch_audio)
            for k, v in zip(batch_idx, e):
                emb_by_group[group][k] = v
            batch_audio.clear()
            batch_idx.clear()

        for i in tqdm(range(n_points), desc=f"{group}"):
            s = int(src_id[i])
            theta_raw = {p: float(theta_cache[i, axis_idx[f"{group}.{p}"]]) for p in GROUP_PARAMS[group]}
            wet = RENDER_FN[group](y_dry_C_by_src[s], theta_raw)
            batch_audio.append(wet.astype(np.float32))
            batch_idx.append(i)
            if len(batch_audio) >= args.batch_size:
                flush()
        flush()

    solo_emb_path = out_dir / "phase3_solo_emb.npz"
    np.savez(
        solo_emb_path,
        emb_reverb=emb_by_group["reverb"], emb_distortion=emb_by_group["distortion"], emb_highshelf=emb_by_group["highshelf"],
        theta=theta_cache, theta_norm=theta_norm_cache, theta_axis_names=np.array(axis_names),
        src_id=src_id, instrument_family=family, theta_group_id=theta_group_id,
    )
    reloaded = np.load(solo_emb_path)
    assert np.allclose(theta_cache, reloaded["theta"]), "저장된 theta가 캐시와 불일치합니다."
    print(f"단독 렌더 임베딩 저장: {solo_emb_path} (theta 일치 확인됨)")

    # ---- 이펙트별 독립 프로브 (1~4차와 동일 구조 — 10축을 한 회귀에 넣지 않는다) ----
    print("이펙트별 독립 프로브 계산 중 (조건 C, N=500)...")
    solo_probe = {}
    raw_by_group = {}
    for group, params in GROUP_PARAMS.items():
        Y = theta_norm_cache[:, [axis_idx[f"{group}.{p}"] for p in params]]
        res, raw = probe_full(emb_by_group[group], Y, src_id, [f"{group}.{p}" for p in params], args.seed, args.n_boot)
        solo_probe.update(res)
        raw_by_group[group] = (params, raw)

    hs_params, hs_raw = raw_by_group["highshelf"]
    null_col_idx = [hs_params.index(p.split(".")[1]) for p in NULL_AXES]
    null_pool = np.concatenate([hs_raw[:, k] for k in null_col_idx])
    null_lo, null_hi = ci_from_raw(null_pool)
    print(f"단독 널 풀(초음파 12k+15k, n={len(null_pool)}) CI = [{null_lo:.4f}, {null_hi:.4f}]")

    ci_overlap_solo = {}
    for e_, p in AXES_C:
        k = f"{e_}.{p}"
        if k in NULL_AXES:
            continue
        lo, hi = solo_probe[k]["probe_r2_ci_low"], solo_probe[k]["probe_r2_ci_high"]
        overlaps = intervals_overlap(lo, hi, null_lo, null_hi)
        ci_overlap_solo[k] = {
            "ci_low": lo, "ci_high": hi, "overlaps_null": overlaps,
            "verdict": "널과 구분 안 됨 (측정 불가)" if overlaps else "널과 유의하게 다름 (신호 있음)",
        }

    # ---- 4-R7: 4방향 비교 ----
    results6_path = out_dir / "results_6.json"
    with open(results6_path) as f:
        r6 = json.load(f)
    task1 = r6["task1_freeze_stratified_probe"]
    task4r2 = r6["task4_resolution_floor_v2"]

    def entry(r2, lo, hi, n, construction, condition, note=""):
        return {"probe_r2": r2, "ci_low": lo, "ci_high": hi, "n": n, "construction": construction, "condition": condition, "note": note}

    four_way = {}
    for k in [f"{e}.{p}" for e, p in AXES_C]:
        four_way[k] = {}

    # 단독A 원본 전체N (과제 1) — reverb는 freeze=0만(유효 표본), pooled(freeze 미분리,
    # 48.7% 무효 포함)는 쓰지 않는다. distortion/highshelf는 애초에 freeze 문제가 없어
    # 전체 N을 그대로 쓴다. ★ 사용자가 인용한 "wet_level 0.271"은 freeze=0 값이다 —
    # pooled 값(0.056)을 썼던 첫 시도는 틀린 대조군이었다.
    for p in ["wet_level", "room_size", "damping", "width"]:
        v = task1["reverb_freeze0"][p]
        four_way[f"reverb.{p}"]["단독A_원본N"] = entry(
            v["probe_r2"], v["probe_r2_ci_low"], v["probe_r2_ci_high"], v["n_rows"], "단독", "A",
            "freeze=0만(유효 표본), N은 축별 상이 — pooled(48.7% 무효 포함)는 오염된 대조군이라 배제")
    v = task1["distortion_full"]["drive_db"]
    four_way["distortion.drive_db"]["단독A_원본N"] = entry(v["probe_r2"], v["probe_r2_ci_low"], v["probe_r2_ci_high"], v["n_rows"], "단독", "A")
    for p in ["gain_db", "cutoff_frequency_hz", "q"]:
        v = task1["highshelf_full"][p]
        four_way[f"highshelf.{p}"]["단독A_원본N"] = entry(v["probe_r2"], v["probe_r2_ci_low"], v["probe_r2_ci_high"], v["n_rows"], "단독", "A",
                                                          "range 500~8000 (조건C는 500~4000)" if p == "cutoff_frequency_hz" else "")

    # 단독A 500 (4-R4 서브샘플)
    for k, v in task4r2["condition_A_n_matched_subsample"].items():
        four_way[k]["단독A_500"] = entry(v["mean"], v["pct2_5"], v["pct97_5"], int(v["n_rows_actual_mean"]), "단독", "A", "src_id 단위 100회 서브샘플 분포")

    # 단독C 500 (신규, 4-R6)
    for k, v in solo_probe.items():
        four_way[k]["단독C_500"] = entry(v["probe_r2"], v["probe_r2_ci_low"], v["probe_r2_ci_high"], v["n_rows"], "단독", "C",
                                        "range 500~4000" if k == "highshelf.cutoff_frequency_hz" else "")
    for k in NULL_AXES:
        v = solo_probe[k]
        four_way.setdefault(k, {})["단독C_500"] = entry(v["probe_r2"], v["probe_r2_ci_low"], v["probe_r2_ci_high"], v["n_rows"], "단독", "C", "[널]")

    # 체인C 500 (4-R2)
    for k, v in task4r2["condition_C_probe_N500"].items():
        four_way.setdefault(k, {})["체인C_500"] = entry(v["probe_r2"], v["probe_r2_ci_low"], v["probe_r2_ci_high"], v["n_rows"], "체인", "C",
                                                        "[널]" if k in NULL_AXES else "")

    # ---- 분리 판정: (a) 조건 효과, (b) N 효과, (c) 체인 효과 ----
    decomposition = {}
    for k, row in four_way.items():
        d = {}
        if "단독C_500" in row and "단독A_500" in row:
            d["조건효과_C500_minus_A500"] = row["단독C_500"]["probe_r2"] - row["단독A_500"]["probe_r2"]
        if "단독A_500" in row and "단독A_원본N" in row:
            d["N효과_A500_minus_A원본N"] = row["단독A_500"]["probe_r2"] - row["단독A_원본N"]["probe_r2"]
        if "체인C_500" in row and "단독C_500" in row:
            d["체인효과_체인C500_minus_단독C500"] = row["체인C_500"]["probe_r2"] - row["단독C_500"]["probe_r2"]
        decomposition[k] = d

    chain_effect_flags = {
        k: (abs(d.get("체인효과_체인C500_minus_단독C500", 0.0)) > 0.05)
        for k, d in decomposition.items() if "체인효과_체인C500_minus_단독C500" in d
    }

    # ---- wet_level 회복 여부 ----
    wl_solo_c = four_way["reverb.wet_level"]["단독C_500"]["probe_r2"]
    wl_task1 = four_way["reverb.wet_level"]["단독A_원본N"]["probe_r2"]  # 참고: pooled라 freeze=0 0.271과 다름
    wl_a500 = four_way["reverb.wet_level"]["단독A_500"]["probe_r2"]
    wet_level_recovered = wl_solo_c > (wl_a500 * 0.5)  # A500 대비 절반 이상 회복하면 "회복"으로 본다
    wet_level_diag = (
        "회복됨 — 체인이 원인이었다는 가설과 일치"
        if wet_level_recovered else
        "회복 안 됨 — 체인이 아니라 조건 C(dry 0.3 정규화) 자체가 원인일 가능성. 별도 조사 필요"
    )

    # ---- 이펙트 순서 (단독C·500) ----
    order_solo_c = sorted(
        [("reverb", float(np.mean([solo_probe[f"reverb.{p}"]["probe_r2"] for p in ["wet_level", "room_size", "damping", "width"]]))),
         ("distortion", solo_probe["distortion.drive_db"]["probe_r2"]),
         ("highshelf", float(np.mean([solo_probe[f"highshelf.{p}"]["probe_r2"] for p in ["gain_db", "cutoff_frequency_hz", "q"]])))],
        key=lambda x: -x[1])

    # ---- damping/width 재판정 (주 근거: 단독C·500) ----
    damping_width_verdict = {k: ci_overlap_solo[k] for k in ["reverb.damping", "reverb.width"]}

    # ---- 그림 ----
    real_axes = [f"{e}.{p}" for e, p in AXES_C]
    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    configs = [("단독A_원본N", "#c3c2b7", 0), ("단독A_500", "#898781", 1), ("단독C_500", "#2a78d6", 2), ("체인C_500", "#e34948", 3)]
    width = 0.2
    x = np.arange(len(real_axes))
    for name, color, off in configs:
        r2s, lo, hi = [], [], []
        for k in real_axes:
            v = four_way[k].get(name)
            if v is None:
                r2s.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            else:
                r2s.append(v["probe_r2"]); lo.append(v["ci_low"]); hi.append(v["ci_high"])
        r2s, lo, hi = np.array(r2s), np.array(lo), np.array(hi)
        yerr = np.array([np.clip(r2s - lo, 0, None), np.clip(hi - r2s, 0, None)])
        ax.bar(x + (off - 1.5) * width, r2s, width, yerr=yerr, capsize=2, label=name, color=color, zorder=3)
    ax.axhspan(null_lo, null_hi, color=COLORS["null"], alpha=0.12, zorder=1, label=f"단독C 널 CI [{null_lo:.3f},{null_hi:.3f}]")
    ax.set_xticks(x); ax.set_xticklabels(real_axes, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Held-out R²")
    ax.set_title("4방향 비교 — 구성(단독/체인) x 조건(A/C) x N")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "composition_comparison.png")
    plt.close(fig)

    # ---- 저장 ----
    elapsed = time.time() - t_start
    r6.setdefault("meta", {})
    r6["meta"].update({
        "task4r7_seed": args.seed, "task4r7_elapsed_sec": elapsed, "task4r7_n_boot": args.n_boot,
        "task4r7_solo_emb_path": str(solo_emb_path),
    })
    r6["task4_r7_composition"] = {
        "method_note": "단독(이펙트별 독립 렌더) 조건C N=500 프로브(4-R6)를 신규 산출하고, "
                        "단독A(원본N, 과제1)/단독A(N=500 서브샘플)/체인C(4-R2)와 4방향 비교(4-R7). "
                        "주 판정 근거는 단독C_500 — 구성·조건·N이 1~4차 구조 + 이번 라운드 조건과 모두 일치하는 유일한 지점.",
        "solo_probe_condition_C_N500": solo_probe,
        "solo_null_pool": {"columns": NULL_AXES, "n_pooled": int(len(null_pool)), "ci_low": null_lo, "ci_high": null_hi},
        "ci_overlap_solo_C_500": ci_overlap_solo,
        "four_way_comparison": four_way,
        "decomposition": decomposition,
        "chain_effect_large_flag_by_axis": chain_effect_flags,
        "wet_level_recovery_check": {
            "solo_C_500": wl_solo_c, "solo_A_500": wl_a500, "solo_A_original_pooled_note": wl_task1,
            "recovered": wet_level_recovered, "diagnosis": wet_level_diag,
        },
        "effect_order_solo_C_500": {"order": [o[0] for o in order_solo_c], "means": dict(order_solo_c)},
        "damping_width_verdict_solo_C_500": damping_width_verdict,
    }
    with open(results6_path, "w") as f:
        json.dump(r6, f, indent=2, ensure_ascii=False)

    print("\n=== 과제 4-R6/R7 결과 ===")
    print(f"\n단독·C·500 프로브 R² (이펙트별 독립):")
    for k in real_axes:
        v = solo_probe[k]
        tag = " [널]" if k in NULL_AXES else ""
        print(f"  {k:<35} R²={v['probe_r2']:.4f}  CI=[{v['probe_r2_ci_low']:.4f},{v['probe_r2_ci_high']:.4f}]{tag}")
    print(f"\n단독 널 풀 CI = [{null_lo:.4f}, {null_hi:.4f}]")
    print("\nCI 중첩 판정 (단독·C·500 기준):")
    for k, v in ci_overlap_solo.items():
        print(f"  {k:<35} {v['verdict']}")
    print(f"\nwet_level 회복 진단: solo_C={wl_solo_c:.4f} vs solo_A_500={wl_a500:.4f} -> {wet_level_diag}")
    print(f"\n이펙트 순서(단독·C·500): {[o[0] for o in order_solo_c]} {dict(order_solo_c)}")
    print("\n분리 판정(체인 효과 큰 축, |Δ|>0.05):")
    for k, flag in chain_effect_flags.items():
        if flag:
            print(f"  ⚠ {k}: Δ(체인C-단독C)={decomposition[k]['체인효과_체인C500_minus_단독C500']:.4f}")
    print(f"\n저장: {results6_path}, {out_dir / 'composition_comparison.png'}, {solo_emb_path}")
    print("★ 여기서 멈춥니다. 4방향 비교와 재판정을 확인한 뒤 과제 5 진행 여부를 결정하세요.")


if __name__ == "__main__":
    main()
