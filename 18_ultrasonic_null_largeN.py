"""CLAP FX Probe — 18_ultrasonic_null_largeN.py (6차 후속 과제 4-R8: 큰 N 널 측정 + 재판정)

4-R6/R7에서 드러난 문제: N=500 CI-중첩 검정은 검정력이 부족해 실재하는 효과도 "측정
불가"로 오판한다(wet_level이 N=13,137에서는 R²=0.271로 명백한 신호인데 N=500에서는
CI가 초음파 널과 겹쳤다). Ridge 프로브는 512차원 특징을 쓰는데 N=500(훈련 ~350행)이면
p/n>1이라 과적합으로 held-out R²가 체계적으로 눌린다. 근본 원인은 "널은 N=500에만
있다"는 N 불일치였다.

이 스크립트는 초음파 축만 3차 highshelf와 같은 규모·조건으로 다시 렌더링해
(800소스 x 16θ = 12,800점, 조건 A, 단독 적용) 큰 N 널을 만든다. 이 널을 기준으로
과제 1의 freeze=0 reverb 값, 3차 distortion/highshelf 값, 이펙트 순서를 재판정한다 —
이제 널과 대상이 같은 크기(N=6,400~13,137)라 CI 중첩 검정이 유효하다.

조건 C·N=500 결과(4-R2/4-R6)는 폐기하지 않는다 — 저검정력 참고값이자 조건 효과·체인
효과 분리(4-R7)에는 여전히 유효하다.

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
from pedalboard import HighShelfFilter, Pedalboard
from scipy.stats import qmc
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
PEAK_TARGET_A = 0.7
SILENCE_PEAK_THRESHOLD = 1e-4

CLAP_REPO_ID = "lukewys/laion_clap"
CLAP_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"

ULTRASONIC_12K_HZ = 12000.0
ULTRASONIC_15K_HZ = 15000.0
ULTRASONIC_Q = 0.7071067811865476
ULTRASONIC_RANGE = (-9.0, 9.0)
N_THETA_PER_SOURCE = 16  # 3차 highshelf 규모(N_SAMPLES_PER_SOURCE["highshelf"]=16)와 일치


def to_raw(u):
    lo, hi = ULTRASONIC_RANGE
    return float(lo + float(np.clip(u, 0.0, 1.0)) * (hi - lo))


def render_ultrasonic(y, gain12, gain15):
    board = Pedalboard([
        HighShelfFilter(cutoff_frequency_hz=ULTRASONIC_12K_HZ, gain_db=gain12, q=ULTRASONIC_Q),
        HighShelfFilter(cutoff_frequency_hz=ULTRASONIC_15K_HZ, gain_db=gain15, q=ULTRASONIC_Q),
    ])
    wet = board(y, SAMPLE_RATE)
    peak = float(np.abs(wet).max())
    if peak > 1.0:
        wet = wet * (0.99 / peak)
    return wet.astype(np.float32)


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
# 04_probe.py / 12_freeze_probe.py와 동일한 프로브 방법론 (특징 수·정규화·
# 하이퍼파라미터 전부 동일 — Ridge alpha=1.0, GroupShuffleSplit test_size=0.3,
# n_splits=5, source-level 부트스트랩 n_boot=1000)
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
    parser = argparse.ArgumentParser(description="6차 후속 과제 4-R8 — 큰 N 널 측정 + 재판정")
    parser.add_argument("--audio-dir", type=str, default="nsynth-test/audio")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.out)
    audio_dir = Path(args.audio_dir)

    print("임베딩 메타데이터 로딩 중 (3차와 동일한 800소스 목록)...")
    d = np.load(args.embeddings, allow_pickle=False)
    dry_mask = d["effect"] == "dry"
    dry_src_ids = d["src_id"][dry_mask]
    filename_by_src = dict(zip(dry_src_ids.tolist(), d["filename"][dry_mask].tolist()))
    family_by_src = dict(zip(dry_src_ids.tolist(), d["instrument_family"][dry_mask].tolist()))
    all_srcs = sorted(filename_by_src.keys())
    print(f"소스 {len(all_srcs)}개 (3차 전체와 동일)")

    print("소스 오디오 로딩 중 (조건 A, 피크 0.7)...")
    y_dry_A_by_src = {}
    for s in tqdm(all_srcs, desc="오디오 로딩"):
        r = load_raw(audio_dir / filename_by_src[s])
        if r is None:
            raise RuntimeError(f"src_id={s}가 무음입니다 — 3차 때는 통과했는데 이상합니다.")
        y_raw, peak = r
        y_dry_A_by_src[s] = (y_raw * (PEAK_TARGET_A / peak)).astype(np.float32)

    print(f"θ 샘플링 중 (소스당 결합 LHS 2차원, n={N_THETA_PER_SOURCE})...")
    point_src, point_family = [], []
    u12_list, u15_list = [], []
    for s in all_srcs:
        sampler = qmc.LatinHypercube(d=2, seed=np.random.default_rng([args.seed, int(s)]))
        unit = sampler.random(n=N_THETA_PER_SOURCE)
        for i in range(N_THETA_PER_SOURCE):
            u12_list.append(float(unit[i, 0]))
            u15_list.append(float(unit[i, 1]))
            point_src.append(s)
            point_family.append(family_by_src[s])
    n_points = len(point_src)
    print(f"평가점 {n_points}개 (소스 {len(all_srcs)} x θ {N_THETA_PER_SOURCE})")

    device = torch.device(args.device)
    if args.device == "mps":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print("CLAP 모델 로딩 중...")
    clap = load_clap(device, Path(__file__).parent / "ckpts")

    embeddings = np.zeros((n_points, 512), dtype=np.float32)
    batch_audio, batch_idx = [], []

    def flush():
        if not batch_audio:
            return
        e = embed_batch(clap, device, batch_audio)
        for k, v in zip(batch_idx, e):
            embeddings[k] = v
        batch_audio.clear()
        batch_idx.clear()

    print("렌더링 + 임베딩 중 (조건 A, 단독 — 초음파 12k+15k 캐스케이드)...")
    for i in tqdm(range(n_points), desc="렌더링+임베딩"):
        s = point_src[i]
        gain12 = to_raw(u12_list[i])
        gain15 = to_raw(u15_list[i])
        wet = render_ultrasonic(y_dry_A_by_src[s], gain12, gain15)
        batch_audio.append(wet)
        batch_idx.append(i)
        if len(batch_audio) >= args.batch_size:
            flush()
    flush()

    src_id_arr = np.array(point_src, dtype=np.int64)
    family_arr = np.array(point_family)
    theta_norm_arr = np.array([u12_list, u15_list], dtype=np.float64).T  # (n,2)
    theta_raw_arr = np.array([[to_raw(u) for u in u12_list], [to_raw(u) for u in u15_list]], dtype=np.float64).T

    npz_path = out_dir / "ultrasonic_null_largeN.npz"
    np.savez(
        npz_path, embeddings=embeddings, theta_norm=theta_norm_arr, theta_raw=theta_raw_arr,
        src_id=src_id_arr, instrument_family=family_arr,
        param_names=np.array(["ultrasonic_12k_gain_db", "ultrasonic_15k_gain_db"]),
    )
    print(f"저장: {npz_path}")

    print("프로브 R² 계산 중 (3차/과제1과 동일 절차, source-level 부트스트랩 CI)...")
    param_names = ["ultrasonic_12k_gain_db", "ultrasonic_15k_gain_db"]
    probe_result, raw = probe_full(embeddings, theta_norm_arr, src_id_arr, param_names, args.seed, args.n_boot)
    null_pool = np.concatenate([raw[:, 0], raw[:, 1]])
    null_lo, null_hi = ci_from_raw(null_pool)
    print(f"큰 N 널(12k+15k 통합, N={n_points}, n_boot_pooled={len(null_pool)}) CI = [{null_lo:.4f}, {null_hi:.4f}]")

    # ---- 재판정: 과제 1(freeze=0 reverb), 3차 distortion/highshelf ----
    results6_path = out_dir / "results_6.json"
    with open(results6_path) as f:
        r6 = json.load(f)
    task1 = r6["task1_freeze_stratified_probe"]

    targets = {}
    for p in ["wet_level", "room_size", "damping", "width"]:
        v = task1["reverb_freeze0"][p]
        targets[f"reverb.{p}"] = {"probe_r2": v["probe_r2"], "ci_low": v["probe_r2_ci_low"], "ci_high": v["probe_r2_ci_high"],
                                   "n_rows": v["n_rows"], "condition": "A", "construction": "단독", "source": "과제1 freeze=0"}
    v = task1["distortion_full"]["drive_db"]
    targets["distortion.drive_db"] = {"probe_r2": v["probe_r2"], "ci_low": v["probe_r2_ci_low"], "ci_high": v["probe_r2_ci_high"],
                                       "n_rows": v["n_rows"], "condition": "A", "construction": "단독", "source": "3차 전체"}
    for p in ["gain_db", "cutoff_frequency_hz", "q"]:
        v = task1["highshelf_full"][p]
        targets[f"highshelf.{p}"] = {"probe_r2": v["probe_r2"], "ci_low": v["probe_r2_ci_low"], "ci_high": v["probe_r2_ci_high"],
                                      "n_rows": v["n_rows"], "condition": "A", "construction": "단독", "source": "3차 전체"}

    ci_overlap_largeN = {}
    for k, v in targets.items():
        overlaps = intervals_overlap(v["ci_low"], v["ci_high"], null_lo, null_hi)
        ci_overlap_largeN[k] = {
            **v, "overlaps_null": overlaps,
            "verdict": "널과 구분 안 됨 (측정 불가)" if overlaps else "널과 유의하게 다름 (신호 있음)",
        }

    order = sorted(
        [("reverb", float(np.mean([task1["reverb_freeze0"][p]["probe_r2"] for p in ["wet_level", "room_size", "damping", "width"]]))),
         ("distortion", task1["distortion_full"]["drive_db"]["probe_r2"]),
         ("highshelf", float(np.mean([task1["highshelf_full"][p]["probe_r2"] for p in ["gain_db", "cutoff_frequency_hz", "q"]])))],
        key=lambda x: -x[1])

    # ---- 그림 ----
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    labels = list(targets.keys())
    r2s = [ci_overlap_largeN[k]["probe_r2"] for k in labels]
    lo = [ci_overlap_largeN[k]["ci_low"] for k in labels]
    hi = [ci_overlap_largeN[k]["ci_high"] for k in labels]
    x = np.arange(len(labels))
    yerr = np.array([np.clip(np.array(r2s) - np.array(lo), 0, None), np.clip(np.array(hi) - np.array(r2s), 0, None)])
    bar_colors = [COLORS[k.split(".")[0]] for k in labels]
    hatch = ["//" if ci_overlap_largeN[k]["overlaps_null"] else None for k in labels]
    bars = ax.bar(x, r2s, yerr=yerr, capsize=3, color=bar_colors, zorder=3)
    for b, hh in zip(bars, hatch):
        if hh:
            b.set_hatch(hh)
            b.set_edgecolor("black")
    ax.axhspan(null_lo, null_hi, color=COLORS["null"], alpha=0.15, zorder=1, label=f"큰N 널 CI(N={n_points}) [{null_lo:.4f},{null_hi:.4f}]")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Held-out R² (조건 A, 단독, 각자 원본 N)")
    ax.set_title("과제 4-R8 — 큰 N 널 기준 재판정 (빗금=널과 CI 중첩)")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "ultrasonic_null_largeN.png")
    plt.close(fig)

    # ---- 저장 ----
    elapsed = time.time() - t_start
    r6.setdefault("meta", {})
    r6["meta"].update({
        "task4r8_seed": args.seed, "task4r8_elapsed_sec": elapsed, "task4r8_n_boot": args.n_boot,
        "task4r8_n_points": n_points, "task4r8_n_sources": len(all_srcs), "task4r8_npz_path": str(npz_path),
    })
    r6["task4_r8_largeN_null"] = {
        "method_note": "초음파 12k/15k만 800소스x16θ=12,800점, 조건A(피크0.7), 단독 적용(체인 아님), "
                        "3차/과제1과 동일 프로브 절차(Ridge alpha=1.0, GroupShuffleSplit, source-level 부트스트랩). "
                        "N=500 CI-중첩 검정(4-R2/4-R6)의 검정력 부족 문제를 해결하기 위한 큰 N 널.",
        "probe_result": probe_result,
        "null_pool": {"n_points": n_points, "n_pooled_boot": int(len(null_pool)), "ci_low": null_lo, "ci_high": null_hi},
        "reassessment_vs_largeN_null": ci_overlap_largeN,
        "effect_order": {"order": [o[0] for o in order], "means": dict(order)},
        "condition_C_N500_results_status": "폐기하지 않음 — 저검정력 참고값, 조건효과/체인효과 분리(4-R7)에는 유효",
    }
    with open(results6_path, "w") as f:
        json.dump(r6, f, indent=2, ensure_ascii=False)

    print("\n=== 과제 4-R8 결과 ===")
    print(f"\n큰 N 널 (N={n_points}, 조건A, 단독): ultrasonic_12k R²={probe_result['ultrasonic_12k_gain_db']['probe_r2']:.4f} "
          f"CI=[{probe_result['ultrasonic_12k_gain_db']['probe_r2_ci_low']:.4f},{probe_result['ultrasonic_12k_gain_db']['probe_r2_ci_high']:.4f}]  "
          f"ultrasonic_15k R²={probe_result['ultrasonic_15k_gain_db']['probe_r2']:.4f} "
          f"CI=[{probe_result['ultrasonic_15k_gain_db']['probe_r2_ci_low']:.4f},{probe_result['ultrasonic_15k_gain_db']['probe_r2_ci_high']:.4f}]")
    print(f"널 풀 CI = [{null_lo:.4f}, {null_hi:.4f}]")
    print("\n재판정 (과제1 freeze=0 / 3차 원본 N 기준):")
    for k, v in ci_overlap_largeN.items():
        print(f"  {k:<28} R²={v['probe_r2']:.4f}  CI=[{v['ci_low']:.4f},{v['ci_high']:.4f}]  N={v['n_rows']}  {v['verdict']}")
    print(f"\n이펙트 순서: {[o[0] for o in order]} {dict(order)}")
    print(f"\n저장: {results6_path}, {out_dir / 'ultrasonic_null_largeN.png'}, {npz_path}")
    print("★ 여기서 멈춥니다. 재판정을 확인한 뒤 다음 단계를 결정하세요.")


if __name__ == "__main__":
    main()
