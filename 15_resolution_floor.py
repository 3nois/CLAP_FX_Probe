"""CLAP FX Probe — 15_resolution_floor.py (6차 후속 과제 4: 새 해상도 바닥 확정 + 순서 재판정)

4차 해상도 바닥(0.0096)은 reverb.width 프로브 R² CI 상단으로 정의했다. 과제 1에서
width가 실제로는 음성 통제가 아님(freeze=0에서 R²=0.066, CI가 0을 명확히 벗어남)이
밝혀져 그 정의가 무효화됐다. 이 스크립트는 진짜 널(이펙트 종류와 무관하게 신호가 없는
축)인 초음파 하이셸프 축(12kHz/15kHz, NSynth Nyquist 8kHz 위)으로 바닥을 다시 정의한다.

  resolution_floor = max(ultrasonic_12k_gain_db R² CI 상단, ultrasonic_15k_gain_db R² CI 상단)

조건 C의 highshelf 센터 렌더링(500점 — 과제 2 캐시의 θ를 그대로 재사용, 재샘플링 없음)만
새로 만든다. FD 캐시에는 센터 임베딩 자체가 없어(점별 미분 벡터만 저장) 이 프로브를 위해
500회만 추가 렌더링한다. 방법론은 04_probe.py/12_freeze_probe.py와 동일
(다변량 Ridge, source-level GroupShuffleSplit held-out R², source-level 부트스트랩 95% CI).

극성 반전(과제 3)은 바닥에 포함하지 않는다. 결과 해석은 이 스크립트가 단정하지 않는다.
README 6차 후속 절의 판정 기준표를 따를 것.
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

HIGHSHELF_PARAMS = ["gain_db", "cutoff_frequency_hz", "q", "ultrasonic_12k_gain_db", "ultrasonic_15k_gain_db"]


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
# 04_probe.py / 12_freeze_probe.py와 동일한 프로브 방법론
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


def bootstrap_r2_ci_multi(X, Y, groups, seed, n_boot=1000, ci=0.95):
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
    all_scores = np.array(all_scores)
    lo = np.percentile(all_scores, (1 - ci) / 2 * 100, axis=0)
    hi = np.percentile(all_scores, (1 + ci) / 2 * 100, axis=0)
    return all_scores.mean(axis=0), lo, hi, int(len(all_scores))


def probe_xyz(X, Y, groups, param_names, seed, n_boot):
    r2_mean, r2_std = held_out_r2_multi(X, Y, groups, seed)
    boot_mean, ci_lo, ci_hi, n_boot_used = bootstrap_r2_ci_multi(X, Y, groups, seed, n_boot=n_boot)
    result = {}
    for i, pname in enumerate(param_names):
        result[pname] = {
            "probe_r2": float(r2_mean[i]), "probe_r2_std": float(r2_std[i]),
            "probe_r2_ci_low": float(ci_lo[i]), "probe_r2_ci_high": float(ci_hi[i]),
            "n_rows": int(len(Y)), "n_sources": int(len(np.unique(groups))), "n_boot_used": n_boot_used,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="6차 후속 과제 4 — 새 해상도 바닥 확정 + 순서 재판정")
    parser.add_argument("--audio-dir", type=str, default="nsynth-test/audio")
    parser.add_argument("--cache", type=str, default="out/phase3_fd_cache.npz")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--results4", type=str, default="out/results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.out)
    audio_dir = Path(args.audio_dir)

    print(f"캐시 로딩 중: {args.cache}")
    c = np.load(args.cache, allow_pickle=False)
    theta = c["theta"]
    theta_norm = c["theta_norm"]
    axis_names = [str(x) for x in c["theta_axis_names"]]
    src_id = c["src_id"]
    family = c["instrument_family"]
    peak_target_c = float(c["peak_target_c"])
    n_points = theta.shape[0]
    axis_idx = {ax: k for k, ax in enumerate(axis_names)}
    hs_idx = [axis_idx[f"highshelf.{p}"] for p in HIGHSHELF_PARAMS]
    print(f"평가점 {n_points}개 로드")

    print("임베딩 메타데이터 로딩 중 (파일명 확인용)...")
    d = np.load(args.embeddings, allow_pickle=False)
    dry_mask = d["effect"] == "dry"
    filename_by_src = dict(zip(d["src_id"][dry_mask].tolist(), d["filename"][dry_mask].tolist()))

    unique_srcs = sorted(set(int(s) for s in src_id))
    print(f"소스 {len(unique_srcs)}개 오디오 로딩 중...")
    y_dry_C_by_src = {}
    for s in tqdm(unique_srcs, desc="오디오 로딩"):
        r = load_raw(audio_dir / filename_by_src[s])
        if r is None:
            raise RuntimeError(f"src_id={s}가 무음입니다.")
        y_raw, peak = r
        y_dry_C_by_src[s] = (y_raw * (peak_target_c / peak)).astype(np.float32)

    print("highshelf 센터 렌더링(조건 C) + 임베딩 중 (500회)...")
    device = torch.device(args.device)
    if args.device == "mps":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    clap = load_clap(device, Path(__file__).parent / "ckpts")

    embeddings = np.zeros((n_points, 512), dtype=np.float32)
    batch_audio, batch_idx = [], []

    def flush():
        if not batch_audio:
            return
        emb = embed_batch(clap, device, batch_audio)
        for k, e in zip(batch_idx, emb):
            embeddings[k] = e
        batch_audio.clear()
        batch_idx.clear()

    for i in tqdm(range(n_points), desc="렌더링+임베딩"):
        s = int(src_id[i])
        theta_raw = {p: float(theta[i, axis_idx[f"highshelf.{p}"]]) for p in HIGHSHELF_PARAMS}
        wet = render_highshelf(y_dry_C_by_src[s], theta_raw)
        batch_audio.append(wet.astype(np.float32))
        batch_idx.append(i)
        if len(batch_audio) >= args.batch_size:
            flush()
    flush()

    print("프로브 R² 계산 중 (다변량 Ridge, source-level 부트스트랩 CI)...")
    Y = theta_norm[:, hs_idx]
    probe_result = probe_xyz(embeddings, Y, src_id, HIGHSHELF_PARAMS, args.seed, args.n_boot)

    ci_high_12k = probe_result["ultrasonic_12k_gain_db"]["probe_r2_ci_high"]
    ci_high_15k = probe_result["ultrasonic_15k_gain_db"]["probe_r2_ci_high"]
    resolution_floor = max(ci_high_12k, ci_high_15k)
    # 두 값 다 0 근방(때로 음수)이라 비율은 무의미해질 수 있다 — 절대 차이로 판단한다.
    # 임계값 0.01은 이 프로브의 CI 폭(수 %p) 대비 "노이즈로 설명 안 되는 차이" 기준.
    transition_leak_abs_diff = abs(ci_high_12k - ci_high_15k)
    transition_leak_flag = transition_leak_abs_diff > 0.01
    transition_leak_ratio = (
        max(abs(ci_high_12k), abs(ci_high_15k)) / min(abs(ci_high_12k), abs(ci_high_15k))
        if min(abs(ci_high_12k), abs(ci_high_15k)) > 1e-9 else float("inf")
    )

    print(f"새 해상도 바닥 = {resolution_floor:.5f} "
          f"(12k CI상단={ci_high_12k:.5f}, 15k CI상단={ci_high_15k:.5f})")

    # ---- 4차 값 로딩 (재판정 대상) ----
    with open(args.results4) as f:
        r4 = json.load(f)
    old_floor = r4["resolution_floor"]["value"]
    r4_params = r4["params"]

    # ---- 과제 1(results_6.json)에서 freeze=0 개별 파라미터 값 로딩 ----
    results6_path = out_dir / "results_6.json"
    with open(results6_path) as f:
        r6 = json.load(f)
    task1 = r6["task1_freeze_stratified_probe"]
    freeze0 = task1["reverb_freeze0"]
    matched = task1["matched_N"]

    def mean_r2(res, params):
        return float(np.mean([res[p]["probe_r2"] for p in params]))

    order_matched = {
        "distortion(matched)": mean_r2(matched["distortion"], ["drive_db"]),
        "reverb(freeze=0,matched)": mean_r2(matched["reverb_freeze0"], ["wet_level", "room_size", "damping", "width"]),
        "highshelf(matched)": mean_r2(matched["highshelf"], ["gain_db", "cutoff_frequency_hz", "q"]),
    }

    # 재판정: 새 바닥 기준 라벨링
    def label(r2):
        return "측정 불가(below resolution)" if r2 < resolution_floor else "resolution 이상"

    reclassification = {}
    for p in ["wet_level", "room_size", "damping", "width"]:
        r2 = freeze0[p]["probe_r2"]
        reclassification[f"reverb.{p} (freeze=0, 조건A, 과제1)"] = {"probe_r2": r2, "old_label": None, "new_label": label(r2)}
    for k, v in r4_params.items():
        reclassification[f"{k} (4차, 조건A, 원본range)"] = {
            "probe_r2": v["probe_r2"], "old_label": v["measurability"], "new_label": label(v["probe_r2"]),
        }
    for p in HIGHSHELF_PARAMS:
        r2 = probe_result[p]["probe_r2"]
        reclassification[f"highshelf.{p} (6차, 조건C, cutoff range 500-4000)"] = {
            "probe_r2": r2, "old_label": None, "new_label": label(r2),
        }

    highshelf_order_above_floor = order_matched["highshelf(matched)"] > resolution_floor
    order_verdict = (
        "highshelf(matched)가 새 바닥을 초과 — reverb > highshelf 순서 주장 가능"
        if highshelf_order_above_floor else
        "highshelf(matched)가 새 바닥 미만 — reverb > highshelf 순서 주장 불가, highshelf는 '측정 불가'로 재분류"
    )

    # ---- 그림 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    ax = axes[0]
    labels = HIGHSHELF_PARAMS
    r2s = [probe_result[p]["probe_r2"] for p in labels]
    lo = [probe_result[p]["probe_r2_ci_low"] for p in labels]
    hi = [probe_result[p]["probe_r2_ci_high"] for p in labels]
    x = np.arange(len(labels))
    yerr = np.array([[max(r - l, 0) for r, l in zip(r2s, lo)], [max(h - r, 0) for r, h in zip(r2s, hi)]])
    bar_colors = [COLORS["null"] if p.startswith("ultrasonic") else COLORS["highshelf"] for p in labels]
    ax.bar(x, r2s, yerr=yerr, capsize=3, color=bar_colors, zorder=3)
    ax.axhline(resolution_floor, color="black", linestyle="--", linewidth=1.2, label=f"새 해상도 바닥={resolution_floor:.4f}")
    ax.axhline(old_floor, color="#898781", linestyle=":", linewidth=1, label=f"4차 바닥(철회)={old_floor:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Held-out R² (조건 C, n=500)")
    ax.set_title("highshelf 프로브 R² — 초음파 통제축 포함")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    ax = axes[1]
    keys = list(order_matched.keys())
    vals = [order_matched[k] for k in keys]
    colors3 = [COLORS["distortion"], COLORS["reverb"], COLORS["highshelf"]]
    ax.bar(np.arange(3), vals, color=colors3, zorder=3)
    ax.axhline(resolution_floor, color="black", linestyle="--", linewidth=1.2, label=f"새 바닥={resolution_floor:.4f}")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(keys, fontsize=8, rotation=10)
    ax.set_ylabel("평균 R² (matched N, 과제 1)")
    ax.set_title("이펙트 간 순서 재판정 (새 바닥 대비)")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    fig.suptitle(f"과제 4 — 새 해상도 바닥(초음파 셸프 기준, n={n_points})")
    fig.tight_layout()
    fig.savefig(out_dir / "resolution_floor_v2.png")
    plt.close(fig)

    elapsed = time.time() - t_start
    r6.setdefault("meta", {})
    r6["meta"].update({"task4_seed": args.seed, "task4_elapsed_sec": elapsed, "task4_n_boot": args.n_boot})
    r6["task4_resolution_floor"] = {
        "highshelf_probe_condition_C": probe_result,
        "resolution_floor": resolution_floor,
        "resolution_floor_definition": "max(ultrasonic_12k_gain_db R² CI상단, ultrasonic_15k_gain_db R² CI상단), 조건 C, n=500",
        "ultrasonic_12k_ci_high": ci_high_12k,
        "ultrasonic_15k_ci_high": ci_high_15k,
        "transition_band_leak_abs_diff": transition_leak_abs_diff,
        "transition_band_leak_ratio_of_abs": transition_leak_ratio,
        "transition_band_leak_flag": transition_leak_flag,
        "old_floor_4th_round": old_floor,
        "old_floor_withdrawn_reason": "reverb.width가 음성 통제가 아님이 과제 1에서 확인됨",
        "effect_order_matched_N": order_matched,
        "effect_order_verdict": order_verdict,
        "reclassification": reclassification,
    }
    with open(results6_path, "w") as f:
        json.dump(r6, f, indent=2, ensure_ascii=False)

    print("\n=== 과제 4 결과 ===")
    print(f"highshelf 프로브 R² (조건 C, n={n_points}):")
    for p in HIGHSHELF_PARAMS:
        v = probe_result[p]
        print(f"  {p:<28} R²={v['probe_r2']:.4f}  CI=[{v['probe_r2_ci_low']:.4f},{v['probe_r2_ci_high']:.4f}]")
    print(f"\n새 해상도 바닥 = {resolution_floor:.5f} (4차 바닥 {old_floor:.5f}은 철회)")
    print(f"전이대역 누출: |12k CI상단 - 15k CI상단| = {transition_leak_abs_diff:.5f}  "
          f"{'⚠ 유의미한 차이' if transition_leak_flag else '(유사함, 노이즈 범위)'}")
    print(f"\n이펙트 순서(matched N): {order_matched}")
    print(f"판정: {order_verdict}")
    print("\n재분류:")
    for k, v in reclassification.items():
        print(f"  {k:<55} R²={v['probe_r2']:.4f}  이전='{v['old_label']}'  신규='{v['new_label']}'")
    print(f"\n저장: {results6_path}, {out_dir / 'resolution_floor_v2.png'}")
    print("★ 여기서 멈춥니다. 새 해상도 바닥과 순서 재판정을 확인한 뒤 과제 5 진행 여부를 결정하세요.")


if __name__ == "__main__":
    main()
