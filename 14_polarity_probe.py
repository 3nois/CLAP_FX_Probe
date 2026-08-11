"""CLAP FX Probe — 14_polarity_probe.py (6차 후속 과제 3: 극성 반전 진단)

★ 해상도 바닥으로 쓰지 않는다 — 역할이 다르다 (과제 4는 초음파 셸프 축으로 별도 확정).

CLAP은 mel 스펙트로그램(= |STFT|, 위상을 버림) 기반이라 |X(f)| = |−X(f)|다. dry를
"이펙트 적용 전"에 부호 반전한 뒤 조건 C 파이프라인 전체(피크 0.3 정규화 → 이펙트)를
통과시키면:

  · 이펙트가 선형(reverb, highshelf — LTI)이고 정규화가 진짜로 |·| 기반 대칭이면
    pipeline(-y) == -pipeline(y)가 부동소수점 수준으로 성립해야 한다 → 임베딩도
    (|STFT|만 보므로) 사실상 동일해야 한다.
  · distortion의 waveshaping이 비대칭(진공관 에뮬 계열은 일부러 짝수 배음을 냄)이면
    pipeline(-y) != -pipeline(y)가 되어 진짜로 다른 오디오가 나오고, 임베딩도 갈라진다.

이 스크립트는 두 가지를 함께 낸다.
  (1) 파형 자체의 대칭성 — ‖wet(y) + wet(-y)‖ / (‖wet(y)‖+‖wet(-y)‖) — 정확한 수치 검사.
  (2) 과제 명세가 요구하는 이진 분류 accuracy/NMI — wet(y) 임베딩 500개 vs wet(-y) 임베딩
      500개를 "극성이 뒤집혔는가"로 분류하는 held-out 정확도(로지스틱 회귀,
      source-level GroupShuffleSplit)와 비지도 NMI(KMeans 2군집).

out/phase3_fd_cache.npz의 500 평가점(src_id, theta, theta_axis_names)을 그대로 재사용한다
— 재샘플링하지 않는다. 재렌더링은 하되(이 진단 전용 6렌더/점), FD 캐시는 건드리지 않는다.

결과 해석은 이 스크립트가 단정하지 않는다. README 6차 후속 절의 판정 기준표를 따를 것.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import librosa
import torch
from huggingface_hub import hf_hub_download
from pedalboard import Distortion, HighShelfFilter, Pedalboard, Reverb
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

SAMPLE_RATE = 48000
DURATION_SEC = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
SILENCE_PEAK_THRESHOLD = 1e-4

CLAP_REPO_ID = "lukewys/laion_clap"
CLAP_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"

ULTRASONIC_12K_HZ = 12000.0
ULTRASONIC_15K_HZ = 15000.0
ULTRASONIC_Q = 0.7071067811865476

GROUP_PARAMS = {
    "reverb": ["wet_level", "room_size", "damping", "width"],
    "distortion": ["drive_db"],
    "highshelf": ["gain_db", "cutoff_frequency_hz", "q", "ultrasonic_12k_gain_db", "ultrasonic_15k_gain_db"],
}
GROUPS = ["reverb", "distortion", "highshelf"]


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


def held_out_accuracy(X, y, groups, seed, n_splits=10, test_size=0.3):
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    accs = []
    for train_idx, test_idx in gss.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
    return float(np.mean(accs)), float(np.std(accs)), len(accs)


def main():
    parser = argparse.ArgumentParser(description="6차 후속 과제 3 — 극성 반전 진단")
    parser.add_argument("--audio-dir", type=str, default="nsynth-test/audio")
    parser.add_argument("--cache", type=str, default="out/phase3_fd_cache.npz")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.out)
    audio_dir = Path(args.audio_dir)

    print(f"캐시 로딩 중: {args.cache}")
    c = np.load(args.cache, allow_pickle=False)
    theta = c["theta"]  # (n_points, 10) raw
    axis_names = [str(x) for x in c["theta_axis_names"]]
    src_id = c["src_id"]
    family = c["instrument_family"]
    filename = c["filename"]
    peak_target_c = float(c["peak_target_c"])
    n_points = theta.shape[0]
    axis_idx = {ax: k for k, ax in enumerate(axis_names)}
    print(f"평가점 {n_points}개 로드 (peak_target_c={peak_target_c})")

    print("임베딩 메타데이터 로딩 중 (파일명 확인용, 캐시에 이미 있으면 생략)...")
    d = np.load(args.embeddings, allow_pickle=False)
    dry_mask = d["effect"] == "dry"
    filename_by_src = dict(zip(d["src_id"][dry_mask].tolist(), d["filename"][dry_mask].tolist()))

    unique_srcs = sorted(set(int(s) for s in src_id))
    print(f"소스 {len(unique_srcs)}개 오디오 로딩 중...")
    y_dry_pos_by_src = {}
    for s in tqdm(unique_srcs, desc="오디오 로딩"):
        fname = filename_by_src.get(s) or str(filename[list(src_id).index(s)])
        r = load_raw(audio_dir / fname)
        if r is None:
            raise RuntimeError(f"src_id={s}가 무음입니다.")
        y_raw, peak = r
        y_dry_pos_by_src[s] = (y_raw * (peak_target_c / peak)).astype(np.float32)

    theta_raw_by_point = []
    for i in range(n_points):
        d_i = {}
        for group, params in GROUP_PARAMS.items():
            d_i[group] = {p: float(theta[i, axis_idx[f"{group}.{p}"]]) for p in params}
        theta_raw_by_point.append(d_i)

    print("렌더링 + 대칭성 계산 중 (pos/neg, 이펙트 3개 x 500점)...")
    device = torch.device(args.device)
    if args.device == "mps":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print("CLAP 모델 로딩 중...")
    clap = load_clap(device, Path(__file__).parent / "ckpts")

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

    sym_stats = {g: {"rel_asym": []} for g in GROUPS}

    all_jobs = []  # (i, group, polarity, wet)
    for i in tqdm(range(n_points), desc="렌더링"):
        s = int(src_id[i])
        y_pos = y_dry_pos_by_src[s]
        y_neg = -y_pos
        for group in GROUPS:
            theta_raw = theta_raw_by_point[i][group]
            wet_pos = RENDER_FN[group](y_pos, theta_raw)
            wet_neg = RENDER_FN[group](y_neg, theta_raw)
            denom = float(np.linalg.norm(wet_pos) + np.linalg.norm(wet_neg)) + 1e-12
            rel_asym = float(np.linalg.norm(wet_pos + wet_neg)) / denom
            sym_stats[group]["rel_asym"].append(rel_asym)
            batch_audio.append(wet_pos.astype(np.float32)); batch_keys.append((i, group, "pos"))
            if len(batch_audio) >= args.batch_size:
                flush()
            batch_audio.append(wet_neg.astype(np.float32)); batch_keys.append((i, group, "neg"))
            if len(batch_audio) >= args.batch_size:
                flush()
    flush()

    print("이진 분류(극성) 계산 중...")
    classification_by_effect = {}
    for group in GROUPS:
        X_pos = np.stack([embeddings_by_job[(i, group, "pos")] for i in range(n_points)])
        X_neg = np.stack([embeddings_by_job[(i, group, "neg")] for i in range(n_points)])
        X = np.concatenate([X_pos, X_neg], axis=0)
        y_label = np.concatenate([np.zeros(n_points), np.ones(n_points)]).astype(int)
        groups_arr = np.concatenate([src_id, src_id])

        acc_mean, acc_std, n_splits_used = held_out_accuracy(X, y_label, groups_arr, args.seed)

        km = KMeans(n_clusters=2, n_init=10, random_state=args.seed).fit(X)
        nmi = float(normalized_mutual_info_score(y_label, km.labels_))

        cos_pos_neg = np.array([
            float(np.dot(X_pos[i], X_neg[i]) / (np.linalg.norm(X_pos[i]) * np.linalg.norm(X_neg[i]) + 1e-12))
            for i in range(n_points)
        ])

        rel_asym_arr = np.array(sym_stats[group]["rel_asym"])
        classification_by_effect[group] = {
            "held_out_accuracy_mean": acc_mean, "held_out_accuracy_std": acc_std, "n_splits_used": n_splits_used,
            "nmi_kmeans2": nmi,
            "cos_emb_pos_vs_neg": {
                "mean": float(cos_pos_neg.mean()), "median": float(np.median(cos_pos_neg)),
                "min": float(cos_pos_neg.min()),
            },
            "waveform_relative_asymmetry": {
                "mean": float(rel_asym_arr.mean()), "median": float(np.median(rel_asym_arr)),
                "max": float(rel_asym_arr.max()),
                "note": "‖wet(y)+wet(-y)‖/(‖wet(y)‖+‖wet(-y)‖). 0에 가까우면 pipeline(-y)==-pipeline(y) (대칭).",
            },
        }

    verdicts = {}
    for group in GROUPS:
        acc = classification_by_effect[group]["held_out_accuracy_mean"]
        verdicts[group] = "asymmetric" if acc > 0.55 else "symmetric"  # 0.5 근방 노이즈 여유 폭 0.05
    if verdicts.get("distortion") == "asymmetric" and all(verdicts[g] == "symmetric" for g in GROUPS if g != "distortion"):
        overall = "distortion만 비대칭 — waveshaping 특성(문서화, 버그 아님)"
    elif all(v == "symmetric" for v in verdicts.values()):
        overall = "전 이펙트 대칭 — 파이프라인 정상"
    else:
        overall = "distortion 외에서도 비대칭 — 정규화/클리핑 코드 점검 필요"

    elapsed = time.time() - t_start
    results6_path = out_dir / "results_6.json"
    results6 = {}
    if results6_path.exists():
        with open(results6_path) as f:
            results6 = json.load(f)
    results6.setdefault("meta", {})
    results6["meta"].update({"task3_seed": args.seed, "task3_elapsed_sec": elapsed, "task3_n_points": n_points})
    results6["task3_polarity_probe"] = {
        "by_effect": classification_by_effect,
        "verdict_by_effect": verdicts,
        "overall_verdict": overall,
        "resolution_floor_role": "제외 — 극성 반전은 해상도 바닥에 포함하지 않는다 (README 참고)",
    }
    with open(results6_path, "w") as f:
        json.dump(results6, f, indent=2, ensure_ascii=False)

    print("\n=== 과제 3 결과 ===")
    for group in GROUPS:
        r = classification_by_effect[group]
        print(f"{group:<12} accuracy={r['held_out_accuracy_mean']:.4f}±{r['held_out_accuracy_std']:.4f}  "
              f"NMI={r['nmi_kmeans2']:.4f}  cos(pos,neg) median={r['cos_emb_pos_vs_neg']['median']:.6f}  "
              f"파형 상대비대칭 median={r['waveform_relative_asymmetry']['median']:.2e}")
    print(f"\n판정: {overall}")
    print(f"\n저장: {results6_path}")
    print("★ 여기서 멈춥니다. 판정을 확인한 뒤 과제 4 진행 여부를 결정하세요.")


if __name__ == "__main__":
    main()
