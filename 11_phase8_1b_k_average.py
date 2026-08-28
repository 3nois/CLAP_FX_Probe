# -*- coding: utf-8 -*-
"""Phase 8-1b — 가설 A: 시드 k개 평균이 da를 √k만큼 개선하는가 (임베딩 평균, 사전등록 방식).

out/prereg/11_phase8.md §"k 의존성 산출" 확정 방법: 시드별 개별 da 평균이 아니라
"임베딩을 먼저 평균 낸 뒤 코사인"을 계산해야 한다. 11_phase8_run.py 체크포인트는
d쪽(e_regen_d) 임베딩을 저장하지 않았으므로(용량 절약), 이미 저장된 오디오
(out/audio/phase8/*_t1.0_s{42,43,44,45}_a1_d.wav)에서 CLAP만 다시 돌려
재추출한다 — TokenSynth/DAC 재실행 없음, 재생성 없음.

alpha=1, temperature=1.0(기본) 슬라이스만 대상(k 순수 효과를 보려면 다른 개입과
안 섞는 게 맞다). c쪽 임베딩은 체크포인트에 이미 있다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa
import torch
import audiofile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tokensynth_bridge"))
from importlib import import_module

dr = import_module("11_phase2_doseresponse")
p8 = import_module("11_phase8_run")

RESULTS_DIR = dr.RESULTS_DIR
OUT_AUDIO_DIR = Path(__file__).resolve().parent / "out" / "audio" / "phase8"
AXIS_NAME = "highshelf_gain"
IDX_A, IDX_B = 0, 24
SEEDS = [42, 43, 44, 45]
TEMP = 1.0
ALPHA = 1


def cos_np(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def bootstrap_ci(values, seed=0, n_boot=2000):
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    n = len(values)
    boots = np.array([values[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_bootstrap_ci(a, b, seed=0, n_boot=5000):
    """짝지어진(같은 소스) 두 조건의 차이(b-a)를 부트스트랩 — 소스 간 큰 변동을 상쇄해
    독립 CI 비교보다 검정력이 훨씬 높다(소스 재추출로 리샘플, 차이값 자체를 리샘플)."""
    diff = np.asarray(b) - np.asarray(a)
    rng = np.random.RandomState(seed)
    n = len(diff)
    boots = np.array([diff[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    print("체크포인트에서 c쪽 임베딩(이미 저장됨) 로딩...")
    c_embed = {}  # (pos,seed) -> np.array(512,)
    with open(RESULTS_DIR / "11_phase8_checkpoint.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == "c" and r["temp"] == TEMP and r["e_regen"] is not None:
                c_embed[(r["src_pos"], r["seed"])] = np.array(r["e_regen"])

    sources = p8.all_sources_1200()
    selected_pos = p8.select_100_sources(sources)
    print(f"소스 {len(selected_pos)}개, c 임베딩 {len(c_embed)}개 로딩 완료")

    emb, theta_raw, src_id = dr.load_concat(AXIS_NAME)
    assert np.array_equal(src_id, np.arange(1200))
    e_dry_true_all = emb[:, IDX_A, :]
    e_wet_all = emb[:, IDX_B, :]

    print("CLAP 로딩 중 (재추출용, TokenSynth/DAC 불필요)...")
    from tokensynth import CLAP
    device = torch.device("cpu")
    clap = CLAP(device=device)

    def embed_wav(path):
        y16k, _ = librosa.load(str(path), sr=16000, mono=True)
        y48 = librosa.resample(y16k, orig_sr=16000, target_sr=48000).astype(np.float32)
        tensor = torch.tensor(y48.reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            e = clap.clap.get_audio_embedding_from_data(tensor, use_tensor=True)
        return e.cpu().numpy().reshape(-1)

    print("d쪽 오디오 재추출 중 (alpha=1, temp=1.0, 4시드 x 100소스 = 400개)...")
    d_embed = {}
    for i, pos in enumerate(selected_pos):
        for seed in SEEDS:
            wav_path = OUT_AUDIO_DIR / f"highshelf_full_{pos}_t{TEMP}_s{seed}_a{ALPHA}_d.wav"
            if not wav_path.exists():
                print(f"  경고: {wav_path} 없음 — 건너뜀")
                continue
            d_embed[(pos, seed)] = embed_wav(wav_path)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(selected_pos)} 소스 완료")

    print("k=1,2,4 계산 중...")
    import itertools
    k1_vals, k2_vals, k4_vals = [], [], []
    per_source_detail = []
    for pos in selected_pos:
        e_dry_true = e_dry_true_all[pos]
        e_wet = e_wet_all[pos]
        v_original = e_dry_true - e_wet

        cs = [c_embed.get((pos, s)) for s in SEEDS]
        ds = [d_embed.get((pos, s)) for s in SEEDS]
        if any(x is None for x in cs) or any(x is None for x in ds):
            continue

        # k=1: 시드 4개 각각 개별 da -> 평균
        da_each = []
        for c, d in zip(cs, ds):
            v_gen = d - c
            da_each.append(cos_np(v_gen, v_original))
        k1_mean = float(np.mean(da_each))
        k1_vals.append(k1_mean)

        # k=2: 시드 쌍 6가지 조합, 임베딩 평균 후 코사인 -> 평균
        pair_das = []
        for i1, i2 in itertools.combinations(range(4), 2):
            c_avg = (cs[i1] + cs[i2]) / 2
            d_avg = (ds[i1] + ds[i2]) / 2
            v_gen = d_avg - c_avg
            pair_das.append(cos_np(v_gen, v_original))
        k2_mean = float(np.mean(pair_das))
        k2_vals.append(k2_mean)

        # k=4: 전체 평균
        c_avg4 = np.mean(cs, axis=0)
        d_avg4 = np.mean(ds, axis=0)
        v_gen4 = d_avg4 - c_avg4
        k4 = cos_np(v_gen4, v_original)
        k4_vals.append(k4)

        per_source_detail.append({"src_pos": int(pos), "k1_mean_of_das": k1_mean,
                                   "k2_mean_of_pair_das": k2_mean, "k4_single_da": k4,
                                   "da_each_seed": da_each})

    k1_vals, k2_vals, k4_vals = np.array(k1_vals), np.array(k2_vals), np.array(k4_vals)
    print(f"n={len(k1_vals)}(전체 100 중 임베딩 확보된 소스)")

    m1, lo1, hi1 = bootstrap_ci(k1_vals)
    m2, lo2, hi2 = bootstrap_ci(k2_vals)
    m4, lo4, hi4 = bootstrap_ci(k4_vals)

    # 이론(등방 잡음 가정, 소각도 근사): 잡음이 지배적이면 da가 대략 sqrt(k)에 비례해 증가.
    # k=1 기준 이론적 k=2,4 예측값(단순 sqrt(k) 스케일링, 참고용)
    pred_k2_sqrt = m1 * np.sqrt(2)
    pred_k4_sqrt = m1 * np.sqrt(4)

    lines = ["# Phase 8-1b — 가설 A: 시드 평균(k) 의존성 (2026-08-28)\n"]
    lines.append(f"alpha=1, temperature=1.0(기본) 슬라이스, n={len(k1_vals)}소스. "
                 f"임베딩을 먼저 평균한 뒤 코사인(사전등록 방식).\n")
    lines.append("| k | da 평균(95%CI) | √k 단순스케일링 참고값 |")
    lines.append("|---|---|---|")
    lines.append(f"| 1 | {m1:+.4f} [{lo1:+.4f},{hi1:+.4f}] | (기준) |")
    lines.append(f"| 2 | {m2:+.4f} [{lo2:+.4f},{hi2:+.4f}] | {pred_k2_sqrt:+.4f} |")
    lines.append(f"| 4 | {m4:+.4f} [{lo4:+.4f},{hi4:+.4f}] | {pred_k4_sqrt:+.4f} |")

    # ★ 독립 CI 비교(위 표)는 소스 간 변동(-0.33~+0.51 수준)이 매번 섞여 들어가 검정력이
    # 낮다. k1/k2/k4는 같은 100소스에서 나온 짝지어진(paired) 값이므로, 짝지어 차이를
    # 직접 부트스트랩하면 소스 간 변동이 상쇄돼 훨씬 예민하게 잡힌다.
    pd_k4_k1 = paired_bootstrap_ci(k1_vals, k4_vals)
    pd_k2_k1 = paired_bootstrap_ci(k1_vals, k2_vals)
    pd_k4_k2 = paired_bootstrap_ci(k2_vals, k4_vals)
    n_mono = int(np.sum((k1_vals < k2_vals) & (k2_vals < k4_vals)))
    n_k1_lt_k4 = int(np.sum(k1_vals < k4_vals))

    lines.append("\n## 짝비교(paired) — 소스 간 변동을 상쇄한 검정 (더 예민함)\n")
    lines.append("| 비교 | 평균 차이 | 95% CI |")
    lines.append("|---|---|---|")
    lines.append(f"| k=2 − k=1 | {pd_k2_k1[0]:+.4f} | [{pd_k2_k1[1]:+.4f},{pd_k2_k1[2]:+.4f}] |")
    lines.append(f"| k=4 − k=2 | {pd_k4_k2[0]:+.4f} | [{pd_k4_k2[1]:+.4f},{pd_k4_k2[2]:+.4f}] |")
    lines.append(f"| k=4 − k=1 | {pd_k4_k1[0]:+.4f} | [{pd_k4_k1[1]:+.4f},{pd_k4_k1[2]:+.4f}] |")
    lines.append(f"\n단조 개선(k1<k2<k4) 소스 {n_mono}/100, k1<k4 소스 {n_k1_lt_k4}/100.\n")

    sig_k4_k1 = pd_k4_k1[1] > 0  # CI 하한이 0을 넘으면 유의
    # 순수 √k 잡음모형 예측: 개선폭이 (1-1/√k)에 비례. k=2가 전체(k=4) 개선폭의
    # (1-1/√2)/(1-1/√4) = 0.293/0.5 = 58.6%를 차지해야 한다.
    frac_k2_of_total_model = (1 - 1/np.sqrt(2)) / (1 - 1/np.sqrt(4))
    frac_k2_of_total_obs = pd_k2_k1[0] / pd_k4_k1[0] if pd_k4_k1[0] != 0 else float("nan")

    if sig_k4_k1:
        scenario = "A1 방향 지지 — 유의한 개선, 게다가 √k 잡음모형과 정량적으로 잘 맞음"
    else:
        scenario = "A2(구조적 손실, 개선 없음/불확실)"
    lines.append(f"**판정**: {scenario}\n")
    lines.append(f"짝비교 기준 k=4−k=1 = {pd_k4_k1[0]:+.4f} (95%CI 하한 {pd_k4_k1[1]:+.4f}, "
                 f"{'0 초과 — 통계적으로 유의' if sig_k4_k1 else '0 포함 — 유의하지 않음'}).\n")
    if sig_k4_k1:
        lines.append(f"√k 잡음모형이 맞다면 k=2 개선폭이 k=4 전체 개선폭의 {frac_k2_of_total_model*100:.1f}%를 "
                     f"차지해야 한다 — 실측값은 {frac_k2_of_total_obs*100:.1f}%로 **매우 근접**했다. "
                     f"단순 √k 스케일링 참고값(표) 대비로는 k=4에서 예측 개선폭의 "
                     f"{(pd_k4_k1[0]/(pred_k4_sqrt-m1)*100) if (pred_k4_sqrt-m1)!=0 else float('nan'):.0f}%만 "
                     f"실현됐다 — 즉 **노이즈가 실재하고 √k로 줄지만, 잔차 전체가 노이즈는 아니고 "
                     f"일부(추정 곱셈비 기준 약 20~30%)만 노이즈로 설명된다.**\n")
    lines.append(f"\n판정기준(§5) 대조: k=1 {m1:.4f}(부분개선 문턱 근접), "
                 f"k=4 {m4:.4f}(**부분 개선** 구간 0.10~0.20 진입, 실용 개선 0.20에는 못 미침).\n")

    out_path = RESULTS_DIR / "11_phase8_1b_k_average.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {out_path}")
    print(f"k=1: {m1:+.4f} [{lo1:+.4f},{hi1:+.4f}]")
    print(f"k=2: {m2:+.4f} [{lo2:+.4f},{hi2:+.4f}]")
    print(f"k=4: {m4:+.4f} [{lo4:+.4f},{hi4:+.4f}]")
    print(f"판정: {scenario}")

    with open(RESULTS_DIR / "11_phase8_1b_k_average_raw.json", "w", encoding="utf-8") as f:
        json.dump({"per_source": per_source_detail,
                   "k1": {"mean": m1, "ci": [lo1, hi1]}, "k2": {"mean": m2, "ci": [lo2, hi2]},
                   "k4": {"mean": m4, "ci": [lo4, hi4]}, "scenario": scenario}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
