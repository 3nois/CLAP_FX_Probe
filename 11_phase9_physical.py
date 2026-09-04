"""CLAP FX Probe — 11_phase9_physical.py (Phase 9 §6-6: M2 물리 지표)

pedalboard 전용 재렌더(11_phase2_render.py의 축 정의·시드 그대로 import), CLAP
재계산 없음. §5-2: distortion은 THD(고조파 피크, f0=NSynth 파일명 pitch에서 유도)/
crest factor/spectral centroid/고역(>8kHz) 에너지비, reverb는 note-off(t=3.0s)
이후 tail 에너지비/decay-slope EDT proxy. 라이브러리 고유 항목당 1회만 렌더링해
캐싱(소스 1,200 x {dry, 18, 24} + test 소스 240 x {12} x 2축)하고, 질의별 재렌더는
하지 않는다.

판정: 질의 자신의 지표보다 검색된(top-1) 항목의 지표가 dry 기준값(train_idx 평균)
쪽으로 유의하게 이동했는가 — 소스 단위 대응 부트스트랩 95% CI (R1 vs 질의, R1 vs R0).
"""
import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np

r0mod = import_module("11_phase9_retrieval")
r1mod = import_module("11_phase9_r1")
m2mod = import_module("11_phase9_m2m3")
r2mod = import_module("11_phase2_render")

unit = r0mod.unit
paired_bootstrap_diff = r0mod.paired_bootstrap_diff

DIST_AXIS = "distortion_drive_db"
REV_AXIS = "reverb_room_size"
LIB_LEVELS = (18, 24)


def audio_metrics_distortion(y, sr, f0, t_end=2.9):
    n = int(t_end * sr)
    seg = y[:n]
    rms = np.sqrt(np.mean(seg ** 2) + 1e-20)
    crest = float(np.max(np.abs(seg)) / (rms + 1e-12))
    spec = np.fft.rfft(seg)
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    mag = np.abs(spec)
    mag2 = mag ** 2
    total = mag2.sum() + 1e-20
    centroid = float((freqs * mag).sum() / (mag.sum() + 1e-20))
    high_ratio = float(mag2[freqs > 8000].sum() / total)
    thd = _thd(freqs, mag, f0) if f0 else float("nan")
    return {"crest": crest, "centroid": centroid, "high_ratio": high_ratio, "thd": thd}


def _thd(freqs, mag, f0, n_harm=20, tol_hz=15.0):
    def peak_near(target):
        band = np.abs(freqs - target) <= tol_hz
        return mag[band].max() if band.any() else 0.0

    a1 = peak_near(f0)
    if a1 <= 0:
        return float("nan")
    harm_sq = sum(peak_near(k * f0) ** 2 for k in range(2, n_harm + 1) if k * f0 < freqs[-1])
    return float(np.sqrt(harm_sq) / a1)


def detect_note_off(y, sr, thresh_frac=0.05, hop_s=0.02, min_run=5, fallback=3.0):
    """dry 신호의 진폭 포락선에서 note-off 시각을 소스별로 검출한다. 고정 t=3.0s
    가정은 organ 등 지속형 계열에서만 성립(§5-2 사전 검증), mallet/guitar/keyboard
    등 감쇠형 계열은 그보다 훨씬 일찍(0.5~2.6s) note-off가 온다 — 검출 실패
    (min_run 연속 저에너지 구간을 못 찾음, 즉 끝까지 지속)시에만 fallback을 쓴다."""
    hop = int(hop_s * sr)
    n_hops = len(y) // hop
    env = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n_hops)])
    thresh = thresh_frac * env.max()
    below = env < thresh
    for i in range(len(below) - min_run):
        if below[i:i + min_run].all():
            return max(i * hop_s, 0.1)
    return fallback


def audio_metrics_reverb(y, sr, note_off):
    def rms(seg):
        return float(np.sqrt(np.mean(seg ** 2) + 1e-20)) if len(seg) else 0.0

    dur = len(y) / sr
    ref = rms(y[int(max(note_off - 0.5, 0.0) * sr):int(note_off * sr)])
    post_end = min(note_off + 1.0, dur)
    post = rms(y[int(note_off * sr):int(post_end * sr)])
    tail_ratio = post / (ref + 1e-12)

    hop = int(0.02 * sr)
    edt_end = min(note_off + 0.5, dur)
    starts = list(range(int(note_off * sr), int(edt_end * sr) - hop, hop))
    db_env = [20 * np.log10(rms(y[s:s + hop]) + 1e-8) for s in starts]
    if len(db_env) >= 2:
        t = np.arange(len(db_env)) * hop / sr
        slope = np.polyfit(t, db_env, 1)[0]
        edt_proxy = float(-10.0 / slope) if slope < -1e-6 else float("nan")
    else:
        edt_proxy = float("nan")
    return {"tail_ratio": tail_ratio, "edt_proxy": edt_proxy}


def render_all(test_set):
    sources = (json.load(open("out/results/11_phase2_sources.json"))["sources"]
               + json.load(open("out/results/11_phase2_sources_ext.json"))["sources"])
    fname_of = {s["src_id"]: s["filename"] for s in sources}
    dist_axis, rev_axis = r2mod.AXES[DIST_AXIS], r2mod.AXES[REV_AXIS]

    dry_dist, dry_rev = {}, {}
    dist_metrics = {12: {}, 18: {}, 24: {}}
    rev_metrics = {12: {}, 18: {}, 24: {}}

    t0 = time.time()
    for src_id in range(1200):
        fname = fname_of[src_id]
        y = r2mod.embed_mod.load_and_preprocess(r2mod.AUDIO_DIR / fname)
        _, pitch, _ = r2mod.parse_nsynth_filename(Path(fname))
        f0 = 440.0 * (2.0 ** ((pitch - 69) / 12.0)) if pitch is not None else None

        note_off = detect_note_off(y, r2mod.SR)
        dry_dist[src_id] = audio_metrics_distortion(y, r2mod.SR, f0)
        dry_rev[src_id] = audio_metrics_reverb(y, r2mod.SR, note_off)

        levels = LIB_LEVELS + (12,) if src_id in test_set else LIB_LEVELS
        for lvl in levels:
            wet_d = dist_axis["board_fn"](dist_axis["levels"][lvl])(y, r2mod.SR)
            dist_metrics[lvl][src_id] = audio_metrics_distortion(wet_d, r2mod.SR, f0)
            wet_r = rev_axis["board_fn"](rev_axis["levels"][lvl])(y, r2mod.SR)
            rev_metrics[lvl][src_id] = audio_metrics_reverb(wet_r, r2mod.SR, note_off)

        if src_id % 200 == 0:
            print(f"  렌더 진행 {src_id}/1200 ({time.time() - t0:.0f}s)")
    print(f"렌더+지표 계산 완료: {time.time() - t0:.0f}s")
    return dry_dist, dry_rev, dist_metrics, rev_metrics


def top1_lib_pos(query_src, q, lib, lib_src, group_of):
    sims = unit(q) @ unit(lib).T
    for i, s in enumerate(query_src):
        mask = np.isin(lib_src, list(group_of[s]))
        sims[i, mask] = -np.inf
    return np.argmax(sims, axis=1)


def lib_pos_to_score(pos_arr, lib_src, dry_m, m18, m24, key):
    out = np.empty(len(pos_arr))
    for i, pos in enumerate(pos_arr):
        s = int(lib_src[pos])
        m = dry_m if pos < 1200 else (m18 if pos < 2400 else m24)
        out[i] = m[s][key]
    return out


def main():
    bypass = r0mod.load_bypass()
    family_arr = r0mod.load_family_array()
    group_of = r0mod.load_dup_groups()
    train_idx, val_idx, test_idx = r0mod.stratified_split(family_arr, seed=r0mod.SEED)
    test_set = set(test_idx.tolist())

    dry_dist, dry_rev, dist_metrics, rev_metrics = render_all(test_set)
    r1_prior = json.load(open("out/results/11_phase9_r1.json"))

    axis_cfg = {
        DIST_AXIS: {"metrics": dist_metrics, "dry": dry_dist, "keys": ["thd", "high_ratio", "crest", "centroid"]},
        REV_AXIS: {"metrics": rev_metrics, "dry": dry_rev, "keys": ["tail_ratio", "edt_proxy"]},
    }

    results = {}
    for axis, cfg in axis_cfg.items():
        emb, theta = r0mod.load_axis(axis)
        model = r1mod.train_b2(axis, emb, bypass, train_idx, val_idx)
        lib, lib_src, lib_is_dry, lib_family = m2mod.build_m2_library(axis, emb, bypass, family_arr)

        print(f"\n=== {axis}: 물리 지표 (모든 축 CI, dry 기준값=train_idx 평균) ===")
        results[axis] = {"levels": {}}
        for lvl in r0mod.QUERY_LEVELS:
            e_wet = emb[:, lvl, :]
            v_hat = unit(r1mod.predict_direction(model, e_wet))
            alpha = r1_prior[axis][str(lvl)]["R1"]["alpha"]

            pos_r0 = top1_lib_pos(test_idx, e_wet[test_idx], lib, lib_src, group_of)
            q_r1 = unit(e_wet[test_idx] + alpha * v_hat[test_idx])
            pos_r1 = top1_lib_pos(test_idx, q_r1, lib, lib_src, group_of)

            print(f"  lvl={lvl} (alpha={alpha}):")
            results[axis]["levels"][lvl] = {"alpha": alpha, "metrics": {}}
            for key in cfg["keys"]:
                dry_ref = float(np.nanmean([cfg["dry"][s][key] for s in train_idx]))
                query_score = np.array([cfg["metrics"][lvl][s][key] for s in test_idx])
                r0_score = lib_pos_to_score(pos_r0, lib_src, cfg["dry"], cfg["metrics"][18], cfg["metrics"][24], key)
                r1_score = lib_pos_to_score(pos_r1, lib_src, cfg["dry"], cfg["metrics"][18], cfg["metrics"][24], key)

                valid = ~(np.isnan(query_score) | np.isnan(r0_score) | np.isnan(r1_score))
                d_query = np.abs(query_score[valid] - dry_ref)
                d_r0 = np.abs(r0_score[valid] - dry_ref)
                d_r1 = np.abs(r1_score[valid] - dry_ref)

                ci_vs_query = paired_bootstrap_diff(d_query, d_r1)  # >0 이면 R1이 질의보다 dry에 가까움
                ci_vs_r0 = paired_bootstrap_diff(d_r0, d_r1)  # >0 이면 R1이 R0보다 dry에 가까움

                print(f"    {key:<10} dry_ref={dry_ref:.4f}  n_valid={valid.sum()}/{len(valid)}  "
                      f"query={d_query.mean():.4f}  R0={d_r0.mean():.4f}  R1={d_r1.mean():.4f}  "
                      f"(질의-R1)CI={[round(c,4) for c in ci_vs_query]}  (R0-R1)CI={[round(c,4) for c in ci_vs_r0]}")

                results[axis]["levels"][lvl]["metrics"][key] = {
                    "dry_ref": dry_ref,
                    "n_valid": int(valid.sum()),
                    "query_dist_to_dry_mean": float(d_query.mean()),
                    "R0_dist_to_dry_mean": float(d_r0.mean()),
                    "R1_dist_to_dry_mean": float(d_r1.mean()),
                    "R1_vs_query_ci": list(ci_vs_query),
                    "R1_vs_R0_ci": list(ci_vs_r0),
                }

    with open("out/results/11_phase9_physical.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n저장: out/results/11_phase9_physical.json")


if __name__ == "__main__":
    main()
