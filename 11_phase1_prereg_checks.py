# -*- coding: utf-8 -*-
"""Phase 1 사전 등록 보강 실측 — 사용자 4건 지시 대응.

1. OAT 기준점 비퇴화 검증: EQ cutoff/q 스윕을 gain=+-6dB에서, reverb
   room_size/damping/width 스윕을 wet_level=0.3에서 — 실제로 움직이는지 확인
2. insertion_cost 신규 측정: reverb room_size/damping/width 축의 wet_level=0.3
   기준점에서 theta_min 지점의 bypass 대비 코사인
3. eq_cascade_intensity 축 설계 퇴화 여부 확인 (소스별 고정 시드 5밴드 패턴)

Phase 0.5 스크립트와 동일한 6소스, 동일 CLAP 인스턴스 재사용 방식.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pedalboard as pb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

embed_mod = import_module("01_embed")

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "nsynth-test" / "audio"
OUT_RESULTS = ROOT / "out" / "results"
OUT_LOGS = ROOT / "out" / "logs"
SR = 48000

CANDIDATE_PREFIXES = ["bass_", "brass_", "flute_", "guitar_", "keyboard_", "vocal_", "string_", "reed_"]


def pick_sources(n=6):
    all_files = sorted(AUDIO_DIR.glob("*.wav"))
    chosen = []
    used_prefix = set()
    for f in all_files:
        for p in CANDIDATE_PREFIXES:
            if f.name.startswith(p) and p not in used_prefix:
                chosen.append(f)
                used_prefix.add(p)
                break
        if len(chosen) >= n:
            break
    return chosen


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)

    torch_device = embed_mod.torch.device("cpu")
    print("CLAP 로딩...")
    clap = embed_mod.load_clap(torch_device, ROOT / "ckpts")

    sources = pick_sources(6)
    print(f"소스 {len(sources)}개: {[s.name for s in sources]}")
    dry_audio = {s.name: embed_mod.load_and_preprocess(s) for s in sources}

    def embed(y):
        return embed_mod.embed_batch(clap, torch_device, [y])[0]

    dry_emb = {name: embed(y) for name, y in dry_audio.items()}
    names = list(dry_audio.keys())

    log = {"oat_checks": [], "insertion_cost": [], "cascade_check": []}

    # ------------------------------------------------------------------
    # 1. OAT 기준점 비퇴화 검증 — EQ cutoff/q at gain=+-6dB
    # ------------------------------------------------------------------
    print("\n=== 1. OAT 기준점 비퇴화 검증 (EQ) ===")
    eq_specs = [
        ("HighShelfFilter", pb.HighShelfFilter, "cutoff_frequency_hz", [500, 1500, 2500, 4000], 0.7071),
        ("HighShelfFilter", pb.HighShelfFilter, "q", [0.1, 0.7, 1.3, 2.0], 2000),
        ("LowShelfFilter", pb.LowShelfFilter, "cutoff_frequency_hz", [30, 90, 150, 200], 0.7071),
        ("LowShelfFilter", pb.LowShelfFilter, "q", [0.1, 0.7, 1.3, 2.0], 100),
        ("PeakFilter", pb.PeakFilter, "cutoff_frequency_hz", [200, 2000, 4000, 6000], 0.7071),
        ("PeakFilter", pb.PeakFilter, "q", [0.1, 0.7, 1.3, 2.0], 1000),
    ]
    test_name = names[0]
    y_test = dry_audio[test_name]
    e_dry_test = dry_emb[test_name]

    for filt_name, filt_cls, sweep_param, sweep_vals, fixed_other in eq_specs:
        for gain_sign, gain_val in [("+6dB", 6.0), ("-6dB", -6.0)]:
            cosines = []
            for v in sweep_vals:
                kwargs = {"gain_db": gain_val, "q": 0.7071, "cutoff_frequency_hz": fixed_other}
                kwargs[sweep_param] = v
                board = pb.Pedalboard([filt_cls(**kwargs)])
                e_out = embed(board(y_test, SR))
                cosines.append(cos(e_dry_test, e_out))
            spread = max(cosines) - min(cosines)
            nondegenerate = spread > 1e-4
            desc = f"{filt_name}.{sweep_param} sweep @ gain={gain_sign}"
            print(f"{desc:50s} spread={spread:.2e} {'OK(움직임)' if nondegenerate else '★ 퇴화 의심'}")
            log["oat_checks"].append({
                "axis": f"{filt_name}.{sweep_param}", "gain_condition": gain_sign,
                "sweep_values": sweep_vals, "cosines": cosines, "spread": spread,
                "nondegenerate": nondegenerate,
            })

    # ------------------------------------------------------------------
    # 1b. OAT 기준점 비퇴화 검증 — Reverb room_size/damping/width at wet_level=0.3
    # ------------------------------------------------------------------
    print("\n=== 1b. OAT 기준점 비퇴화 검증 (Reverb) ===")
    reverb_specs = [
        ("room_size", [0.05, 0.3, 0.6, 0.85], {"damping": 0.1, "width": 0.7}),
        ("damping", [0.0, 0.33, 0.66, 1.0], {"room_size": 0.5, "width": 0.7}),
        ("width", [0.0, 0.33, 0.66, 1.0], {"room_size": 0.5, "damping": 0.1}),
    ]
    for axis, sweep_vals, others in reverb_specs:
        cosines = []
        for v in sweep_vals:
            kw = dict(others)
            kw[axis] = v
            kw["wet_level"] = 0.3
            kw["dry_level"] = 0.7
            kw["freeze_mode"] = 0.0
            board = pb.Pedalboard([pb.Reverb(**kw)])
            e_out = embed(board(y_test, SR))
            cosines.append(cos(e_dry_test, e_out))
        spread = max(cosines) - min(cosines)
        nondegenerate = spread > 1e-4
        print(f"Reverb.{axis:10s} sweep @ wet_level=0.3   spread={spread:.2e} {'OK(움직임)' if nondegenerate else '★ 퇴화 의심'}")
        log["oat_checks"].append({
            "axis": f"Reverb.{axis}", "gain_condition": "wet_level=0.3",
            "sweep_values": sweep_vals, "cosines": cosines, "spread": spread,
            "nondegenerate": nondegenerate,
        })

    # ------------------------------------------------------------------
    # 2. insertion_cost — Reverb room_size/damping/width 축 (wet_level=0.3 기준,
    #    각 축의 theta_min 지점), bypass 대비. distortion과 wet_level 축 자체는
    #    Phase 0.5 noop_checks 재사용(재측정 불필요).
    # ------------------------------------------------------------------
    print("\n=== 2. insertion_cost (Reverb room_size/damping/width, theta_min @ wet_level=0.3) ===")
    theta_min_specs = [
        ("room_size", 0.05, {"damping": 0.1, "width": 0.7}),
        ("damping", 0.0, {"room_size": 0.5, "width": 0.7}),
        ("width", 0.0, {"room_size": 0.5, "damping": 0.1}),
    ]
    for axis, theta_min_val, others in theta_min_specs:
        cosines = []
        for name, y in dry_audio.items():
            kw = dict(others)
            kw[axis] = theta_min_val
            kw["wet_level"] = 0.3
            kw["dry_level"] = 0.7
            kw["freeze_mode"] = 0.0
            board = pb.Pedalboard([pb.Reverb(**kw)])
            e_out = embed(board(y, SR))
            cosines.append(cos(dry_emb[name], e_out))
        min_cos, mean_cos = min(cosines), float(np.mean(cosines))
        print(f"Reverb.{axis:10s} theta_min={theta_min_val}  min_cos={min_cos:.6f} mean_cos={mean_cos:.6f}")
        log["insertion_cost"].append({
            "axis": f"Reverb.{axis}", "theta_min": theta_min_val, "reference": "wet_level=0.3",
            "min_cos": min_cos, "mean_cos": mean_cos,
        })

    # ------------------------------------------------------------------
    # 3. eq_cascade_intensity 설계 퇴화 검증
    # ------------------------------------------------------------------
    print("\n=== 3. eq_cascade_intensity 퇴화 검증 (소스별 고정 시드) ===")
    band_freqs = {"low_shelf": 100.0, "first_band": 400.0, "second_band": 2000.0,
                  "third_band": 4000.0, "high_shelf": 3500.0}

    def build_cascade_board(gains, s):
        boards = [
            pb.LowShelfFilter(cutoff_frequency_hz=band_freqs["low_shelf"], gain_db=s * gains["low_shelf"], q=0.7071),
            pb.PeakFilter(cutoff_frequency_hz=band_freqs["first_band"], gain_db=s * gains["first_band"], q=0.7),
            pb.PeakFilter(cutoff_frequency_hz=band_freqs["second_band"], gain_db=s * gains["second_band"], q=0.7),
            pb.PeakFilter(cutoff_frequency_hz=band_freqs["third_band"], gain_db=s * gains["third_band"], q=0.7),
            pb.HighShelfFilter(cutoff_frequency_hz=band_freqs["high_shelf"], gain_db=s * gains["high_shelf"], q=0.7071),
        ]
        return pb.Pedalboard(boards)

    s_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    for idx, name in enumerate(names):
        rng = np.random.default_rng(42 + idx)
        gains = {b: float(rng.uniform(-15, 15)) for b in band_freqs}
        y = dry_audio[name]
        cosines = []
        for s in s_grid:
            board = build_cascade_board(gains, s)
            e_out = embed(board(y, SR))
            cosines.append(cos(dry_emb[name], e_out))
        spread = max(cosines) - min(cosines)
        s0_is_noop = abs(cosines[0] - 1.0) < 1e-4
        print(f"{name:20s} seed={42+idx} gains={ {k: round(v,1) for k,v in gains.items()} }")
        print(f"    cos@s={s_grid} = {[f'{c:.6f}' for c in cosines]}  spread={spread:.2e}  s=0 no-op:{s0_is_noop}")
        log["cascade_check"].append({
            "source": name, "seed": 42 + idx, "gains": gains, "band_freqs": band_freqs,
            "s_grid": s_grid, "cosines": cosines, "spread": spread, "s0_is_noop": s0_is_noop,
        })

    with open(OUT_RESULTS / "11_phase1_prereg_checks.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {OUT_RESULTS / '11_phase1_prereg_checks.json'}")


if __name__ == "__main__":
    main()
