# -*- coding: utf-8 -*-
"""Phase 0.5 — pedalboard API 실측: 무효과 값 검증 + 파라미터 상호 무효화 스캔.

사용자 지시서 11차 §4 대응:
  1. 대상 필터의 실제 파라미터 시그니처 전수 출력 (코드 앞부분, 텍스트로 산출)
  2. 각 파라미터의 "무효과 값"을 실측 확인 (dry 대비 cos > 0.9999)
  3. 파라미터 간 무효화 관계 스캔 (유한차분이 정확히 0이 되는 조합)
  4. freeze_mode류 이산 스위치는 무효화 값 고정하고 기록

렌더링 없이(캐시 없이) 소수 소스로 빠르게 끝내는 진단 스크립트다 — Phase 2용 대량
렌더링이 아니다.
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

# 패밀리 다양성을 위해 6개 소스 선택 (파일명 접두어로 대충 층화)
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


def render_with_board(y, board):
    return board(y, SR)


def main():
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)

    torch_device = embed_mod.torch.device("cpu")
    print("CLAP 로딩...")
    clap = embed_mod.load_clap(torch_device, ROOT / "ckpts")

    sources = pick_sources(6)
    print(f"소스 {len(sources)}개 선택: {[s.name for s in sources]}")
    dry_audio = {}
    for s in sources:
        y = embed_mod.load_and_preprocess(s)
        dry_audio[s.name] = y

    def embed(y):
        return embed_mod.embed_batch(clap, torch_device, [y])[0]

    print("dry 임베딩 계산...")
    dry_emb = {name: embed(y) for name, y in dry_audio.items()}

    log = {"noop_checks": [], "interaction_scan": []}

    # ---------------------------------------------------------------
    # 1. 무효과 값 실측 — 후보 (파라미터, 무효과값, 보드 생성 함수)
    # ---------------------------------------------------------------
    noop_candidates = [
        ("HighShelfFilter.gain_db=0", lambda: pb.Pedalboard([pb.HighShelfFilter(cutoff_frequency_hz=4000, gain_db=0.0, q=0.7071)])),
        ("LowShelfFilter.gain_db=0", lambda: pb.Pedalboard([pb.LowShelfFilter(cutoff_frequency_hz=400, gain_db=0.0, q=0.7071)])),
        ("PeakFilter.gain_db=0", lambda: pb.Pedalboard([pb.PeakFilter(cutoff_frequency_hz=2000, gain_db=0.0, q=0.7071)])),
        ("Distortion.drive_db=0", lambda: pb.Pedalboard([pb.Distortion(drive_db=0.0)])),
        ("Reverb.wet_level=0(dry_level=1)", lambda: pb.Pedalboard([pb.Reverb(room_size=0.5, damping=0.5, wet_level=0.0, dry_level=1.0, width=1.0, freeze_mode=0.0)])),
        ("Reverb.all_default_wet0", lambda: pb.Pedalboard([pb.Reverb(wet_level=0.0)])),
    ]

    print("\n=== 1. 무효과 값 실측 ===")
    for label, board_fn in noop_candidates:
        cosines = []
        for name, y in dry_audio.items():
            board = board_fn()
            y_out = render_with_board(y, board)
            e_out = embed(y_out)
            c = cos(dry_emb[name], e_out)
            cosines.append(c)
        min_cos = min(cosines)
        mean_cos = float(np.mean(cosines))
        passed = min_cos > 0.9999
        print(f"{label:45s} min_cos={min_cos:.8f} mean_cos={mean_cos:.8f} {'PASS' if passed else '★ FAIL'}")
        log["noop_checks"].append({
            "label": label, "min_cos": min_cos, "mean_cos": mean_cos,
            "per_source_cos": dict(zip(dry_audio.keys(), cosines)), "passed": passed,
        })

    # ---------------------------------------------------------------
    # 2. 파라미터 간 무효화 관계 스캔
    #    "게이트" 후보를 수동 지정 후 유한차분 스캔 (freeze_mode 선례 반복 방지)
    # ---------------------------------------------------------------
    print("\n=== 2. 파라미터 상호 무효화 스캔 ===")
    interaction_specs = [
        # (설명, gate_param, gate_value, tested_param, board_factory(gate_val, test_val))
        ("Reverb: wet_level=0 -> room_size 변화 무효?", "wet_level", 0.0, "room_size",
         lambda wl, rs: pb.Pedalboard([pb.Reverb(room_size=rs, damping=0.5, wet_level=wl, dry_level=1.0-wl, width=1.0, freeze_mode=0.0)])),
        ("Reverb: wet_level=0 -> damping 변화 무효?", "wet_level", 0.0, "damping",
         lambda wl, dp: pb.Pedalboard([pb.Reverb(room_size=0.5, damping=dp, wet_level=wl, dry_level=1.0-wl, width=1.0, freeze_mode=0.0)])),
        ("Reverb: freeze_mode=1 -> room_size 변화 무효?(기지 결함 재확인)", "freeze_mode", 1.0, "room_size",
         lambda fz, rs: pb.Pedalboard([pb.Reverb(room_size=rs, damping=0.5, wet_level=0.3, dry_level=0.5, width=1.0, freeze_mode=fz)])),
        ("HighShelf: gain_db=0 -> cutoff 변화 무효?", "gain_db", 0.0, "cutoff_frequency_hz",
         lambda g, fc: pb.Pedalboard([pb.HighShelfFilter(cutoff_frequency_hz=fc, gain_db=g, q=0.7071)])),
        ("HighShelf: gain_db=0 -> q 변화 무효?", "gain_db", 0.0, "q",
         lambda g, q: pb.Pedalboard([pb.HighShelfFilter(cutoff_frequency_hz=4000, gain_db=g, q=q)])),
        ("PeakFilter: gain_db=0 -> cutoff 변화 무효?", "gain_db", 0.0, "cutoff_frequency_hz",
         lambda g, fc: pb.Pedalboard([pb.PeakFilter(cutoff_frequency_hz=fc, gain_db=g, q=0.7071)])),
        ("Distortion: drive_db=0 -> 실제로 완전 무효?(진짜 no-op 아닐 수 있음)", "drive_db", 0.0, "drive_db_probe",
         lambda d, _: pb.Pedalboard([pb.Distortion(drive_db=d)])),
    ]

    test_source_name = list(dry_audio.keys())[0]
    y_test = dry_audio[test_source_name]
    e_dry_test = dry_emb[test_source_name]

    for desc, gate_p, gate_v, test_p, board_fn in interaction_specs:
        test_values = [0.0, 0.3, 0.7, 1.0] if test_p not in ("cutoff_frequency_hz",) else [500, 2000, 6000, 8000]
        deltas = []
        for tv in test_values:
            try:
                board = board_fn(gate_v, tv)
            except Exception as e:
                print(f"  [skip: {e}]")
                continue
            y_out = render_with_board(y_test, board)
            e_out = embed(y_out)
            c = cos(e_dry_test, e_out)
            deltas.append((tv, c))
        cos_values = [c for _, c in deltas]
        spread = max(cos_values) - min(cos_values) if cos_values else None
        is_nullified = spread is not None and spread < 1e-4
        print(f"{desc}")
        print(f"    {gate_p}={gate_v}, {test_p} 스윕 결과 cos 값: {[f'{c:.6f}' for _, c in deltas]}  spread={spread:.2e}  {'★ 무효화 확인' if is_nullified else '정상(변화함)'}")
        log["interaction_scan"].append({
            "description": desc, "gate_param": gate_p, "gate_value": gate_v, "test_param": test_p,
            "sweep": deltas, "spread": spread, "nullified": is_nullified,
        })

    with open(OUT_RESULTS / "11_phase0_api_audit.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {OUT_RESULTS / '11_phase0_api_audit.json'}")


if __name__ == "__main__":
    main()
