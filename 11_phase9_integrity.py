"""CLAP FX Probe — 11_phase9_integrity.py (Phase 9 §6-1 무결성 검사)

캐시 전용, CLAP 재계산 없음. 사전 등록(out/prereg/11_phase9.md) 실행 전 확인:
  1. bypass 라이브러리(1,200) src_id 연속성
  2. highshelf 레벨12 == bypass (파이프라인 동일성, cos == 1.0)
  3. §0-1 천장 R@1/R@10 재현 (0.9150 / 0.5142 / 0.2958)
  4. §0-1 표에 없던 중간 레벨(12/18/24)의 R0 기준선 — 이후 예측 등록에 사용
재현 실패 시 종료 코드 1로 중단.
"""
import hashlib
import json
import sys

import numpy as np

AXES = ["distortion_drive_db", "reverb_room_size", "highshelf_gain"]
LIB_LEVELS = {"distortion_drive_db": 24, "reverb_room_size": 24, "highshelf_gain": 24}
EXPECTED_R1 = {"highshelf_gain": 0.9150, "reverb_room_size": 0.5142, "distortion_drive_db": 0.2958}
QUERY_LEVELS = [12, 18, 24]
TRUE_DRY_COS_MIN = 0.9999  # 결함 0.5 감사(out/results/11_phase0_ranges.md)의 PASS 기준


DUPGROUPS_PATH = "out/results/11_phase9_dupgroups.json"


def load_duplicate_groups():
    """결함 22 (NSynth 데이터셋 속성, 계측 결함 아님): 라이브러리 1,200곡 중 20곡이
    서로 다른 파일명이지만 바이트 단위로 동일한 오디오. out/results/11_phase9_dupgroups.json
    캐시가 있으면 그걸 쓰고(전 팔 공유), 없으면 MD5로 새로 만든다."""
    import os
    if os.path.exists(DUPGROUPS_PATH):
        d = json.load(open(DUPGROUPS_PATH))
        group_of = {int(k): set(v) for k, v in d["group_of_src_id"].items()}
        for i in range(1200):
            group_of.setdefault(i, {i})
        return group_of
    base = json.load(open("out/results/11_phase2_sources.json"))["sources"]
    ext = json.load(open("out/results/11_phase2_sources_ext.json"))["sources"]
    hashes = {}
    for s in base + ext:
        with open(f"nsynth-test/audio/{s['filename']}", "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        hashes.setdefault(h, []).append(s["src_id"])
    group_of = {}
    for ids in hashes.values():
        for i in ids:
            group_of[i] = set(ids)
    return group_of


def unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def load_axis(axis):
    m = np.load(f"out/caches/11_phase2_{axis}.npz", allow_pickle=True)
    x = np.load(f"out/caches/11_phase2ext_{axis}.npz", allow_pickle=True)
    emb = np.concatenate([m["embeddings"], x["embeddings"]], axis=0)
    src = np.concatenate([m["src_id"], x["src_id"]], axis=0)
    order = np.argsort(src)
    return emb[order], src[order], m["theta_raw"]


def recall_at_k(query, library, self_idx, group_of, ks=(1, 5, 10)):
    """group-aware recall: top-k에 질의 소스의 중복-오디오 그룹 구성원이 하나라도
    있으면 적중으로 센다 (결함 22 — 20/1,200 소스가 바이트 단위로 중복)."""
    sims = unit(query) @ unit(library).T  # (n_query, n_lib)
    order = np.argsort(-sims, axis=1)
    out = {}
    for k in ks:
        topk = order[:, :k]
        hit = np.array([
            bool(group_of[self_idx[i]] & set(topk[i].tolist())) for i in range(len(self_idx))
        ])
        out[k] = float(hit.mean())
    return out, np.array([sims[i, self_idx[i]] for i in range(len(self_idx))])


def main():
    fail = False

    print("=== 1. bypass 라이브러리 연속성 ===")
    b = np.load("out/caches/11_phase2_bypass.npz", allow_pickle=True)
    be = np.load("out/caches/11_phase2ext_bypass.npz", allow_pickle=True)
    bypass = np.concatenate([b["embeddings"], be["embeddings"]], axis=0)
    bypass_src = np.concatenate([b["src_id"], be["src_id"]], axis=0)
    order = np.argsort(bypass_src)
    bypass, bypass_src = bypass[order], bypass_src[order]
    ok = np.array_equal(bypass_src, np.arange(1200))
    print(f"  src_id 0..1199 연속: {ok}")
    fail |= not ok

    print("\n=== 2. highshelf 레벨12 vs bypass (파이프라인 동일성) ===")
    hs_emb, hs_src, hs_theta = load_axis("highshelf_gain")
    lvl12_val = float(hs_theta[12])
    assert np.array_equal(hs_src, np.arange(1200))
    cos12 = np.sum(unit(hs_emb[:, 12, :]) * unit(bypass), axis=-1)
    print(f"  theta[12] = {lvl12_val} dB (기대: 0.0)")
    print(f"  cos(레벨12, bypass) min={cos12.min():.8f} mean={cos12.mean():.8f}")
    ok = lvl12_val == 0.0 and cos12.min() > TRUE_DRY_COS_MIN
    fail |= not ok
    print(f"  판정: {'OK' if ok else 'FAIL'} (기준 cos_min > {TRUE_DRY_COS_MIN}, Phase 0.5 감사와 동일)")

    print("\n=== 2.5 결함 22: 중복 오디오 그룹 로드 ===")
    group_of = load_duplicate_groups()
    dup_sizes = sorted({len(g) for g in group_of.values() if len(g) > 1})
    n_dup = sum(1 for g in group_of.values() if len(g) > 1)
    print(f"  중복 그룹 크기: {dup_sizes}  (중복에 속한 소스 수: {n_dup}/1200)")

    print("\n=== 3. §0-1 천장 R@1/R@10 재현 (무처리 질의=축 최대치, group-aware recall) ===")
    axis_data = {}
    for axis in AXES:
        emb, src, theta = load_axis(axis)
        axis_data[axis] = (emb, src, theta)
        q = emb[:, LIB_LEVELS[axis], :]
        rec, cos_self = recall_at_k(q, bypass, np.arange(1200), group_of)
        print(f"  {axis:<20} theta={theta[LIB_LEVELS[axis]]:>7.3f}  cos_median={np.median(cos_self):.4f}  "
              f"R@1={rec[1]:.4f} (기대 {EXPECTED_R1[axis]:.4f})  R@10={rec[10]:.4f}")
        ok = abs(rec[1] - EXPECTED_R1[axis]) < 1e-3
        fail |= not ok
        if not ok:
            print(f"    ★ R@1 재현 실패 — 기대 {EXPECTED_R1[axis]:.4f}, 실측 {rec[1]:.4f}")

    print("\n=== 4. 중간 레벨(12/18/24) R0 기준선 (group-aware) — 사전 등록 예측용 ===")
    for axis in ["distortion_drive_db", "reverb_room_size"]:
        emb, src, theta = axis_data[axis]
        for lvl in QUERY_LEVELS:
            q = emb[:, lvl, :]
            rec, cos_self = recall_at_k(q, bypass, np.arange(1200), group_of)
            print(f"  {axis:<20} lvl={lvl:>2} theta={theta[lvl]:>7.3f}  cos_median={np.median(cos_self):.4f}  "
                  f"R@1={rec[1]:.4f}  R@5={rec[5]:.4f}  R@10={rec[10]:.4f}")

    print("\n=== 5. 패밀리 층화 카운트 (oat_emb) ===")
    o = np.load("out/caches/oat_emb.npz", allow_pickle=True)
    fam = o["instrument_family"]
    counts = {f: int((fam == f).sum()) for f in sorted(set(fam.tolist()))}
    print(f"  {counts}")
    ok = all(v == 120 for v in counts.values()) and len(counts) == 10
    fail |= not ok

    print("\n=== 6. oat_emb ↔ Phase2 소스 정렬 확인 (동일 소스 가정 검증) ===")
    # oat_emb의 reverb=room_size, distortion=drive_db 스윕이 Phase2와 같은 1,200 소스를
    # 가리키는지 확인 — dry(레벨0)들끼리, wet(레벨2)들끼리 서로 매우 높은 코사인이어야 함
    # (두 캐시가 다른 렌더 파라미터를 썼을 수 있으므로 완전한 1.0은 기대하지 않음).
    for oat_effect, phase2_axis in [("distortion", "distortion_drive_db"), ("reverb", "reverb_room_size")]:
        idx = list(o["effect_names"]).index(oat_effect)
        oat_dry = o["emb"][:, idx, 0, :]
        cos_dry = np.sum(unit(oat_dry) * unit(bypass), axis=-1)
        print(f"  {oat_effect:<12} dry(oat) vs bypass(phase2)  cos median={np.median(cos_dry):.6f}  min={cos_dry.min():.6f}")

    print(f"\n=== 최종 판정: {'FAIL — 중단' if fail else 'PASS — 6-2로 진행 가능'} ===")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
