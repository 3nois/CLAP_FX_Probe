"""9차 Phase 3-3R-3 — 블라인드 청취 세트 구성 (층화 추출).

Phase 3-3R-1의 150쌍(directional_agreement 계산 완료)에서 상/중/하 3분위 각 8쌍씩
24쌍을 뽑는다. 각 층 안에서 이펙트·패밀리가 고르게 섞이도록 이펙트별로 먼저
할당량을 배분하고, 그 안에서 패밀리 중복을 피하며 뽑는다. 이렇게 하면
"지표가 높은 쌍이 실제로 더 잘 들리는가"까지 덤으로 검증된다.

조건(c/d)을 A/B 어느 쪽에 배치할지는 시드 고정 무작위로 정하고, blind_manifest.json
에는 조건 정보를 절대 넣지 않는다(answer_key.json에만 정답을 남긴다).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
N_PER_STRATUM = 8
BUILD_SEED = 7


def allocate_by_effect(rows, n_pick, effects, seed):
    """이펙트별로 n_pick을 최대한 고르게 배분하고, 각 배분 안에서 패밀리 중복을 피해 뽑는다."""
    rng = np.random.RandomState(seed)
    by_effect = {e: [r for r in rows if r["effect"] == e] for e in effects}
    base = n_pick // len(effects)
    rem = n_pick - base * len(effects)
    quota = {e: base for e in effects}
    for e in rng.permutation(effects)[:rem]:
        quota[e] += 1

    picked = []
    used_families_global = set()
    for e in effects:
        pool = by_effect[e][:]
        rng.shuffle(pool)
        q = min(quota[e], len(pool))
        # 패밀리 중복을 피해 우선 선택, 부족하면 나머지로 채움
        preferred = [r for r in pool if r["family"] not in used_families_global]
        chosen = []
        for r in preferred:
            if len(chosen) >= q:
                break
            chosen.append(r)
            used_families_global.add(r["family"])
        if len(chosen) < q:
            remaining = [r for r in pool if r not in chosen]
            for r in remaining:
                if len(chosen) >= q:
                    break
                chosen.append(r)
        picked.extend(chosen)
    return picked


def main():
    parser = argparse.ArgumentParser(description="9차 Phase 3-3R-3 — 블라인드 청취 세트 구성")
    parser.add_argument("--results", type=str, default="out/results/results_9_phase3_3R.json")
    parser.add_argument("--audio-dir", type=str, default="out/audio")
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    with open(args.results) as f:
        r = json.load(f)
    rows = r["rows"]
    effects = sorted(set(row["effect"] for row in rows))

    da = np.array([row["directional_agreement"] for row in rows])
    order = np.argsort(da)
    n = len(rows)
    third = n // 3
    strata = {
        "low": [rows[i] for i in order[:third]],
        "mid": [rows[i] for i in order[third:2 * third]],
        "high": [rows[i] for i in order[2 * third:]],
    }
    print(f"층별 개수: low={len(strata['low'])} mid={len(strata['mid'])} high={len(strata['high'])}")
    print(f"층 경계 directional_agreement: low_max={da[order[third-1]]:.4f}, "
          f"mid_range=[{da[order[third]]:.4f},{da[order[2*third-1]]:.4f}], high_min={da[order[2*third]]:.4f}")

    audio_dir = Path(args.audio_dir)
    blind_dir = audio_dir / "blind"
    blind_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    for stratum_name in ["high", "mid", "low"]:
        picked = allocate_by_effect(strata[stratum_name], N_PER_STRATUM, effects, seed=BUILD_SEED)
        for row in picked:
            selected.append((stratum_name, row))

    print(f"\n선정된 쌍 {len(selected)}개 (층별 {N_PER_STRATUM}개씩)")

    rng = np.random.RandomState(BUILD_SEED + 1)
    order_idx = rng.permutation(len(selected))  # 쌍 제시 순서도 무작위
    selected = [selected[i] for i in order_idx]

    manifest_pairs = []
    answer_pairs = []
    for pair_id, (stratum, row) in enumerate(selected, start=1):
        c_is_a = bool(rng.randint(0, 2))
        pid_str = f"{pair_id:02d}"
        fname_a = f"blind_{pid_str}_A.wav"
        fname_b = f"blind_{pid_str}_B.wav"
        src_c = audio_dir / row["wav_c"]
        src_d = audio_dir / row["wav_d"]
        if c_is_a:
            shutil.copyfile(src_c, blind_dir / fname_a)
            shutil.copyfile(src_d, blind_dir / fname_b)
            correct = "B"  # d(=더 dry) 가 B
        else:
            shutil.copyfile(src_d, blind_dir / fname_a)
            shutil.copyfile(src_c, blind_dir / fname_b)
            correct = "A"  # d 가 A

        manifest_pairs.append({"id": pair_id, "a": fname_a, "b": fname_b})
        answer_pairs.append({
            "id": pair_id, "correct": correct, "src_pos": row["src_pos"], "family": row["family"],
            "effect": row["effect"], "stratum": stratum, "directional_agreement": row["directional_agreement"],
        })

    with open(blind_dir / "blind_manifest.json", "w") as f:
        json.dump({"pairs": manifest_pairs}, f, indent=2, ensure_ascii=False)
    with open(blind_dir / "answer_key.json", "w") as f:
        json.dump({"pairs": answer_pairs}, f, indent=2, ensure_ascii=False)

    print(f"\n저장: {blind_dir/'blind_manifest.json'}, {blind_dir/'answer_key.json'}")
    print(f"wav 48개 복사 완료: {blind_dir}")
    print("\n층별/이펙트별 구성:")
    import collections
    ctr = collections.Counter((a["stratum"], a["effect"]) for a in answer_pairs)
    for k in sorted(ctr):
        print(f"  {k}: {ctr[k]}")
    fam_ctr = collections.Counter((a["stratum"], a["family"]) for a in answer_pairs)
    print("\n층별 패밀리 분포:")
    for k in sorted(fam_ctr):
        print(f"  {k}: {fam_ctr[k]}")


if __name__ == "__main__":
    main()
