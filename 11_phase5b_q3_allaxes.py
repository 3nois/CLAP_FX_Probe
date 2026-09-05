# -*- coding: utf-8 -*-
"""Phase 5b — Q3(within/between/gap)을 1,200소스 캐시가 있는 전 주축으로 확장.

원 Phase 5 는 사전등록에서 "21개 주축 전부를 돌리면 계산량이 지나치게 크다"는
이유로 5축으로 축소했다. 이 근거는 Q4~Q6(신경망 학습·오디오 생성)에는 타당하나
**Q3 에는 해당하지 않는다** — Q3 은 캐시된 임베딩의 코사인 계산과 부트스트랩뿐이라
전 축을 돌려도 수 분이다. 축 선택이 네 질문에 일괄 적용되면서 함께 축소되었다.

★ 사후 확장이므로 원 5축 결과를 대체하지 않고 병기한다.
★ gn6 계열 6축은 1,200소스 캐시가 없어(400 만 존재) 제외한다.

출력: out/results/11_phase5b_q3_allaxes.md / .json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

fam_mod = import_module("20_family_cosine_oat")

ROOT = Path(__file__).resolve().parent
CACHE, RESULTS = ROOT / "out" / "caches", ROOT / "out" / "results"
SEED, N_BOOT = 0, 300

ORIGINAL_5 = ["distortion_drive_db", "reverb_room_size",
              "highshelf_gain", "lowshelf_gain", "peak_gain"]
ADDED = ["reverb_wet_level", "reverb_damping", "reverb_width",
         "eq_cascade_intensity",
         "highshelf_cutoff_gp6", "highshelf_q_gp6",
         "lowshelf_cutoff_gp6", "lowshelf_q_gp6",
         "peak_cutoff_gp6", "peak_q_gp6"]
INTERVALS = [("전범위", 0, 24), ("하위1/3", 0, 8), ("중위1/3", 8, 16), ("상위1/3", 16, 24)]


def families_1200():
    a = json.loads((RESULTS / "11_phase2_sources.json").read_text(encoding="utf-8"))["sources"]
    b = json.loads((RESULTS / "11_phase2_sources_ext.json").read_text(encoding="utf-8"))["sources"]
    return np.array([s["family"] for s in sorted(a + b, key=lambda z: z["src_id"])])


def load_axis(axis):
    base = np.load(CACHE / f"11_phase2_{axis}.npz")
    ext = np.load(CACHE / f"11_phase2ext_{axis}.npz")
    emb = np.concatenate([base["embeddings"], ext["embeddings"]])
    sid = np.concatenate([base["src_id"], ext["src_id"]])
    o = np.argsort(sid)
    return emb[o].astype(np.float64), sid[o]


PARTIAL = None  # main 에서 설정


def main():
    """호출당 시간 제한이 있어 재개 가능하게 만든다.

    부분 결과를 out/results/11_phase5b_partial.json 에 축 단위로 누적하고,
    이미 끝난 축은 건너뛴다. 전 축이 끝나면 최종 문서를 쓴다.
    인자로 이번 호출에서 처리할 최대 축 수를 줄 수 있다(기본 6).
    """
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    partial_path = RESULTS / "11_phase5b_partial.json"
    done = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else {}

    fam = families_1200()
    t0 = time.time()
    axes = ORIGINAL_5 + ADDED
    todo = [a for a in axes if a not in done][:budget]

    for ai, axis in enumerate(todo, 1):
        emb, _ = load_axis(axis)
        out = []
        for name, i0, i1 in INTERVALS:
            # 결함 18 준수 — 차분 후 정규화 (끝점 각각 정규화 금지)
            v = emb[:, i1, :] - emb[:, i0, :]
            U = fam_mod.unit(v)
            w, b, g = fam_mod.bootstrap_within_between(U, fam, n_boot=N_BOOT, seed=SEED)
            ci = fam_mod.ci95(g)
            out.append({
                "axis": axis, "interval": name,
                "is_original_5": axis in ORIGINAL_5,
                "within_mean": float(w.mean()), "between_mean": float(b.mean()),
                "gap_mean": float(g.mean()), "gap_ci": [float(ci[0]), float(ci[1])],
                "verdict": "within > between" if ci[0] > 0 else "within ≈ between",
            })
        done[axis] = out
        partial_path.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [{len(done)}/{len(axes)}] {axis}  ({time.time()-t0:.0f}s)", flush=True)

    remaining = [a for a in axes if a not in done]
    if remaining:
        print(f"\n남은 축 {len(remaining)}개: {remaining}")
        print("같은 명령을 다시 실행하면 이어서 진행합니다.")
        return

    rows = [r for a in axes for r in done[a]]
    n_pos = sum(1 for r in rows if r["gap_ci"][0] > 0)
    added = [r for r in rows if not r["is_original_5"]]
    n_pos_added = sum(1 for r in added if r["gap_ci"][0] > 0)

    L = ["# Phase 5b — Q3 전 주축 확장 (사후 분석)\n",
         "원 Phase 5 는 계산량을 이유로 5축으로 축소했으나, 그 근거는 Q4~Q6 에만",
         "해당한다. Q3 은 캐시 전용이라 전 축을 돌려도 수 분이다.",
         "**원 5축 결과를 대체하지 않고 병기한다.**\n",
         f"- 축 {len(axes)}개 × 구간 {len(INTERVALS)}개 = **{len(rows)}조합** "
         f"(원 설계는 5 × 4 = 20조합)",
         f"- gn6 계열 6축은 1,200소스 캐시가 없어 제외",
         f"- 부트스트랩 {N_BOOT}회, seed {SEED}, 소요 {time.time()-t0:.0f}초\n",
         "## 판정\n", "```",
         f"gap CI 하한 > 0 인 조합    {n_pos} / {len(rows)}",
         f"  이 중 신규 추가분          {n_pos_added} / {len(added)}",
         "```\n",
         "## 전체\n",
         "| 축 | 구간 | within | between | gap | gap 95% CI | 원5축 |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['axis']} | {r['interval']} | {r['within_mean']:.4f} | "
                 f"{r['between_mean']:.4f} | **{r['gap_mean']:.4f}** | "
                 f"[{r['gap_ci'][0]:.4f}, {r['gap_ci'][1]:.4f}] | "
                 f"{'○' if r['is_original_5'] else ''} |")

    g_all = [r["gap_mean"] for r in rows]
    g_org = [r["gap_mean"] for r in rows if r["is_original_5"]]
    g_add = [r["gap_mean"] for r in added]
    L += ["", "## gap 분포 비교\n", "```",
          f"원 5축 20조합      {min(g_org):.4f} ~ {max(g_org):.4f}   (중앙값 {np.median(g_org):.4f})",
          f"신규 10축 40조합   {min(g_add):.4f} ~ {max(g_add):.4f}   (중앙값 {np.median(g_add):.4f})",
          f"전체 {len(rows)}조합      {min(g_all):.4f} ~ {max(g_all):.4f}   (중앙값 {np.median(g_all):.4f})",
          "```\n"]

    (RESULTS / "11_phase5b_q3_allaxes.md").write_text("\n".join(L), encoding="utf-8")
    (RESULTS / "11_phase5b_q3_allaxes.json").write_text(
        json.dumps({"seed": SEED, "n_boot": N_BOOT, "rows": rows}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    print(f"\ngap CI 하한 > 0 : {n_pos}/{len(rows)}  (신규분 {n_pos_added}/{len(added)})")
    print(f"gap 범위 전체 {min(g_all):.4f} ~ {max(g_all):.4f}")
    print(f"저장: out/results/11_phase5b_q3_allaxes.md   ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
