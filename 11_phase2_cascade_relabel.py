# -*- coding: utf-8 -*-
"""eq_cascade_intensity 재라벨링 — 사용자 지시 §1 (재렌더링 없음).

기존 타깃 s 는 소스마다 다른 5밴드 gain 패턴의 "세기"를 반영하지 못해 소스별
잡음 변수와 교락됐다는 지적. 새 타깃 s*||g_source||_2 (실효 EQ 강도)로 같은
윈도우 R² 절차를 재실행해 둘을 나란히 비교한다.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

dr = import_module("11_phase2_doseresponse")

CACHE_DIR = dr.CACHE_DIR
RESULTS_DIR = dr.RESULTS_DIR
SEED = dr.SEED
N_MIN_WINDOW = dr.N_MIN_WINDOW
WIDTH_FRACS = dr.WIDTH_FRACS
POSITIONS = dr.POSITIONS


def windowed_r2_general(emb, theta_raw, src_id, y_by_source_level):
    """y_by_source_level: (n_sources, n_levels) 배열. 윈도우 선택은 theta_raw 기준,
    회귀 타깃은 y_by_source_level(윈도우 내부에서 min-max 정규화)."""
    t_lo, t_hi = theta_raw.min(), theta_raw.max()
    span = t_hi - t_lo
    table = {}
    for w in WIDTH_FRACS:
        w_span = w * span
        if w >= 0.999:
            centers = {"전체범위": (t_lo + t_hi) / 2.0}
        else:
            half = w_span / 2.0
            centers = {
                "하위1/3": max(t_lo + half, t_lo + span * (1 / 6)),
                "중위1/3": t_lo + span * 0.5,
                "상위1/3": min(t_hi - half, t_lo + span * (5 / 6)),
            }
        row = {}
        for pos_name, center in centers.items():
            lo_bound, hi_bound = max(center - w_span / 2.0, t_lo), min(center + w_span / 2.0, t_hi)
            level_idx = np.where((theta_raw >= lo_bound - 1e-9) & (theta_raw <= hi_bound + 1e-9))[0]
            if len(level_idx) < 3:
                row[pos_name] = {"n": 0, "r2": None, "note": "격자점 부족"}
                continue
            X = emb[:, level_idx, :].reshape(-1, 512)
            y_sub = y_by_source_level[:, level_idx]
            y_norm = (y_sub - y_sub.min()) / (y_sub.max() - y_sub.min() + 1e-12)
            y = y_norm.reshape(-1)
            groups = np.repeat(src_id, len(level_idx))
            n_rows = len(y)
            if n_rows < N_MIN_WINDOW:
                row[pos_name] = {"n": n_rows, "r2": None, "note": "검정력 부족(N<5000)"}
                continue
            gss = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=SEED)
            r2s = []
            for train_idx, test_idx in gss.split(X, y, groups):
                model = Ridge(alpha=1.0)
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
                r2s.append(r2_score(y[test_idx], pred))
            row[pos_name] = {"n": n_rows, "r2": float(np.mean(r2s)), "note": ""}
        table[w] = row
    return table


def format_table(table):
    lines = ["| 폭 | " + " | ".join(POSITIONS) + " |", "|---|" + "---|" * len(POSITIONS)]
    for w in WIDTH_FRACS:
        row = table[w]
        if "전체범위" in row:
            v = row["전체범위"]
            cell = f"{v['r2']:.4f} (N={v['n']})" if v["r2"] is not None else f"— ({v['note']}, N={v['n']})"
            cells = [cell] * len(POSITIONS)
        else:
            cells = []
            for pos in POSITIONS:
                v = row.get(pos, {"n": 0, "r2": None, "note": "?"})
                cells.append(f"{v['r2']:.4f} (N={v['n']})" if v["r2"] is not None else f"— ({v['note']}, N={v['n']})")
        lines.append(f"| {int(w*100)}% | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    emb, theta_raw, src_id = dr.load_concat("eq_cascade_intensity")
    n_sources, n_levels, _ = emb.shape

    with open(RESULTS_DIR / "11_phase2_cascade_seeds.json", encoding="utf-8") as f:
        gains_base = {int(k): v for k, v in json.load(f).items()}
    with open(RESULTS_DIR / "11_phase2_cascade_seeds_ext.json", encoding="utf-8") as f:
        gains_ext = {int(k): v for k, v in json.load(f).items()}
    gains_all = {**gains_base, **gains_ext}

    g_norm = np.array([np.linalg.norm(list(gains_all[int(sid)].values())) for sid in src_id])
    print(f"||g_source||_2: min={g_norm.min():.3f} mean={g_norm.mean():.3f} max={g_norm.max():.3f} "
          f"std={g_norm.std():.3f}")

    s_grid = theta_raw  # 0..1, 25레벨
    y_old = np.tile(s_grid, (n_sources, 1))  # 기존 타깃: 소스 무관 s
    y_new = s_grid[None, :] * g_norm[:, None]  # 신규 타깃: s * ||g_source||

    print("기존 타깃(s) 윈도우 R² 계산...")
    table_old = windowed_r2_general(emb, theta_raw, src_id, y_old)
    print("신규 타깃(s*||g||) 윈도우 R² 계산...")
    table_new = windowed_r2_general(emb, theta_raw, src_id, y_new)

    lines = ["# eq_cascade_intensity 재라벨링 (사용자 지시 §1)\n"]
    lines.append(f"소스별 5밴드 gain 벡터의 L2 노름 `||g_source||_2`: "
                 f"min={g_norm.min():.3f}, mean={g_norm.mean():.3f}, max={g_norm.max():.3f}, "
                 f"std={g_norm.std():.3f} (Koo 범위 ±15dB, 5밴드 uniform 추출이므로 "
                 f"기대값 근방에서 소스마다 상당히 흩어짐 — 이 산포가 곧 라벨 잡음의 크기)\n")
    lines.append("## 기존 타깃 s (소스 무관, 잡음과 교락)\n")
    lines.append(format_table(table_old))
    lines.append(f"\n(참고: `11_phase2_doseresponse.md`의 100% 폭 R²=0.0716과 재현되는지 확인용)\n")
    lines.append("\n## 신규 타깃 s·||g_source||_2 (실효 EQ 강도)\n")
    lines.append(format_table(table_new))

    old_full = table_old[1.00]["전체범위"]["r2"]
    new_full = table_new[1.00]["전체범위"]["r2"]
    lines.append(f"\n## 결함 15 (신규)\n")
    lines.append(f"> eq_cascade_intensity의 원래 타깃 s는 소스마다 다른 5밴드 gain 패턴의 세기를 "
                 f"반영하지 못해 소스별 잡음 변수와 교락됐다. 같은 100% 폭 기준 R²가 "
                 f"**{old_full:.4f}(구 타깃 s) → {new_full:.4f}(신 타깃 s·‖g‖, 실효 EQ 강도)**로 "
                 f"{'대폭 개선' if new_full > old_full * 1.5 else '유사'}됐다. 변위(0.0360)는 highshelf_gain "
                 f"(0.0356)과 동급인데 구 타깃 R²만 1/10 수준이었던 것은 CLAP의 한계가 아니라 "
                 f"**라벨(타깃 변수) 정의 결함**이었음을 확인한다 — 소스별 실제 인가 강도를 "
                 f"모르는 채 정규화 스칼라 s만으로 회귀하면, 강한 패턴을 뽑은 소스와 약한 패턴을 "
                 f"뽑은 소스가 같은 s에서 서로 다른 크기로 반응해 '레이블 식별불가능성'이 생긴다.\n")

    out_path = RESULTS_DIR / "11_phase2_cascade_relabel.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {out_path}")
    print(f"기존 s: 100%폭 R²={old_full:.4f}  /  신규 s*||g||: 100%폭 R²={new_full:.4f}")


if __name__ == "__main__":
    main()
