# -*- coding: utf-8 -*-
"""Phase 2 산출 — 사용자 지시 §F. 축마다 (a)변위 (b)JND (c)윈도우 R² (d)곡률을 낸다.

기준점은 e(theta_min)이며 e_bypass는 별도 insertion_cost로만 쓴다(§2 분리 유지).
윈도우 R²는 폭 w x {하위/중위/상위 1/3} 표로 낸다 — 단일 R² 열 금지(지시 마지막 문단).
gn6 축 6개는 400소스만 있으므로 별도 절에서 N을 명시하고 주축과 직접 비교하지 않는다.
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

render_mod = import_module("11_phase2_render")

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "out" / "caches"
RESULTS_DIR = ROOT / "out" / "results"

SEED = 0
N_BOOT = 2000
N_MIN_WINDOW = 5000
WIDTH_FRACS = [0.20, 0.40, 0.60, 0.80, 1.00]
POSITIONS = ["하위1/3", "중위1/3", "상위1/3"]

EXT_AXES = [
    "distortion_drive_db", "reverb_wet_level", "reverb_room_size", "reverb_damping", "reverb_width",
    "highshelf_gain", "lowshelf_gain", "peak_gain",
    "highshelf_cutoff_gp6", "lowshelf_cutoff_gp6", "peak_cutoff_gp6",
    "highshelf_q_gp6", "lowshelf_q_gp6", "peak_q_gp6",
    "eq_cascade_intensity",
]
NULL_AXES = ["null_12k_gain", "null_15k_gain"]
GN6_AXES = ["highshelf_cutoff_gn6", "lowshelf_cutoff_gn6", "peak_cutoff_gn6",
            "highshelf_q_gn6", "lowshelf_q_gn6", "peak_q_gn6"]
EQ_GAIN_AXES = ["highshelf_gain", "lowshelf_gain", "peak_gain", "null_12k_gain", "null_15k_gain"]


def load_concat(axis_name):
    base = np.load(CACHE_DIR / f"11_phase2_{axis_name}.npz")
    ext_path = CACHE_DIR / f"11_phase2ext_{axis_name}.npz"
    theta_raw = base["theta_raw"]
    if ext_path.exists():
        ext = np.load(ext_path)
        assert np.allclose(theta_raw, ext["theta_raw"])
        emb = np.concatenate([base["embeddings"], ext["embeddings"]], axis=0)
        src_id = np.concatenate([base["src_id"], ext["src_id"]])
    else:
        emb = base["embeddings"]
        src_id = base["src_id"]
    order = np.argsort(src_id)
    return emb[order], theta_raw, src_id[order]


def theta_min_index(axis_name, theta_raw):
    if axis_name in EQ_GAIN_AXES:
        return int(np.argmin(np.abs(theta_raw)))
    return 0


def cos_rows(a, b):
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return num / den


def bootstrap_ci(x, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(x)
    boots = np.array([np.mean(x[rng.randint(0, n, n)]) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def build_null_floor(null_axis_data):
    """(displacement 풀 분포 95백분위, JND용 인접스텝 풀 분포 95백분위)."""
    disp_pool, jnd_pool = [], []
    for axis_name, (emb, theta_raw, src_id) in null_axis_data.items():
        idx0 = theta_min_index(axis_name, theta_raw)
        for li in range(emb.shape[1]):
            disp_pool.append(1.0 - cos_rows(emb[:, idx0, :], emb[:, li, :]))
        for i in range(emb.shape[1] - 1):
            jnd_pool.append(np.linalg.norm(emb[:, i + 1, :] - emb[:, i, :], axis=-1))
    disp_pool = np.concatenate(disp_pool)
    jnd_pool = np.concatenate(jnd_pool)
    return float(np.percentile(disp_pool, 95)), float(np.percentile(jnd_pool, 95))


def displacement_curve(emb, theta_raw, idx0):
    n_levels = emb.shape[1]
    rows = []
    for li in range(n_levels):
        d = 1.0 - cos_rows(emb[:, idx0, :], emb[:, li, :])
        lo, hi = bootstrap_ci(d)
        rows.append({"theta": float(theta_raw[li]), "mean_d": float(d.mean()), "ci_lo": lo, "ci_hi": hi})
    return rows


def jnd_curve(emb, theta_raw, null_jnd_p95):
    n_levels = emb.shape[1]
    first_clear = None
    rows = []
    for i in range(n_levels - 1):
        delta = np.linalg.norm(emb[:, i + 1, :] - emb[:, i, :], axis=-1)
        lo, hi = bootstrap_ci(delta)
        clears = lo > null_jnd_p95
        if clears and first_clear is None:
            first_clear = i
        rows.append({"theta_from": float(theta_raw[i]), "theta_to": float(theta_raw[i + 1]),
                     "mean_delta": float(delta.mean()), "ci_lo": lo, "clears_null": bool(clears)})
    return rows, first_clear


def windowed_r2_table(emb, theta_raw, src_id):
    n_levels = emb.shape[1]
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
            lo_bound, hi_bound = center - w_span / 2.0, center + w_span / 2.0
            lo_bound, hi_bound = max(lo_bound, t_lo), min(hi_bound, t_hi)
            level_mask = (theta_raw >= lo_bound - 1e-9) & (theta_raw <= hi_bound + 1e-9)
            level_idx = np.where(level_mask)[0]
            if len(level_idx) < 3:
                row[pos_name] = {"n": 0, "r2": None, "note": "격자점 부족"}
                continue
            X = emb[:, level_idx, :].reshape(-1, 512)
            theta_sub = theta_raw[level_idx]
            theta_norm = (theta_sub - theta_sub.min()) / (theta_sub.max() - theta_sub.min() + 1e-12)
            y = np.tile(theta_norm, emb.shape[0])
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


def curvature_summary(emb):
    n_levels = emb.shape[1]
    kappas = []
    for i in range(1, n_levels - 1):
        k = np.linalg.norm(emb[:, i + 1, :] - 2 * emb[:, i, :] + emb[:, i - 1, :], axis=-1)
        kappas.append(k.mean())
    kappas = np.array(kappas)
    max_i = int(np.argmax(kappas)) + 1  # +1: 원래 인덱스 오프셋 보정
    third = n_levels // 3
    region = "하위1/3" if max_i < third else ("중위1/3" if max_i < 2 * third else "상위1/3")
    return {"mean_kappa": float(kappas.mean()), "max_kappa": float(kappas.max()),
            "max_at_theta_idx": max_i, "max_region": region, "curve": kappas.round(6).tolist()}


def format_axis_report(axis_name, emb, theta_raw, src_id, null_disp_p95, null_jnd_p95, n_label):
    idx0 = theta_min_index(axis_name, theta_raw)
    lines = [f"### {axis_name}  (N={n_label}, theta_min={theta_raw[idx0]:.3g})\n"]

    disp = displacement_curve(emb, theta_raw, idx0)
    max_d = max(r["mean_d"] for r in disp)
    lines.append(f"**(a) 변위**: 최대 mean_d={max_d:.4f} (널 바닥 95백분위={null_disp_p95:.2e}). "
                 f"theta={disp[0]['theta']:.2g}→{disp[-1]['theta']:.2g} 전 구간 요약 — "
                 f"시작 {disp[0]['mean_d']:.2e}, 중앙 {disp[len(disp)//2]['mean_d']:.4f}, "
                 f"끝 {disp[-1]['mean_d']:.4f}\n")

    jnd_rows, first_clear = jnd_curve(emb, theta_raw, null_jnd_p95)
    if first_clear is not None:
        jnd_theta = jnd_rows[first_clear]["theta_to"]
        lines.append(f"**(b) JND**: theta_min에서 첫 번째로 널 바닥(p95={null_jnd_p95:.4f})을 넘는 "
                     f"지점 = theta={jnd_theta:.3g} (格자 {first_clear+1}번째 스텝)\n")
    else:
        lines.append(f"**(b) JND**: 전 구간에서 널 바닥을 넘는 스텝 없음 — 이 축은 인접-스텝 "
                     f"해상도로는 측정 불가\n")

    r2t = windowed_r2_table(emb, theta_raw, src_id)
    lines.append("**(c) 윈도우 R²**\n")
    lines.append("| 폭 | " + " | ".join(POSITIONS) + " |")
    lines.append("|---|" + "---|" * len(POSITIONS))
    for w in WIDTH_FRACS:
        row = r2t[w]
        cells = []
        if "전체범위" in row:
            v = row["전체범위"]
            cell = f"{v['r2']:.4f} (N={v['n']})" if v["r2"] is not None else f"— ({v['note']}, N={v['n']})"
            cells = [cell] * len(POSITIONS)
        else:
            for pos in POSITIONS:
                v = row.get(pos, {"n": 0, "r2": None, "note": "?"})
                cells.append(f"{v['r2']:.4f} (N={v['n']})" if v["r2"] is not None else f"— ({v['note']}, N={v['n']})")
        lines.append(f"| {int(w*100)}% | " + " | ".join(cells) + " |")
    lines.append("")

    kap = curvature_summary(emb)
    lines.append(f"**(d) 곡률**: mean={kap['mean_kappa']:.4f}, max={kap['max_kappa']:.4f} "
                 f"({kap['max_region']}에서 최대)\n")

    return "\n".join(lines), {
        "axis": axis_name, "n": n_label, "displacement": disp, "jnd": jnd_rows,
        "windowed_r2": {str(w): v for w, v in r2t.items()}, "curvature": kap,
    }


def main():
    lines = ["# Phase 2 산출 — 변위·JND·윈도우 R²·곡률 (사용자 지시 §F)\n"]
    lines.append("기준점은 `e(theta_min)`(§2 분리 유지, bypass는 insertion_cost로만 별도 보고됨 — "
                 "`11_phase1.md §2.2`, `11_phase2_integrity.md` 참고). 1,200소스(400+800) 캐시 "
                 "사용, gn6 6축만 400소스(확장 안 함).\n")

    raw_dump = {}

    null_axis_data = {a: load_concat(a) for a in NULL_AXES}
    null_disp_p95, null_jnd_p95 = build_null_floor(null_axis_data)
    lines.append(f"**널 바닥**: displacement 95백분위={null_disp_p95:.3e}, "
                 f"JND(인접스텝 L2) 95백분위={null_jnd_p95:.4f} (null_12k+15k, 1200소스, 25레벨 풀링)\n")

    lines.append("## 주축 (1,200소스)\n")
    for axis_name in EXT_AXES:
        emb, theta_raw, src_id = load_concat(axis_name)
        text, raw = format_axis_report(axis_name, emb, theta_raw, src_id, null_disp_p95, null_jnd_p95, 1200)
        lines.append(text)
        raw_dump[axis_name] = raw
        print(f"완료: {axis_name}")

    lines.append("## 널 축 자체 (참고, 1,200소스)\n")
    for axis_name in NULL_AXES:
        emb, theta_raw, src_id = null_axis_data[axis_name]
        text, raw = format_axis_report(axis_name, emb, theta_raw, src_id, null_disp_p95, null_jnd_p95, 1200)
        lines.append(text)
        raw_dump[axis_name] = raw
        print(f"완료: {axis_name}")

    lines.append("## gn6 보조 축 (★ 400소스만 — 주축과 직접 비교 금지, 부스트/컷 비대칭 점검용)\n")
    for axis_name in GN6_AXES:
        emb, theta_raw, src_id = load_concat(axis_name)  # ext 없으니 400 그대로
        text, raw = format_axis_report(axis_name, emb, theta_raw, src_id, null_disp_p95, null_jnd_p95, 400)
        lines.append(text)
        raw_dump[axis_name] = raw
        print(f"완료: {axis_name}")

    out_md = RESULTS_DIR / "11_phase2_doseresponse.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {out_md}")

    with open(RESULTS_DIR / "11_phase2_doseresponse_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_dump, f, indent=2, ensure_ascii=False)
    print(f"저장: {RESULTS_DIR / '11_phase2_doseresponse_raw.json'}")


if __name__ == "__main__":
    main()
