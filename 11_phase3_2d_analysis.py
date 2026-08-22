# -*- coding: utf-8 -*-
"""3-2. 게이트 히트맵 + context별 R²(A|B) — out/prereg/11_phase3.md 확정 방법론.

(a) 게이트 sanity check: EQ 쌍 gain=0 행이 평평해야 함(위반시 즉시 중단)
(b) R²(A|B=b): context 를 b 로 고정한 슬라이스에서 focus 축 예측 held-out R², 13개 값
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

base = import_module("11_phase2_render")
dr = import_module("11_phase2_doseresponse")
CACHE_DIR = base.CACHE_DIR
RESULTS_DIR = base.RESULTS_DIR
FIG_DIR = base.ROOT / "out" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

PAIRS_2D = ["reverb_wet_room", "reverb_wet_damping", "reverb_room_damping",
            "highshelf_gain_cutoff", "lowshelf_gain_cutoff", "peak_gain_cutoff"]
EQ_PAIRS = ["highshelf_gain_cutoff", "lowshelf_gain_cutoff", "peak_gain_cutoff"]
SEED = 0
N_MIN = 5000


def ref_index_for_axis(axis_name, grid):
    if axis_name == "gain":
        return int(np.argmin(np.abs(grid)))
    return 0


def cos_rows(a, b):
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return num / den


def displacement_heatmap(name):
    d = np.load(CACHE_DIR / f"11_phase3_2d_{name}.npz")
    emb = d["embeddings"]
    g1, g2 = d["grid1"], d["grid2"]
    axis1, axis2 = str(d["axis1"]), str(d["axis2"])
    mean_emb = emb.mean(axis=0)
    r1 = ref_index_for_axis(axis1, g1)
    r2 = ref_index_for_axis(axis2, g2)
    e_ref = mean_emb[r1, r2]
    D = 1.0 - np.array([[np.dot(e_ref, mean_emb[i, j]) / (np.linalg.norm(e_ref) * np.linalg.norm(mean_emb[i, j]) + 1e-12)
                          for j in range(len(g2))] for i in range(len(g1))])
    return D, g1, g2, axis1, axis2, r1, r2


def gate_sanity_check(null_p95):
    lines = ["### 3-2(a) 게이트 sanity check\n"]
    all_flat = True
    for name in EQ_PAIRS:
        D, g1, g2, axis1, axis2, r1, r2 = displacement_heatmap(name)
        gain_idx = r1 if axis1 == "gain" else r2
        row = D[gain_idx, :] if axis1 == "gain" else D[:, gain_idx]
        max_d = float(np.max(row))
        flat = max_d <= null_p95 * 3  # 널 바닥의 3배까지 여유(수치 노이즈 허용)
        if not flat:
            all_flat = False
        lines.append(f"- {name}: gain=0 행 최대 변위 = {max_d:.2e} (널 바닥 95백분위={null_p95:.2e}, "
                     f"3배 허용선={null_p95*3:.2e}) — {'PASS(평평)' if flat else '★★★ FAIL(평평하지 않음)'}")
    return all_flat, "\n".join(lines)


def context_probe(name):
    d = np.load(CACHE_DIR / f"11_phase3_2d_{name}.npz")
    emb = d["embeddings"]  # (1200,13,13,512)
    g1, g2 = d["grid1"], d["grid2"]
    axis1, axis2 = str(d["axis1"]), str(d["axis2"])
    src_id = d["src_id"]
    n_context = len(g2)  # axis2를 context로 고정, axis1(focus)을 예측

    results = []
    for b in range(n_context):
        X = emb[:, :, b, :].reshape(-1, 512)  # (1200*13, 512)
        theta = g1
        theta_norm = (theta - theta.min()) / (theta.max() - theta.min() + 1e-12)
        y = np.tile(theta_norm, emb.shape[0])
        groups = np.repeat(src_id, len(theta))
        n_rows = len(y)
        if n_rows < N_MIN:
            results.append({"context_val": float(g2[b]), "n": n_rows, "r2": None, "note": "검정력 부족(N<5000)"})
            continue
        gss = GroupShuffleSplit(n_splits=3, test_size=0.1, random_state=SEED)
        r2s = []
        for train_idx, test_idx in gss.split(X, y, groups):
            n_train = int(len(train_idx) * 0.8 / 0.9)
            train_sub = train_idx[:n_train]
            model = Ridge(alpha=1.0)
            model.fit(X[train_sub], y[train_sub])
            pred = model.predict(X[test_idx])
            r2s.append(r2_score(y[test_idx], pred))
        results.append({"context_val": float(g2[b]), "n": n_rows, "r2": float(np.mean(r2s)), "note": ""})
    return {"pair": name, "focus": axis1, "context": axis2, "probe": results}


def plot_2d(name, D, g1, g2, axis1, axis2, probe_result):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    im = axes[0].imshow(D, origin="lower", aspect="auto", cmap="viridis",
                         extent=[g2.min(), g2.max(), g1.min(), g1.max()])
    axes[0].set_xlabel(axis2); axes[0].set_ylabel(axis1)
    axes[0].set_title(f"{name}\n변위 히트맵")
    plt.colorbar(im, ax=axes[0])

    xs = [r["context_val"] for r in probe_result["probe"]]
    r2s = [r["r2"] if r["r2"] is not None else np.nan for r in probe_result["probe"]]
    colors = ["#2a78d6" if r["r2"] is not None else "#bbbbbb" for r in probe_result["probe"]]
    axes[1].bar(range(len(xs)), [max(v, 0) if not np.isnan(v) else 0 for v in r2s], color=colors)
    axes[1].set_xticks(range(len(xs)))
    axes[1].set_xticklabels([f"{x:.2g}" for x in xs], rotation=45, fontsize=7)
    axes[1].set_xlabel(f"{axis2} (context)")
    axes[1].set_ylabel(f"R²({axis1} | {axis2})")
    axes[1].set_title("context별 held-out R² (회색=검정력 부족)")
    fig.tight_layout()
    fig_path = FIG_DIR / f"11_phase3_2d_{name}.pdf"
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def main():
    null_axis_data = {a: dr.load_concat(a) for a in dr.NULL_AXES}
    null_disp_p95, _ = dr.build_null_floor(null_axis_data)
    print(f"널 바닥(displacement) p95 = {null_disp_p95:.3e}")

    all_flat, gate_report = gate_sanity_check(null_disp_p95)
    print(gate_report)
    if not all_flat:
        with open(RESULTS_DIR / "11_phase3_2d.md", "w", encoding="utf-8") as f:
            f.write("# 3-2 결과 — ★★★ 게이트 sanity check FAIL, 중단\n\n" + gate_report)
        print("\n★★★★★ 게이트 sanity check 실패 — 즉시 중단, 사람 보고 필요 ★★★★★")
        return

    lines = ["# 3-2. 게이트 히트맵 + context별 R²(A|B)\n", gate_report, "\n"]
    all_probe_results = {}
    for name in PAIRS_2D:
        D, g1, g2, axis1, axis2, r1, r2 = displacement_heatmap(name)
        probe_result = context_probe(name)
        fig_path = plot_2d(name, D, g1, g2, axis1, axis2, probe_result)
        all_probe_results[name] = probe_result

        r2_vals = [r["r2"] for r in probe_result["probe"] if r["r2"] is not None]
        lines.append(f"### {name} (focus={axis1}, context={axis2})\n")
        lines.append(f"변위 범위: [{D.min():.4f}, {D.max():.4f}]. R²({axis1}|{axis2}) 범위: "
                     f"[{min(r2_vals):.4f}, {max(r2_vals):.4f}] (유효 {len(r2_vals)}/{len(g2)}개)")
        lines.append(f"그림: `{fig_path.name}`\n")
        lines.append("| context 값 | N | R² |")
        lines.append("|---|---|---|")
        for r in probe_result["probe"]:
            r2_str = f"{r['r2']:.4f}" if r["r2"] is not None else f"— ({r['note']})"
            lines.append(f"| {r['context_val']:.3g} | {r['n']} | {r2_str} |")
        lines.append("")
        print(f"완료: {name}  R2 range=[{min(r2_vals):.4f},{max(r2_vals):.4f}]")

    out_path = RESULTS_DIR / "11_phase3_2d.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n저장: {out_path}")

    with open(RESULTS_DIR / "11_phase3_2d_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_probe_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
