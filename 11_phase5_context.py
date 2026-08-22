# -*- coding: utf-8 -*-
"""Phase 5-D — Branch B: context를 입력에 추가 — out/prereg/11_phase5.md 확정 방법론.

Phase 3의 2-D 캐시(13레벨)를 재사용. 전범위 handle(focus 축 최대-최소)을
context(다른 축) 값마다 따로 예측한다. context 없는 기준(512차원 입력)과
context 포함(513차원 입력, 정규화된 context 값 이어붙임)을 비교한다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

base = import_module("11_phase2_render")
pred_mod = import_module("21_handle_predict_phase1")
q3q4 = import_module("11_phase5_q3q4")


class MLPDualHeadCtx(pred_mod.MLPDualHead):
    """입력 차원(임베딩+context)과 출력 차원(임베딩만, 512)이 다를 때 쓰는 최소 확장.
    forward()는 부모 클래스 그대로 — trunk/dir_out/mag_out 구성만 입력·출력 차원을 분리."""

    def __init__(self, input_dim, output_dim=512, hidden=1024, dropout=0.1):
        nn.Module.__init__(self)
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.dir_out = nn.Linear(hidden, output_dim)
        self.mag_out = nn.Linear(hidden, 1)

CACHE_DIR = base.CACHE_DIR
RESULTS_DIR = base.RESULTS_DIR
SEED = 0

PAIRS = [
    ("highshelf_gain_cutoff", "gain", "cutoff"),
    ("lowshelf_gain_cutoff", "gain", "cutoff"),
    ("peak_gain_cutoff", "gain", "cutoff"),
    ("reverb_wet_room", "room_size", "wet_level"),
]


def run_pair(cache_name, focus_name, context_name):
    d = np.load(CACHE_DIR / f"11_phase3_2d_{cache_name}.npz")
    emb = d["embeddings"]  # (1200,13,13,512)
    g1, g2 = d["grid1"], d["grid2"]
    axis1, axis2 = str(d["axis1"]), str(d["axis2"])
    src_id = d["src_id"]
    n_src = emb.shape[0]
    family = q3q4.get_family_array_1200()
    # family는 src_id 0..1199 순서로 만들어짐 — emb의 src_id 순서와 맞춰 정렬
    order = np.argsort(src_id)
    emb, src_id = emb[order], src_id[order]
    family_this = family[src_id]

    focus_is_axis1 = (axis1 == focus_name)
    focus_grid = g1 if focus_is_axis1 else g2
    context_grid = g2 if focus_is_axis1 else g1
    n_focus, n_context = len(focus_grid), len(context_grid)
    i_min, i_max = 0, n_focus - 1
    context_norm = (context_grid - context_grid.min()) / (context_grid.max() - context_grid.min() + 1e-12)

    def get_slice(focus_idx, context_idx):
        return emb[:, focus_idx, context_idx, :] if focus_is_axis1 else emb[:, context_idx, focus_idx, :]

    X_noctx_parts, X_ctx_parts, Y_parts, fam_parts, src_parts, pos_parts = [], [], [], [], [], []
    for ci in range(n_context):
        e_min = get_slice(i_min, ci)
        e_max = get_slice(i_max, ci)
        X_noctx_parts.append(e_min)
        ctx_col = np.full((n_src, 1), context_norm[ci], dtype=np.float32)
        X_ctx_parts.append(np.concatenate([e_min, ctx_col], axis=1))
        Y_parts.append(e_max - e_min)
        fam_parts.append(family_this)
        src_parts.append(src_id)
        pos_parts.append(np.arange(n_src))

    X_noctx = np.concatenate(X_noctx_parts, axis=0)
    X_ctx = np.concatenate(X_ctx_parts, axis=0)
    Y_all = np.concatenate(Y_parts, axis=0)
    fam_all = np.concatenate(fam_parts, axis=0)
    src_all = np.concatenate(src_parts, axis=0)
    pos_all = np.concatenate(pos_parts, axis=0)

    train_idx, val_idx, test_idx, _ = pred_mod.stratified_split(family_this, SEED)

    def sub(X, idx):
        mask = np.isin(pos_all, idx)
        return X[mask], Y_all[mask], fam_all[mask], src_all[mask]

    def train_eval(X_full, dim):
        Xtr, Ytr, Ftr, _ = sub(X_full, train_idx)
        Xva, Yva, _, _ = sub(X_full, val_idx)
        Xte, Yte, Fte, Ste = sub(X_full, test_idx)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32); Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
        Xva_t = torch.tensor(Xva, dtype=torch.float32); Yva_t = torch.tensor(Yva, dtype=torch.float32)
        model = MLPDualHeadCtx(input_dim=dim, output_dim=512, hidden=1024, dropout=0.1)
        model, best_val_cos, epochs = pred_mod.train_dual_head(
            model, Xtr_t, Ytr_t, Xva_t, Yva_t, SEED, max_epochs=300, patience=30, weight_decay=1e-4)
        model.eval()
        Xte_t = torch.tensor(Xte, dtype=torch.float32)
        with torch.no_grad():
            dp, mp = model(Xte_t)
        dp = dp.numpy()
        Yte_dir = pred_mod.unit_np(Yte)
        cos = np.sum(dp * Yte_dir, axis=-1)
        mean_cos, lo, hi = pred_mod.bootstrap_cos_ci(cos, Ste, SEED)
        return {"cos_mean": mean_cos, "cos_ci": [lo, hi], "n_test": len(Yte)}

    r_noctx = train_eval(X_noctx, dim=512)
    r_ctx = train_eval(X_ctx, dim=513)
    return {"pair": cache_name, "focus": focus_name, "context": context_name,
            "without_context": r_noctx, "with_context": r_ctx,
            "improvement": r_ctx["cos_mean"] - r_noctx["cos_mean"]}


def main():
    lines = ["# Phase 5-D — context를 입력에 추가 (Branch B)\n"]
    lines.append("| 축쌍 | focus | context | context 없음 cos | context 포함 cos | 개선폭 |")
    lines.append("|---|---|---|---|---|---|")
    all_results = []
    for cache_name, focus_name, context_name in PAIRS:
        r = run_pair(cache_name, focus_name, context_name)
        all_results.append(r)
        lines.append(f"| {cache_name} | {focus_name} | {context_name} | "
                     f"{r['without_context']['cos_mean']:.4f} [{r['without_context']['cos_ci'][0]:.3f},{r['without_context']['cos_ci'][1]:.3f}] | "
                     f"{r['with_context']['cos_mean']:.4f} [{r['with_context']['cos_ci'][0]:.3f},{r['with_context']['cos_ci'][1]:.3f}] | "
                     f"**{r['improvement']:+.4f}** |")
        print(f"완료: {cache_name}  no_ctx={r['without_context']['cos_mean']:.4f}  "
              f"ctx={r['with_context']['cos_mean']:.4f}  개선={r['improvement']:+.4f}")

    out_path = RESULTS_DIR / "11_phase5_q4_context.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n저장: {out_path}")
    with open(RESULTS_DIR / "11_phase5_q4_context_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
