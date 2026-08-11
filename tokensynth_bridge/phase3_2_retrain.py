"""9차 Phase 3-2 — TokenSynth 임베딩 공간에서 8차 모델 재학습.

8차(21_handle_predict_phase1.py)와 완전히 동일한 구조·분할·시드를 쓰되, 입력
임베딩만 Phase 3-1에서 새로 뽑은 out/caches/oat_emb_ts.npz(TokenSynth 공간)로
바꾼다. 정방향(g: e_dry -> v)과 B2(h: 레벨1|2 임베딩 -> v_to_dry, ★실제로 β 스윕에
쓸 모델)만 재학습한다 — B1/B3/C/D는 이번 재학습 대상이 아니다.

8차 값(정방향 0.776/0.860/0.823, B2 0.714/0.823/0.813)과 나란히 비교해 크게
다른지 확인한다. out/results/results_8.json은 건드리지 않고 새 파일에 저장한다.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EFFECT_NAMES = ["reverb", "distortion", "highshelf"]

# 8차 값 (비교용 참고 상수 — out/results/results_8.json에서 재확인 가능)
PHASE8_FORWARD_MLP_COS = {"reverb": 0.7759, "distortion": 0.8599, "highshelf": 0.8232}
PHASE8_B2_MLP_COS = {"reverb": 0.7144, "distortion": 0.8230, "highshelf": 0.8134}


def stratified_split(family_arr, seed, train_frac=0.8, val_frac=0.1):
    rng = np.random.RandomState(seed)
    families = sorted(set(family_arr.tolist()))
    train_idx, val_idx, test_idx = [], [], []
    per_family = {}
    for fam in families:
        idx = np.where(family_arr == fam)[0]
        perm = rng.permutation(idx)
        n = len(perm)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        tr, va, te = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
        train_idx.append(tr); val_idx.append(va); test_idx.append(te)
        per_family[fam] = {"n_total": int(n), "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))}
    return (np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx), per_family)


class MLPDualHead(nn.Module):
    def __init__(self, dim=512, hidden=1024, dropout=0.1):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.dir_out = nn.Linear(hidden, dim)
        self.mag_out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        raw = self.dir_out(h)
        direction = raw / (raw.norm(dim=-1, keepdim=True) + 1e-8)
        magnitude = F.softplus(self.mag_out(h)).squeeze(-1)
        return direction, magnitude


def unit_np(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def dual_head_loss(dir_pred, mag_pred, dir_true, mag_true, lambda_mag=0.3):
    cos = (dir_pred * dir_true).sum(-1)
    dir_loss = (1 - cos).mean()
    mag_loss = (torch.abs(mag_pred - mag_true) / (mag_true + 1e-8)).mean()
    return dir_loss + lambda_mag * mag_loss


def train_dual_head(model, X_train, Y_train, X_val, Y_val, seed, max_epochs=300, patience=30, lr=1e-3,
                     weight_decay=1e-4, lambda_mag=0.3):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Yt_dir = F.normalize(Y_train, dim=-1); Yt_mag = Y_train.norm(dim=-1)
    Yv_dir = F.normalize(Y_val, dim=-1); Yv_mag = Y_val.norm(dim=-1)
    best_val_cos = -2.0; best_state = None; patience_ctr = 0; epoch = 0
    for epoch in range(max_epochs):
        model.train()
        dp, mp = model(X_train)
        loss = dual_head_loss(dp, mp, Yt_dir, Yt_mag, lambda_mag)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            dpv, mpv = model(X_val)
            val_cos = (dpv * Yv_dir).sum(-1).mean().item()
        if val_cos > best_val_cos:
            best_val_cos = val_cos; best_state = copy.deepcopy(model.state_dict()); patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_val_cos, epoch + 1


def bootstrap_cos_ci(cos_per_row, row_to_source, seed, n_boot=1000):
    sources = np.unique(row_to_source)
    rng = np.random.RandomState(seed)
    src_to_rows = {s: np.where(row_to_source == s)[0] for s in sources}
    means = []
    for _ in range(n_boot):
        boot_srcs = rng.choice(sources, size=len(sources), replace=True)
        rows = np.concatenate([src_to_rows[s] for s in boot_srcs])
        means.append(float(cos_per_row[rows].mean()))
    means = np.array(means)
    return float(means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fit_and_eval_mlp(Xtr, Ytr, Xva, Yva, Xte, Yte, src_te, seed, max_epochs, patience):
    model = MLPDualHead(512, 1024, 0.1)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32); Yva_t = torch.tensor(Yva, dtype=torch.float32)
    model, val_cos, epochs_used = train_dual_head(model, Xtr_t, Ytr_t, Xva_t, Yva_t, seed, max_epochs, patience)
    model.eval()
    with torch.no_grad():
        dp, mp = model(torch.tensor(Xte, dtype=torch.float32))
    dp = dp.numpy(); mp = mp.numpy()
    Yte_dir = unit_np(Yte)
    cos = np.sum(dp * Yte_dir, axis=-1)
    mean_cos, lo, hi = bootstrap_cos_ci(cos, src_te, seed)
    return {"cos_mean": mean_cos, "cos_ci": [lo, hi], "cos_median": float(np.median(cos)),
            "val_cos": val_cos, "epochs_used": epochs_used, "n_test": int(len(cos))}, model


def main():
    parser = argparse.ArgumentParser(description="9차 Phase 3-2 — TokenSynth 공간에서 정방향/B2 재학습")
    parser.add_argument("--oat-emb-ts", type=str, default="out/caches/oat_emb_ts.npz")
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out)
    results_dir = out_dir / "results"
    figures_dir = out_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"로딩 중: {args.oat_emb_ts}")
    d = np.load(args.oat_emb_ts, allow_pickle=False)
    emb = d["emb"]
    src_id = d["src_id"]
    family = d["instrument_family"]
    n_sources = emb.shape[0]

    train_idx, val_idx, test_idx, per_family = stratified_split(family, args.seed)
    print(f"분할(8차와 동일 시드): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    forward_results = {}
    b2_results = {}
    for ei, effect in enumerate(EFFECT_NAMES):
        # ---- 정방향 ----
        X = emb[:, ei, 0, :]
        Y = emb[:, ei, 2, :] - emb[:, ei, 0, :]
        res, _ = fit_and_eval_mlp(X[train_idx], Y[train_idx], X[val_idx], Y[val_idx],
                                   X[test_idx], Y[test_idx], src_id[test_idx], args.seed, args.max_epochs, args.patience)
        forward_results[effect] = res
        print(f"  [정방향] {effect}: cos={res['cos_mean']:.4f} {res['cos_ci']} (8차={PHASE8_FORWARD_MLP_COS[effect]:.4f})")

        # ---- B2 (레벨1+2 혼합) ----
        X1 = emb[:, ei, 1, :]; Y1 = emb[:, ei, 0, :] - emb[:, ei, 1, :]
        X2 = emb[:, ei, 2, :]; Y2 = emb[:, ei, 0, :] - emb[:, ei, 2, :]
        X_all = np.concatenate([X1, X2], axis=0)
        Y_all = np.concatenate([Y1, Y2], axis=0)
        pos_all = np.concatenate([np.arange(n_sources), np.arange(n_sources)])
        src_all_dup = np.concatenate([src_id, src_id])

        def sub(idx_src):
            mask = np.isin(pos_all, idx_src)
            return X_all[mask], Y_all[mask], src_all_dup[mask]

        Xtr2, Ytr2, _ = sub(train_idx)
        Xva2, Yva2, _ = sub(val_idx)
        Xte2, Yte2, Ste2 = sub(test_idx)

        res2, model_b2 = fit_and_eval_mlp(Xtr2, Ytr2, Xva2, Yva2, Xte2, Yte2, Ste2, args.seed, args.max_epochs, args.patience)
        b2_results[effect] = res2
        print(f"  [B2]     {effect}: cos={res2['cos_mean']:.4f} {res2['cos_ci']} (8차={PHASE8_B2_MLP_COS[effect]:.4f})")

        torch.save(model_b2.state_dict(), figures_dir.parent / "caches" / f"b2_model_ts_{effect}.pt")

    # ---- 8차 대비 비교 ----
    comparison = {}
    for effect in EFFECT_NAMES:
        fwd_diff = forward_results[effect]["cos_mean"] - PHASE8_FORWARD_MLP_COS[effect]
        b2_diff = b2_results[effect]["cos_mean"] - PHASE8_B2_MLP_COS[effect]
        comparison[effect] = {
            "forward_ts": forward_results[effect]["cos_mean"], "forward_phase8": PHASE8_FORWARD_MLP_COS[effect], "forward_diff": fwd_diff,
            "b2_ts": b2_results[effect]["cos_mean"], "b2_phase8": PHASE8_B2_MLP_COS[effect], "b2_diff": b2_diff,
            "large_difference_flag": abs(fwd_diff) > 0.1 or abs(b2_diff) > 0.1,
        }

    elapsed_note = "정방향+B2, MLP만(8차와 동일 구조·시드)"
    results = {
        "meta": {"split": per_family, "seed": args.seed, "max_epochs": args.max_epochs, "patience": args.patience,
                  "source_npz": args.oat_emb_ts, "note": elapsed_note},
        "depends_on_surrogate": "none",
        "forward_ts": forward_results,
        "b2_ts": b2_results,
        "comparison_with_phase8": comparison,
    }
    results_path = results_dir / "results_9_phase3_2.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 3-2 요약: TokenSynth 공간 vs 8차(우리 파이프라인 공간) ===")
    for effect in EFFECT_NAMES:
        c = comparison[effect]
        flag = " ⚠ 큰 차이" if c["large_difference_flag"] else ""
        print(f"  {effect:<12} 정방향: {c['forward_ts']:.4f} vs {c['forward_phase8']:.4f} (Δ{c['forward_diff']:+.4f})   "
              f"B2: {c['b2_ts']:.4f} vs {c['b2_phase8']:.4f} (Δ{c['b2_diff']:+.4f}){flag}")
    print(f"\n저장: {results_path}")
    print("B2 모델 가중치: out/caches/b2_model_ts_{reverb,distortion,highshelf}.pt (Phase 3-4에서 재사용)")
    print("★ 여기서 멈춥니다. 8차 값과 비교를 확인한 뒤 3-3 진행 여부를 결정하세요.")


if __name__ == "__main__":
    main()
