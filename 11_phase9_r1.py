"""CLAP FX Probe — 11_phase9_r1.py (Phase 9 §6-4: B2 재학습 + R1)

§0.2 정정: oat_emb 대신 Phase2-native 데이터(25레벨, 자체 family)로
MLPDualHead(21_handle_predict_phase1.py 재사용)를 재학습한다. 학습 입력은
평가 레벨과 동일한 {12,18,24} 세 레벨을 풀링(21_handle_predict_phase1.py의
B2가 레벨1·2를 풀링하던 방식과 동일한 논리) — X=e_wet, Y=e_bypass-e_wet(raw).

§4 누수 점검: Y는 train_idx 소스만으로 구성, alpha는 val_idx에서만 선택,
test_idx는 마지막 평가 1회에만 사용.
"""
import json
from importlib import import_module

import numpy as np
import torch

b2mod = import_module("21_handle_predict_phase1")
r0mod = import_module("11_phase9_retrieval")

AXES, ALPHA_GRID, QUERY_LEVELS, SEED = r0mod.AXES, r0mod.ALPHA_GRID, r0mod.QUERY_LEVELS, r0mod.SEED
unit = r0mod.unit
load_axis, load_bypass = r0mod.load_axis, r0mod.load_bypass
load_family_array, load_dup_groups = r0mod.load_family_array, r0mod.load_dup_groups
stratified_split, group_hits = r0mod.stratified_split, r0mod.group_hits
paired_bootstrap_diff, recall_table = r0mod.paired_bootstrap_diff, r0mod.recall_table

MAX_EPOCHS = 300
PATIENCE = 30


def build_b2_training_set(emb, bypass, train_idx, levels=QUERY_LEVELS):
    """평가 레벨 {12,18,24}을 그대로 풀링해 학습 표적을 만든다(§0.2, 누수 없음:
    train_idx 소스만 사용)."""
    Xs, Ys = [], []
    for lvl in levels:
        e_wet = emb[:, lvl, :][train_idx]
        Xs.append(e_wet)
        Ys.append(bypass[train_idx] - e_wet)
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def train_b2(axis, emb, bypass, train_idx, val_idx, seed=SEED):
    X_train, Y_train = build_b2_training_set(emb, bypass, train_idx)
    # val: 레벨별로 각각 검증 손실을 볼 필요는 없음(모델은 레벨 정보를 안 받음) — 동일하게 풀링
    X_val, Y_val = build_b2_training_set(emb, bypass, val_idx)

    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    Ytr_t = torch.tensor(Y_train, dtype=torch.float32)
    Xva_t = torch.tensor(X_val, dtype=torch.float32)
    Yva_t = torch.tensor(Y_val, dtype=torch.float32)

    model = b2mod.MLPDualHead(512, 1024, 0.1)
    model, best_val_cos, epochs_used = b2mod.train_dual_head(
        model, Xtr_t, Ytr_t, Xva_t, Yva_t, seed, MAX_EPOCHS, PATIENCE
    )
    print(f"  [B2 학습] {axis}: n_train={len(X_train)} n_val={len(X_val)} "
          f"val_cos={best_val_cos:.4f} epochs={epochs_used}")
    return model


def predict_direction(model, e_wet):
    model.eval()
    with torch.no_grad():
        dp, _ = model(torch.tensor(e_wet, dtype=torch.float32))
    return dp.numpy()


def main():
    bypass = load_bypass()
    family_arr = load_family_array()
    group_of = load_dup_groups()
    train_idx, val_idx, test_idx = stratified_split(family_arr, seed=SEED)

    prior = json.load(open("out/results/11_phase9_r0r3ror.json"))
    results = {}

    for axis in AXES:
        emb, theta = load_axis(axis)
        model = train_b2(axis, emb, bypass, train_idx, val_idx)
        results[axis] = {}

        for lvl in QUERY_LEVELS:
            e_wet = emb[:, lvl, :]
            v_hat = unit(predict_direction(model, e_wet))
            # 진단용: test에서의 held-out 방향 코사인 (B2 자체 품질, 결과문서용)
            v_true = unit(bypass - e_wet)
            cos_te = np.sum(v_hat[test_idx] * v_true[test_idx], axis=-1)

            print(f"\n=== {axis} lvl={lvl} theta={theta[lvl]:.3f} | B2 held-out cos "
                  f"mean={cos_te.mean():.4f} median={np.median(cos_te):.4f} ===")

            best_alpha, best_val_r1 = None, -1.0
            for alpha in ALPHA_GRID:
                q_val = unit(e_wet[val_idx] + alpha * v_hat[val_idx])
                sims = unit(q_val) @ unit(bypass).T
                hits = group_hits(val_idx, sims, group_of, 1)
                r1 = hits.mean()
                if r1 > best_val_r1:
                    best_val_r1, best_alpha = r1, alpha

            q_test = unit(e_wet[test_idx] + best_alpha * v_hat[test_idx])
            rec = recall_table(test_idx, q_test, bypass, group_of)
            r0_rec = recall_table(test_idx, e_wet[test_idx], bypass, group_of)  # 대응부트스트랩용 재계산

            print(f"  R1        alpha={best_alpha:<5} R@1={rec[1]['mean']:.4f} {rec[1]['ci']}  "
                  f"R@5={rec[5]['mean']:.4f}  R@10={rec[10]['mean']:.4f} {rec[10]['ci']}")
            diff_lo, diff_hi = paired_bootstrap_diff(rec[10]["hits"], r0_rec[10]["hits"])
            print(f"      R1-R0 R@10 대응부트스트랩 CI = [{diff_lo:.4f}, {diff_hi:.4f}]  "
                  f"(참고 R0 R@1={r0_rec[1]['mean']:.4f})")

            r3 = prior[axis][str(lvl)]["R3"]
            ror = prior[axis][str(lvl)]["Ror"]
            print(f"      참고: R3 R@1={r3['1']['mean']:.4f} (alpha={r3['alpha']})  "
                  f"Ror R@1={ror['1']['mean']:.4f} (alpha={ror['alpha']})")

            results[axis][lvl] = {
                "theta": float(theta[lvl]),
                "b2_heldout_cos_mean": float(cos_te.mean()),
                "b2_heldout_cos_median": float(np.median(cos_te)),
                "R1": {
                    "alpha": best_alpha,
                    **{k: {"mean": v["mean"], "ci": v["ci"]} for k, v in rec.items()},
                    "vs_R0_r10_diff_ci": [diff_lo, diff_hi],
                },
            }

    with open("out/results/11_phase9_r1.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n저장: out/results/11_phase9_r1.json")


if __name__ == "__main__":
    main()
