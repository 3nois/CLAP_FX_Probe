"""CLAP FX Probe — 11_phase9_retrieval.py (Phase 9 §6-3: R0 / R3 / Ror)

캐시 전용, CLAP 재계산 없음. §3 규약: family 층화 60/20/20(seed=0),
train에서 v_bar/v_true 정의, val에서 alpha 선택(축·레벨·팔마다 독립),
test에서 1회 평가. recall은 group-aware(결함 22, out/results/11_phase9_dupgroups.json).

R1(B2 예측 방향)은 §6-4에서 별도 실행.
"""
import json

import numpy as np

AXES = ["distortion_drive_db", "reverb_room_size"]
QUERY_LEVELS = [12, 18, 24]
ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0]
SEED = 0
N_BOOT = 1000


def unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def load_axis(axis):
    m = np.load(f"out/caches/11_phase2_{axis}.npz", allow_pickle=True)
    x = np.load(f"out/caches/11_phase2ext_{axis}.npz", allow_pickle=True)
    emb = np.concatenate([m["embeddings"], x["embeddings"]], axis=0)
    src = np.concatenate([m["src_id"], x["src_id"]], axis=0)
    order = np.argsort(src)
    return emb[order], m["theta_raw"]


def load_bypass():
    b = np.load("out/caches/11_phase2_bypass.npz", allow_pickle=True)
    be = np.load("out/caches/11_phase2ext_bypass.npz", allow_pickle=True)
    emb = np.concatenate([b["embeddings"], be["embeddings"]], axis=0)
    src = np.concatenate([b["src_id"], be["src_id"]], axis=0)
    order = np.argsort(src)
    return emb[order]


def load_family_array():
    base = json.load(open("out/results/11_phase2_sources.json"))["sources"]
    ext = json.load(open("out/results/11_phase2_sources_ext.json"))["sources"]
    fam = {s["src_id"]: s["family"] for s in base + ext}
    return np.array([fam[i] for i in range(1200)])


def load_dup_groups():
    d = json.load(open("out/results/11_phase9_dupgroups.json"))
    group_of = {int(k): set(v) for k, v in d["group_of_src_id"].items()}
    for i in range(1200):
        group_of.setdefault(i, {i})
    return group_of


def stratified_split(family_arr, seed, train_frac=0.6, val_frac=0.2):
    rng = np.random.RandomState(seed)
    families = sorted(set(family_arr.tolist()))
    train_idx, val_idx, test_idx = [], [], []
    for fam in families:
        idx = np.where(family_arr == fam)[0]
        perm = rng.permutation(idx)
        n = len(perm)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train_idx.append(perm[:n_train])
        val_idx.append(perm[n_train:n_train + n_val])
        test_idx.append(perm[n_train + n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx)


def group_hits(query_idx, sims, group_of, k):
    """query_idx: 라이브러리 내 질의 소스의 실제 src_id 목록 (평가용 self 식별자).
    sims: (n_query, 1200). top-k 중 질의 소스의 중복 그룹 구성원이 있으면 적중."""
    order = np.argsort(-sims, axis=1)
    hits = np.zeros(len(query_idx), dtype=bool)
    for i, qi in enumerate(query_idx):
        topk = set(order[i, :k].tolist())
        hits[i] = bool(group_of[qi] & topk)
    return hits


def bootstrap_ci(hits, n_boot=N_BOOT, seed=0):
    rng = np.random.RandomState(seed)
    n = len(hits)
    means = np.array([hits[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_diff(hits_a, hits_b, n_boot=N_BOOT, seed=0):
    rng = np.random.RandomState(seed)
    n = len(hits_a)
    diffs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs[b] = hits_a[idx].mean() - hits_b[idx].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def recall_table(query_idx, q, bypass, group_of, ks=(1, 5, 10)):
    sims = unit(q) @ unit(bypass).T
    out = {}
    for k in ks:
        hits = group_hits(query_idx, sims, group_of, k)
        lo, hi = bootstrap_ci(hits)
        out[k] = {"mean": float(hits.mean()), "ci": [lo, hi], "hits": hits}
    return out


def main():
    bypass = load_bypass()
    family_arr = load_family_array()
    group_of = load_dup_groups()
    train_idx, val_idx, test_idx = stratified_split(family_arr, seed=SEED)
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    results = {}
    for axis in AXES:
        emb, theta = load_axis(axis)
        results[axis] = {}
        for lvl in QUERY_LEVELS:
            e_wet = emb[:, lvl, :]
            v_true = unit(bypass - e_wet)  # per-source true correction direction
            v_bar = unit(v_true[train_idx].mean(axis=0, keepdims=True))[0]

            print(f"\n=== {axis} lvl={lvl} theta={theta[lvl]:.3f} ===")
            res_lvl = {"theta": float(theta[lvl])}

            # R0: alpha 없음
            r0 = recall_table(test_idx, e_wet[test_idx], bypass, group_of)
            print(f"  R0        R@1={r0[1]['mean']:.4f} {r0[1]['ci']}  "
                  f"R@5={r0[5]['mean']:.4f}  R@10={r0[10]['mean']:.4f} {r0[10]['ci']}")
            res_lvl["R0"] = {k: {"mean": v["mean"], "ci": v["ci"]} for k, v in r0.items()}

            for arm_name, v_arm in [("R3", v_bar), ("Ror", None)]:
                # val에서 alpha 선택 (R@1 기준)
                best_alpha, best_val_r1 = None, -1.0
                for alpha in ALPHA_GRID:
                    if arm_name == "R3":
                        q_val = unit(e_wet[val_idx] + alpha * v_arm)
                    else:
                        q_val = unit(e_wet[val_idx] + alpha * v_true[val_idx])
                    sims = unit(q_val) @ unit(bypass).T
                    hits = group_hits(val_idx, sims, group_of, 1)
                    r1 = hits.mean()
                    if r1 > best_val_r1:
                        best_val_r1, best_alpha = r1, alpha

                if arm_name == "R3":
                    q_test = unit(e_wet[test_idx] + best_alpha * v_arm)
                else:
                    q_test = unit(e_wet[test_idx] + best_alpha * v_true[test_idx])
                rec = recall_table(test_idx, q_test, bypass, group_of)
                print(f"  {arm_name:<9} alpha={best_alpha:<5} R@1={rec[1]['mean']:.4f} {rec[1]['ci']}  "
                      f"R@5={rec[5]['mean']:.4f}  R@10={rec[10]['mean']:.4f} {rec[10]['ci']}")
                diff_lo, diff_hi = paired_bootstrap_diff(rec[10]["hits"], r0[10]["hits"])
                print(f"      {arm_name}-R0 R@10 대응부트스트랩 CI = [{diff_lo:.4f}, {diff_hi:.4f}]")
                res_lvl[arm_name] = {
                    "alpha": best_alpha,
                    **{k: {"mean": v["mean"], "ci": v["ci"]} for k, v in rec.items()},
                    "vs_R0_r10_diff_ci": [diff_lo, diff_hi],
                }
            results[axis][lvl] = res_lvl

    with open("out/results/11_phase9_r0r3ror.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n저장: out/results/11_phase9_r0r3ror.json")


if __name__ == "__main__":
    main()
