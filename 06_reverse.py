"""CLAP FX Probe — 06_reverse.py (4차 개정: 단사성 정의 명확화)

역방향 사상: e_wet → (e_dry, 이펙트 종류, θ). 분류 헤드 + 회귀 헤드 + 재구성 헤드를
정방향과 동일 용량·동일 src_id 분리로 학습한다.

  cycle consistency: e_dry →[정방향 대리모델]→ e_wet' →[역방향]→ e_dry''
  기준선은 cos(e_dry, e_wet) — "아무 처리도 안 했을 때"의 값. 이를 상회하지 못하면
  역방향이 무의미하다. highshelf는 이 기준선이 이미 천장(≈0.997~0.999)이라 개선
  여지가 거의 없다는 점을 플롯/주석에 명시한다.

  단사성 진단: wet 임베딩 공간에서 서로 다른 소스가 최근접 이웃으로 충돌하는 비율.
  충돌이 잦으면 역방향은 원리적으로 불가능하며 이 또한 유효한 결과다.

4차: 3차는 collision_rate=0.018(threshold=0.99)과 nn_cosine_median=0.9953이 함께
보고되어 모순처럼 보였다 — 중앙값이 threshold를 넘는데 충돌률이 1.8%일 리 없다는
지적. 실제로는 모순이 아니라 정의가 불충분히 설명된 것이었다: collision은 "최근접
이웃이 '다른 소스'이면서 유사도가 threshold를 넘는 경우"만 센다. 같은 소스의 다른
θ끼리 유사도가 높은 것(오히려 정상 — 같은 악기의 다른 이펙트 강도는 임베딩이 가까워야
자연스럽다)은 collision이 아니다. 즉 nn_cosine_median이 높은 이유는 대부분의 점에서
최근접 이웃이 "같은 소스의 다른 θ"이기 때문일 수 있다. 이를 same-source/
different-source로 분리해 명시적으로 보고하고, threshold도 0.95/0.99/0.999 세 값
모두에서 계산해 단일 threshold 선택에 좌우되지 않게 한다.

결과 해석은 이 스크립트가 단정하지 않는다. README의 판정 기준표를 따를 것.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

EFFECTS = ["reverb", "distortion", "highshelf"]
COLORS = {"reverb": "#2a78d6", "distortion": "#eb6834", "highshelf": "#1baf7a", "baseline": "#898781"}
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


class SurrogateMLP(nn.Module):
    """02_surrogate.py와 동일 구조 — 정방향 모델 로딩용."""

    def __init__(self, in_dim, hidden=1024, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReverseModel(nn.Module):
    """e_wet → (이펙트 종류, θ, e_dry). 정방향과 동일 용량(hidden=1024, 2-layer)."""

    def __init__(self, in_dim=512, hidden=1024, n_effects=3, theta_dim=9):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.effect_head = nn.Linear(hidden, n_effects)
        self.theta_head = nn.Linear(hidden, theta_dim)
        self.dry_head = nn.Linear(hidden, in_dim)  # e_dry_pred = e_wet + dry_head(...) (residual)

    def forward(self, e_wet):
        h = self.shared(e_wet)
        return self.effect_head(h), self.theta_head(h), e_wet + self.dry_head(h)


def load_embeddings(path: Path):
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_config(out_dir: Path):
    with open(out_dir / "embed_config.json") as f:
        return json.load(f)


def cosine_rows(a, b):
    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return (a_n * b_n).sum(axis=1)


def split_sources(unique_src_ids, seed, test_size=0.3):
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_src_ids)
    n_test = max(1, int(round(len(shuffled) * test_size)))
    return set(shuffled[n_test:].tolist()), set(shuffled[:n_test].tolist())


def injectivity_diagnostic(embeddings, src_ids, thresholds=(0.95, 0.99, 0.999), n_sample=3000, seed=0):
    """wet 임베딩 공간에서의 단사성(injectivity) 진단.

    ★ collision의 정의: 한 점의 "최근접 이웃"이 (a) 자신과 "다른 소스"이고, (b) 코사인
    유사도가 threshold를 넘는 경우만을 collision으로 센다. 같은 소스(같은 악기)의 다른
    θ끼리 최근접 이웃으로 잡히는 것은 collision이 아니다 — 오히려 정상적으로 기대되는
    구조다 (같은 악기의 다른 이펙트 강도는 임베딩이 서로 가까워야 자연스럽다).
    nn_cosine_median이 threshold보다 높다고 해서 collision_rate가 높아야 하는 것은
    아니다: 최근접 이웃 "대부분"이 같은 소스의 다른 θ일 수 있기 때문이다. 이를
    same-source/different-source로 분리해 별도로 보고한다.
    """
    rng = np.random.RandomState(seed)
    n = min(n_sample, len(embeddings))
    idx = rng.choice(len(embeddings), size=n, replace=False)
    X = embeddings[idx].astype(np.float64)
    srcs = src_ids[idx]

    X_norm = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    sim = X_norm @ X_norm.T
    np.fill_diagonal(sim, -1.0)
    nn_idx = sim.argmax(axis=1)
    nn_sim = sim[np.arange(n), nn_idx]
    different_source = srcs != srcs[nn_idx]

    same_source_sim = nn_sim[~different_source]
    diff_source_sim = nn_sim[different_source]

    collision_by_threshold = {}
    for t in thresholds:
        collision_by_threshold[str(t)] = {
            "collision_rate": float(((nn_sim > t) & different_source).mean()),
            "same_source_neighbor_rate_above_threshold": float(((nn_sim > t) & ~different_source).mean()),
        }

    return {
        "definition_note": "collision = 최근접 이웃이 '다른 소스'이면서 유사도>threshold인 경우만. "
        "같은 소스의 다른 θ가 최근접 이웃인 것은 collision이 아니다.",
        "thresholds": [float(t) for t in thresholds],
        "collision_by_threshold": collision_by_threshold,
        "nn_cosine_mean_overall": float(nn_sim.mean()),
        "nn_cosine_median_overall": float(np.median(nn_sim)),
        "nn_cosine_median_same_source": float(np.median(same_source_sim)) if len(same_source_sim) else None,
        "nn_cosine_median_diff_source": float(np.median(diff_source_sim)) if len(diff_source_sim) else None,
        "frac_nearest_neighbor_is_same_source": float((~different_source).mean()),
        "n_sampled": int(n),
    }


def plot_injectivity(injectivity, out_path):
    thresholds = injectivity["thresholds"]
    collision_rates = [injectivity["collision_by_threshold"][str(t)]["collision_rate"] for t in thresholds]
    same_src_rates = [injectivity["collision_by_threshold"][str(t)]["same_source_neighbor_rate_above_threshold"] for t in thresholds]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)

    ax = axes[0]
    x = np.arange(len(thresholds))
    width = 0.35
    ax.bar(x - width / 2, collision_rates, width, label="충돌률 (다른 소스, 유사도>threshold)", color="#eb6834", zorder=3)
    ax.bar(x + width / 2, same_src_rates, width, label="같은 소스인데 유사도>threshold", color=COLORS["baseline"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlabel("threshold")
    ax.set_ylabel("비율")
    ax.set_title("Threshold별 충돌률")
    ax.legend(frameon=False, fontsize=7)
    style_axis(ax)

    ax = axes[1]
    labels = ["전체", "같은 소스", "다른 소스"]
    vals = [
        injectivity["nn_cosine_median_overall"],
        injectivity["nn_cosine_median_same_source"],
        injectivity["nn_cosine_median_diff_source"],
    ]
    ax.bar(np.arange(3), vals, color=[COLORS["baseline"], "#2a78d6", "#eb6834"], zorder=3)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("최근접 이웃 코사인 중앙값")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"최근접 이웃 성격 분리 (같은 소스 비율={injectivity['frac_nearest_neighbor_is_same_source']:.2f})")
    style_axis(ax)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cycle_consistency(cycle_by_effect, baseline_by_effect, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    x = np.arange(len(EFFECTS))
    width = 0.35
    cycle_vals = [cycle_by_effect[e] for e in EFFECTS]
    baseline_vals = [baseline_by_effect[e] for e in EFFECTS]

    ax.bar(x - width / 2, cycle_vals, width, label="cycle: cos(e_dry, e_dry'')", color=[COLORS[e] for e in EFFECTS], zorder=3)
    ax.bar(x + width / 2, baseline_vals, width, label="기준선: cos(e_dry, e_wet)", color=COLORS["baseline"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(EFFECTS)
    ax.set_ylabel("코사인 유사도")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cycle Consistency — 기준선(아무 처리 안 함)을 넘어서야 의미 있음")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="역방향 사상 모델 + cycle consistency + 단사성 진단 (3차)")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device == "mps":
        import os

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    out_dir = Path(args.out)
    emb_path = Path(args.embeddings)
    if not (out_dir / "surrogate_model.pt").exists():
        raise FileNotFoundError(f"{out_dir / 'surrogate_model.pt'}가 없습니다. 먼저 02_surrogate.py를 실행하세요.")

    device = torch.device(args.device)
    ckpt = torch.load(out_dir / "surrogate_model.pt", map_location=device, weights_only=False)
    fwd_model = SurrogateMLP(in_dim=ckpt["in_dim"]).to(device)
    fwd_model.load_state_dict(ckpt["state_dict"])
    fwd_model.eval()

    d = load_embeddings(emb_path)
    config = load_config(emb_path.parent)
    theta_slots = {e: tuple(v) for e, v in config["theta_slots"].items()}
    theta_width = config["theta_width"]

    dry_mask = d["effect"] == "dry"
    dry_by_src = dict(zip(d["src_id"][dry_mask].tolist(), d["embeddings"][dry_mask]))

    unique_srcs = np.unique(d["src_id"])
    train_srcs, test_srcs = split_sources(unique_srcs, args.seed, args.test_size)
    print(f"소스 분할: train {len(train_srcs)}개 / test {len(test_srcs)}개 (src_id 기준, 정방향과 동일 규칙)")

    non_dry = d["effect"] != "dry"
    src_ids = d["src_id"][non_dry]
    dry_all = np.stack([dry_by_src[s] for s in src_ids]).astype(np.float32)
    wet_all = d["embeddings"][non_dry].astype(np.float32)
    theta_all = d["theta_norm"][non_dry].astype(np.float32)
    effect_names_all = d["effect"][non_dry]
    effect_idx_all = np.array([EFFECTS.index(e) for e in effect_names_all], dtype=np.int64)

    train_mask = np.array([s in train_srcs for s in src_ids])
    test_mask = np.array([s in test_srcs for s in src_ids])

    print("역방향 모델 학습 중...")
    torch.manual_seed(args.seed)
    model = ReverseModel(theta_dim=theta_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    wet_tr = torch.tensor(wet_all[train_mask], device=device)
    dry_tr = torch.tensor(dry_all[train_mask], device=device)
    theta_tr = torch.tensor(theta_all[train_mask], device=device)
    effect_tr = torch.tensor(effect_idx_all[train_mask], device=device)

    wet_te = torch.tensor(wet_all[test_mask], device=device)
    dry_te = torch.tensor(dry_all[test_mask], device=device)
    theta_te = torch.tensor(theta_all[test_mask], device=device)
    effect_te = torch.tensor(effect_idx_all[test_mask], device=device)

    n_train = wet_tr.shape[0]
    batch_size = min(64, n_train)
    model.train()
    for _epoch in range(args.epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            effect_logits, theta_pred, dry_pred = model(wet_tr[idx])
            loss_effect = F.cross_entropy(effect_logits, effect_tr[idx])
            loss_theta = F.mse_loss(theta_pred, theta_tr[idx])
            loss_dry = F.mse_loss(dry_pred, dry_tr[idx]) + (1 - F.cosine_similarity(dry_pred, dry_tr[idx], dim=1)).mean()
            (loss_effect + loss_theta + loss_dry).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        effect_logits_te, theta_pred_te, dry_pred_te = model(wet_te)
        effect_acc = (effect_logits_te.argmax(dim=1) == effect_te).float().mean().item()

        # 파라미터별 R² (활성 슬롯만) — held-out
        theta_pred_np = theta_pred_te.cpu().numpy()
        theta_true_np = theta_te.cpu().numpy()
        param_r2 = {}
        for e in EFFECTS:
            start_i, end_i = theta_slots[e]
            e_mask = (effect_te.cpu().numpy() == EFFECTS.index(e))
            if e_mask.sum() < 3:
                continue
            for i, pname in enumerate(config["param_order"][e]):
                col = start_i + i
                yt = theta_true_np[e_mask, col]
                yp = theta_pred_np[e_mask, col]
                ss_res = float(np.sum((yt - yp) ** 2))
                ss_tot = float(np.sum((yt - yt.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else None
                param_r2[f"{e}.{pname}"] = r2

        recon_cos = F.cosine_similarity(dry_pred_te, dry_te, dim=1).cpu().numpy()

    print("Cycle consistency 계산 중...")
    onehots = {e: torch.zeros(len(EFFECTS), device=device) for e in EFFECTS}
    for e in EFFECTS:
        onehots[e][EFFECTS.index(e)] = 1.0

    cycle_by_effect, baseline_by_effect = {}, {}
    recon_cos_by_effect = {}
    with torch.no_grad():
        for e in EFFECTS:
            e_mask = test_mask & (d["effect"][non_dry] == e)
            if e_mask.sum() == 0:
                continue
            dry_e = torch.tensor(dry_all[e_mask], device=device)
            theta_e = torch.tensor(theta_all[e_mask], device=device)
            wet_e_real = torch.tensor(wet_all[e_mask], device=device)
            onehot_e = onehots[e].unsqueeze(0).expand(dry_e.shape[0], -1)

            x_full = torch.cat([dry_e, onehot_e, theta_e], dim=1)
            wet_prime = dry_e + fwd_model(x_full)  # 정방향(대리모델)

            _eff_logits, _theta_pred, dry_double_prime = model(wet_prime)  # 역방향

            cycle_cos = F.cosine_similarity(dry_e, dry_double_prime, dim=1)
            baseline_cos = F.cosine_similarity(dry_e, wet_e_real, dim=1)
            cycle_by_effect[e] = float(cycle_cos.mean().item())
            baseline_by_effect[e] = float(baseline_cos.mean().item())

            e_mask_local = effect_te.cpu().numpy() == EFFECTS.index(e)
            recon_cos_by_effect[e] = float(recon_cos[e_mask_local].mean()) if e_mask_local.sum() > 0 else None

    print("단사성(injectivity) 진단 중...")
    injectivity = injectivity_diagnostic(wet_all, src_ids, seed=args.seed)

    print("그림 저장 중...")
    plot_cycle_consistency(cycle_by_effect, baseline_by_effect, out_dir / "cycle_consistency.png")
    plot_injectivity(injectivity, out_dir / "injectivity.png")

    results_path = out_dir / "results.json"
    results_json = {}
    if results_path.exists():
        with open(results_path) as f:
            results_json = json.load(f)

    results_json["reverse_model"] = {
        "effect_type_accuracy": effect_acc,
        "param_r2": param_r2,
        "dry_reconstruction_cosine_by_effect": recon_cos_by_effect,
        "cycle_consistency": cycle_by_effect,
        "cycle_baseline": baseline_by_effect,
        "cycle_note": "cycle_consistency가 cycle_baseline(=cos(e_dry,e_wet), 아무 처리도 안 했을 때의 값)을 "
        "넘지 못하면 역방향 모델이 무의미하다. highshelf는 baseline 자체가 이미 천장이라 개선 여지가 거의 없다.",
        "injectivity": injectivity,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "test_size": args.test_size,
    }
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"완료: {results_path}, {out_dir}/cycle_consistency.png, {out_dir}/injectivity.png")
    coll_99 = injectivity["collision_by_threshold"]["0.99"]["collision_rate"]
    print(f"이펙트 종류 분류 정확도: {effect_acc:.3f}, 단사성 충돌률(threshold=0.99): {coll_99:.4f}")


if __name__ == "__main__":
    main()
