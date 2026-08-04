"""CLAP FX Probe — 02_surrogate.py (3차)

01_embed.py가 만든 (e_dry, θ, e_wet) 데이터로 residual MLP 대리모델을 학습한다.

  e' = e_dry + MLP(e_dry, effect_onehot, θ)

이 대리모델은 미분 가능하므로 03_jacobian.py에서 J = ∂e'/∂θ를 autograd로 계산해
분석한다. 이 스크립트는 그 전 단계로 (1) 대리모델을 학습하고 (2) H1~H5 위계를
재구성해 "θ와 e_dry 각각에 얼마나 의존해야 wet을 잘 예측하는가"를 비교한다.

  H1  Δ = V·θ + b            (선형, θ만, 소스 무관)           — J = 상수
  H2  Δ = g(θ)·v              (방향 고정, 크기만 θ 의존)        — J 방향 고정
  H3  Δ = MLP(θ)               (비선형, θ만, 소스 무관)         — J = J(θ)
  H4  Δ = M(e_dry)·θ           (θ에 선형, 계수는 e_dry의 함수)   — J = J(e_dry)
  H5  Δ = MLP(e_dry, effect, θ) (완전 비선형)                   — J = J(e_dry, θ)

★ 대리모델은 실제 CLAP의 미분이 아니라 학습된 근사의 미분이다. held-out 코사인이
낮으면 야코비안 해석 전체가 무의미하다 — surrogate_quality.png로 반드시 확인할 것.

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
from sklearn.linear_model import Ridge

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
H_LADDER_COLORS = {"H1": "#86b6ef", "H2": "#5598e7", "H3": "#2a78d6", "H4": "#1c5cab", "H5": "#104281"}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load_embeddings(path: Path):
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_config(out_dir: Path):
    with open(out_dir / "embed_config.json") as f:
        return json.load(f)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return (a_n * b_n).sum(axis=1)


def split_sources(unique_src_ids: np.ndarray, seed: int, test_size: float = 0.3):
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_src_ids)
    n_test = max(1, int(round(len(shuffled) * test_size)))
    test_srcs = set(shuffled[:n_test].tolist())
    train_srcs = set(shuffled[n_test:].tolist())
    return train_srcs, test_srcs


# ---------------------------------------------------------------------------
# H5 — 완전 비선형 residual MLP (기존 2차 사상 모델과 동일 계열, 입력만 θ 벡터로 확장)
# ---------------------------------------------------------------------------


class SurrogateMLP(nn.Module):
    """524(e_dry 512 + 이펙트 원핫 3 + θ 9) → 512(Δ). GELU + LayerNorm."""

    def __init__(self, in_dim, hidden=1024, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class HyperLinearHead(nn.Module):
    """H4: Δ = M(e_dry)·θ, M은 e_dry로부터 예측되는 (512×θ_dim) 행렬. J=J(e_dry), θ에는 선형."""

    def __init__(self, in_dim=512, theta_dim=5, out_dim=512, hidden=256):
        super().__init__()
        self.theta_dim = theta_dim
        self.out_dim = out_dim
        self.hyper = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim * theta_dim))

    def forward(self, e_dry, theta):
        b = e_dry.shape[0]
        m = self.hyper(e_dry).view(b, self.out_dim, self.theta_dim)
        return torch.bmm(m, theta.unsqueeze(-1)).squeeze(-1)


class ThetaOnlyMLP(nn.Module):
    """H3: Δ = MLP(θ), e_dry 미사용. J=J(θ), 소스 무관."""

    def __init__(self, theta_dim, out_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(theta_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, theta):
        return self.net(theta)


def build_effect_dataset(d, dry_by_src, effect_name, theta_slots):
    mask = d["effect"] == effect_name
    src_ids = d["src_id"][mask]
    dry = np.stack([dry_by_src[s] for s in src_ids]).astype(np.float32)
    wet = d["embeddings"][mask].astype(np.float32)
    start, end = theta_slots[effect_name]
    theta = d["theta_norm"][mask][:, start:end].astype(np.float32)
    return {"dry": dry, "wet": wet, "theta": theta, "src_id": src_ids}


def train_torch_model(model, forward_fn, dry, theta, wet, train_mask, test_mask, device, epochs, lr, seed, shuffle_labels=False):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    wet_train = wet[train_mask].copy()
    if shuffle_labels:
        perm = rng.permutation(len(wet_train))
        wet_train = wet_train[perm]

    dry_tr = torch.tensor(dry[train_mask], device=device)
    theta_tr = torch.tensor(theta[train_mask], device=device)
    wet_tr = torch.tensor(wet_train, device=device)

    dry_te = torch.tensor(dry[test_mask], device=device)
    theta_te = torch.tensor(theta[test_mask], device=device)
    wet_te = torch.tensor(wet[test_mask], device=device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = dry_tr.shape[0]
    batch_size = min(64, max(1, n_train))

    model.train()
    for _epoch in range(epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            pred = forward_fn(model, dry_tr[idx], theta_tr[idx])
            mse = F.mse_loss(pred, wet_tr[idx])
            cos_loss = (1 - F.cosine_similarity(pred, wet_tr[idx], dim=1)).mean()
            loss = mse + cos_loss
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_te = forward_fn(model, dry_te, theta_te)
        test_cos = F.cosine_similarity(pred_te, wet_te, dim=1).mean().item()
    return float(test_cos), model


def fit_H1(dry, theta, wet, train_mask, test_mask):
    """H1: Δ = V·θ + b, e_dry 미사용, θ에 선형 (Ridge multi-output)."""
    delta_train = wet[train_mask] - dry[train_mask]
    model = Ridge(alpha=1.0)
    model.fit(theta[train_mask], delta_train)
    delta_pred = model.predict(theta[test_mask])
    pred = dry[test_mask] + delta_pred
    return float(cosine_rows(pred, wet[test_mask]).mean())


def fit_H2(dry, theta, wet, train_mask, test_mask):
    """H2: Δ = g(θ)·v, v는 학습 delta의 평균 방향(고정), g는 θ에 대한 선형 회귀."""
    delta_train = wet[train_mask] - dry[train_mask]
    v = delta_train.mean(axis=0)
    norm = np.linalg.norm(v)
    v_unit = v / norm if norm > 1e-12 else v
    proj_train = delta_train @ v_unit
    g_model = Ridge(alpha=1.0)
    g_model.fit(theta[train_mask], proj_train)
    g_test = g_model.predict(theta[test_mask])
    pred = dry[test_mask] + g_test[:, None] * v_unit
    return float(cosine_rows(pred, wet[test_mask]).mean())


def run_hierarchy_for_effect(effect_name, dataset, train_srcs, test_srcs, device, epochs, lr, seed):
    dry, theta, wet, src_id = dataset["dry"], dataset["theta"], dataset["wet"], dataset["src_id"]
    train_mask = np.array([s in train_srcs for s in src_id])
    test_mask = np.array([s in test_srcs for s in src_id])
    theta_dim = theta.shape[1]

    identity_cos = float(cosine_rows(dry[test_mask], wet[test_mask]).mean())
    h1 = fit_H1(dry, theta, wet, train_mask, test_mask)
    h2 = fit_H2(dry, theta, wet, train_mask, test_mask)

    h3_model = ThetaOnlyMLP(theta_dim)
    h3_cos, _ = train_torch_model(
        h3_model, lambda m, e, t: m(t), dry, theta, wet, train_mask, test_mask, device, epochs, lr, seed
    )

    h4_model = HyperLinearHead(theta_dim=theta_dim)
    h4_cos, _ = train_torch_model(
        h4_model, lambda m, e, t: m(e, t), dry, theta, wet, train_mask, test_mask, device, epochs, lr, seed
    )

    return {
        "identity": identity_cos,
        "H1": h1,
        "H2": h2,
        "H3": h3_cos,
        "H4": h4_cos,
        "n_test_rows": int(test_mask.sum()),
        "n_train_rows": int(train_mask.sum()),
    }


def plot_hierarchy(ladder_results, h5_by_effect, shuffle_control, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150, sharey=True)
    h_levels = ["H1", "H2", "H3", "H4", "H5"]
    for ax, effect_name in zip(axes, EFFECTS):
        ladder = ladder_results[effect_name]
        scores = [ladder["H1"], ladder["H2"], ladder["H3"], ladder["H4"], h5_by_effect[effect_name]]
        x = np.arange(len(h_levels))
        ax.bar(x, scores, color=[H_LADDER_COLORS[h] for h in h_levels], zorder=3)
        ax.axhline(ladder["identity"], color=INK_SECONDARY, linestyle="--", linewidth=1.2, zorder=2, label="identity")
        ax.axhline(shuffle_control, color=COLORS["baseline"], linestyle=":", linewidth=1.2, zorder=2, label="셔플")
        ax.set_xticks(x)
        ax.set_xticklabels(h_levels)
        ax.set_title(effect_name)
        style_axis(ax)
    axes[0].set_ylabel("Held-out cos(e', e_wet)")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("H1~H5 위계 사다리 (3차: 결합 θ 기반) — 어느 칸에서 구조가 잡히는가")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_surrogate_quality(real_cos, shuffled_cos, identity_cos, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    labels = ["identity\n(e'=e_dry)", "셔플\n(동일 용량)", "대리모델\n(실제 레이블)"]
    values = [identity_cos, shuffled_cos, real_cos]
    colors = [COLORS["baseline"], "#c3c2b7", "#2a78d6"]
    ax.bar(np.arange(3), values, color=colors, zorder=3)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Held-out cos(e', e_wet)")
    ax.set_ylim(0, 1.05)
    ax.set_title("대리모델 신뢰도 — 야코비안 해석의 전제 조건")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="residual MLP 대리모델 학습 + H1~H5 위계 재구성 (3차)")
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

    emb_path = Path(args.embeddings)
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path}가 없습니다. 먼저 01_embed.py를 실행하세요.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_embeddings(emb_path)
    config = load_config(Path(args.embeddings).parent)
    theta_slots = {e: tuple(v) for e, v in config["theta_slots"].items()}

    dry_mask = d["effect"] == "dry"
    dry_by_src = dict(zip(d["src_id"][dry_mask], d["embeddings"][dry_mask]))

    unique_srcs = np.unique(d["src_id"])
    train_srcs, test_srcs = split_sources(unique_srcs, args.seed, args.test_size)
    print(f"소스 분할: train {len(train_srcs)}개 / test {len(test_srcs)}개 (src_id 기준)")

    device = torch.device(args.device)

    # ---- H5(전체 대리모델) 풀링 학습 — 세 이펙트를 원핫으로 함께 넣는 단일 모델 ----
    non_dry = d["effect"] != "dry"
    src_ids = d["src_id"][non_dry]
    dry_all = np.stack([dry_by_src[s] for s in src_ids]).astype(np.float32)
    wet_all = d["embeddings"][non_dry].astype(np.float32)
    theta_all = d["theta_norm"][non_dry].astype(np.float32)
    effect_names_all = d["effect"][non_dry]
    onehot = np.zeros((len(effect_names_all), len(EFFECTS)), dtype=np.float32)
    for i, name in enumerate(EFFECTS):
        onehot[:, i] = (effect_names_all == name).astype(np.float32)
    x_full = np.concatenate([dry_all, onehot, theta_all], axis=1)

    train_mask_all = np.array([s in train_srcs for s in src_ids])
    test_mask_all = np.array([s in test_srcs for s in src_ids])

    print("H5 대리모델(전체 풀링) 학습 중 — 실제 레이블...")
    # SurrogateMLP는 입력이 (dry+onehot+theta) 통짜라 train_torch_model(별도 dry/theta 인자)과
    # 시그니처가 달라 전용 루프를 쓴다.
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    x_tr = torch.tensor(x_full[train_mask_all], device=device)
    dry_tr_t = torch.tensor(dry_all[train_mask_all], device=device)
    wet_tr_t = torch.tensor(wet_all[train_mask_all], device=device)
    x_te = torch.tensor(x_full[test_mask_all], device=device)
    dry_te_t = torch.tensor(dry_all[test_mask_all], device=device)
    wet_te_t = torch.tensor(wet_all[test_mask_all], device=device)

    def train_h5(shuffle_labels):
        model = SurrogateMLP(in_dim=x_full.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        wet_target = wet_tr_t.clone()
        if shuffle_labels:
            perm = torch.tensor(rng.permutation(len(wet_target)))
            wet_target = wet_target[perm]
        n_train = x_tr.shape[0]
        batch_size = min(64, n_train)
        model.train()
        for _epoch in range(args.epochs):
            perm = torch.randperm(n_train)
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                delta = model(x_tr[idx])
                pred = dry_tr_t[idx] + delta
                mse = F.mse_loss(pred, wet_target[idx])
                cos_loss = (1 - F.cosine_similarity(pred, wet_target[idx], dim=1)).mean()
                (mse + cos_loss).backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_te = dry_te_t + model(x_te)
            test_cos = F.cosine_similarity(pred_te, wet_te_t, dim=1).mean().item()
        return float(test_cos), model

    real_cos, h5_model = train_h5(shuffle_labels=False)
    print("H5 대리모델 셔플 통제 학습 중...")
    shuffled_cos, _ = train_h5(shuffle_labels=True)
    identity_cos_pooled = float(cosine_rows(dry_all[test_mask_all], wet_all[test_mask_all]).mean())

    # 이펙트별 H5 held-out 코사인 (같은 풀링 모델을 이펙트별로 슬라이스해 평가)
    h5_by_effect = {}
    with torch.no_grad():
        for e in EFFECTS:
            e_test = test_mask_all & (effect_names_all == e)
            x_e = torch.tensor(x_full[e_test], device=device)
            dry_e = torch.tensor(dry_all[e_test], device=device)
            wet_e = torch.tensor(wet_all[e_test], device=device)
            pred_e = dry_e + h5_model(x_e)
            h5_by_effect[e] = float(F.cosine_similarity(pred_e, wet_e, dim=1).mean().item())

    print("H1~H4 사다리(이펙트별) 계산 중...")
    ladder_results = {}
    for e in EFFECTS:
        dataset = build_effect_dataset(d, dry_by_src, e, theta_slots)
        ladder_results[e] = run_hierarchy_for_effect(e, dataset, train_srcs, test_srcs, device, args.epochs, args.lr, args.seed)

    print("그림 저장 중...")
    plot_hierarchy(ladder_results, h5_by_effect, shuffled_cos, out_dir / "hierarchy.png")
    plot_surrogate_quality(real_cos, shuffled_cos, identity_cos_pooled, out_dir / "surrogate_quality.png")

    # 대리모델 가중치 저장 — 03_jacobian.py / 04_text_alignment.py가 재사용
    torch.save(
        {
            "state_dict": h5_model.state_dict(),
            "in_dim": x_full.shape[1],
            "theta_slots": theta_slots,
            "effects": EFFECTS,
        },
        out_dir / "surrogate_model.pt",
    )

    print("중립 레벨 확인 중 (θ=0 앵커가 진짜 dry와 같은가)...")
    neutral_check = {}
    for e in EFFECTS:
        anchor_mask = (d["effect"] == e) & (d["is_anchor"])
        srcs = d["src_id"][anchor_mask]
        embs = d["embeddings"][anchor_mask]
        dry_arr = np.stack([dry_by_src[s] for s in srcs])
        neutral_check[e] = float(cosine_rows(embs, dry_arr).mean())

    results_path = out_dir / "results.json"
    results_json = {}
    if results_path.exists():
        with open(results_path) as f:
            results_json = json.load(f)

    results_json["meta"] = {
        "experiment_version": config["experiment_version"],
        "sampling": config["sampling"],
        "param_space": config["param_space"],
        "n_samples_per_source": config["n_samples_per_source"],
    }
    results_json["neutral_check"] = {"cos_by_effect": neutral_check}

    results_json["surrogate"] = {
        "held_out_cos_real": real_cos,
        "held_out_cos_shuffled": shuffled_cos,
        "held_out_cos_identity": identity_cos_pooled,
        "held_out_cos_by_effect": h5_by_effect,
        "hierarchy_H1_to_H5": {
            e: {
                "identity": ladder_results[e]["identity"],
                "H1": ladder_results[e]["H1"],
                "H2": ladder_results[e]["H2"],
                "H3": ladder_results[e]["H3"],
                "H4": ladder_results[e]["H4"],
                "H5": h5_by_effect[e],
                "shuffle_control": shuffled_cos,
            }
            for e in EFFECTS
        },
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "test_size": args.test_size,
        "n_train_sources": len(train_srcs),
        "n_test_sources": len(test_srcs),
    }
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"완료: {out_dir / 'surrogate_model.pt'}, {results_path}, {out_dir}/hierarchy.png, {out_dir}/surrogate_quality.png")


if __name__ == "__main__":
    main()
