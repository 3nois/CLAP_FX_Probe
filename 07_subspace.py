"""CLAP FX Probe — 07_subspace.py (3차, 부차 — 우선순위 낮음)

★ 주의: 이 분석은 "손잡이가 악기마다 다른가"에 답하지 못한다 (그건 03_jacobian.py의
악기 패밀리 분석이 담당). 이건 "왜 TokenSynth가 이 정보를 무시하는가"에 답하는 별개
질문이다 — 이펙트 방향이 악기 판별에 쓰이는 부분공간과 겹치면, TokenSynth의 조건화가
악기 정체성 축에 쏠려 이펙트 축을 "밀어낼" 수 있다는 가설의 2단계 예비 진단이다.

악기 패밀리 LDA로 판별 부분공간을 얻고, 각 파라미터의 야코비안 평균 방향이 그
부분공간에 얼마나 들어가는지(‖proj‖/‖v‖)를 부트스트랩 CI와 함께 낸다.

  0에 가까움 → 이펙트 방향이 악기 판별 축과 직교
  1에 가까움 → 악기 판별 부분공간 내부

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch.func import jacrev, vmap

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
    def __init__(self, in_dim, hidden=1024, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_embeddings(path: Path):
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_config(out_dir: Path):
    with open(out_dir / "embed_config.json") as f:
        return json.load(f)


def build_forward_fn(model, effect_name, theta_slots, theta_width, device):
    onehot = torch.zeros(len(EFFECTS), device=device)
    onehot[EFFECTS.index(effect_name)] = 1.0
    start, end = theta_slots[effect_name]
    pre, post = start, theta_width - end

    def f(theta_active, dry):
        theta_full = torch.cat([torch.zeros(pre, device=device), theta_active, torch.zeros(post, device=device)])
        x = torch.cat([dry, onehot, theta_full])
        delta = model(x.unsqueeze(0)).squeeze(0)
        return dry + delta

    return f


def mean_jacobian_direction(model, effect_name, param_idx, theta_dim, theta_slots, theta_width, dry_samples, device, seed, n_theta=15):
    f = build_forward_fn(model, effect_name, theta_slots, theta_width, device)
    rng = np.random.RandomState(seed)
    theta_draws = rng.uniform(0, 1, size=(n_theta, theta_dim)).astype(np.float32)
    theta_batch = np.repeat(theta_draws, len(dry_samples), axis=0)
    dry_batch = np.tile(dry_samples, (n_theta, 1))
    theta_t = torch.tensor(theta_batch, device=device)
    dry_t = torch.tensor(dry_batch, device=device)
    J = vmap(jacrev(f, argnums=0), in_dims=(0, 0))(theta_t, dry_t)
    return J[:, :, param_idx].mean(dim=0).detach().cpu().numpy()


def fit_lda_basis(dry_embeddings, family_labels):
    unique = np.unique(family_labels)
    n_components = min(len(unique) - 1, dry_embeddings.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(dry_embeddings, family_labels)
    basis, _ = np.linalg.qr(lda.scalings_[:, :n_components])
    return basis


def projection_ratio(vec, basis):
    proj = basis @ (basis.T @ vec)
    return float(np.linalg.norm(proj) / (np.linalg.norm(vec) + 1e-12))


def bootstrap_projection_ratio(vec, dry_embeddings, family_labels, src_ids, seed, n_boot=300, ci=0.95):
    unique_srcs = np.unique(src_ids)
    rng = np.random.RandomState(seed)
    ratios = []
    for _ in range(n_boot):
        boot_srcs = rng.choice(unique_srcs, size=len(unique_srcs), replace=True)
        rows = np.concatenate([np.where(src_ids == s)[0] for s in boot_srcs])
        try:
            basis = fit_lda_basis(dry_embeddings[rows], family_labels[rows])
        except Exception:
            continue
        ratios.append(projection_ratio(vec, basis))
    ratios = np.array(ratios)
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return float(ratios.mean()), float(np.percentile(ratios, lo_pct)), float(np.percentile(ratios, hi_pct)), int(len(ratios))


def plot_subspace_projection(results, out_path):
    keys = list(results.keys())
    means = [results[k]["mean"] for k in keys]
    ci_lo = [results[k]["ci_low"] for k in keys]
    ci_hi = [results[k]["ci_high"] for k in keys]
    colors = [COLORS[k.split(".")[0]] for k in keys]
    yerr = np.array([[m - lo, hi - m] for m, lo, hi in zip(means, ci_lo, ci_hi)]).T
    yerr = np.clip(yerr, 0, None)

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=yerr, capsize=3, color=colors, zorder=3)
    ax.axhline(0.0, color=GRID_COLOR, linewidth=1)
    ax.axhline(1.0, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("‖proj_instrument(J[:,i])‖ / ‖J[:,i]‖ (부트스트랩 95% CI)")
    ax.set_ylim(0, 1.1)
    ax.set_title("이펙트 방향의 악기 판별 부분공간 투영 비율 (부차 분석)")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="악기 판별 부분공간 투영 (3차, 부차)")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=300)
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
    model = SurrogateMLP(in_dim=ckpt["in_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d = load_embeddings(emb_path)
    config = load_config(emb_path.parent)
    theta_slots = {e: tuple(v) for e, v in config["theta_slots"].items()}
    theta_width = config["theta_width"]
    param_order_by_effect = config["param_order"]

    dry_mask = d["effect"] == "dry"
    dry_embeddings_all = d["embeddings"][dry_mask]
    dry_src_ids = d["src_id"][dry_mask]
    dry_families = d["instrument_family"][dry_mask]
    dry_by_src = dict(zip(dry_src_ids.tolist(), dry_embeddings_all))

    rng = np.random.RandomState(args.seed)
    unique_srcs = np.array(sorted(dry_by_src.keys()))
    dry_sample_srcs = rng.choice(unique_srcs, size=min(30, len(unique_srcs)), replace=False)
    dry_samples = np.stack([dry_by_src[s] for s in dry_sample_srcs]).astype(np.float32)

    print("파라미터별 야코비안 평균 방향 계산 중...")
    directions = {}
    for e in EFFECTS:
        param_order = param_order_by_effect[e]
        for i, pname in enumerate(param_order):
            key = f"{e}.{pname}"
            directions[key] = mean_jacobian_direction(model, e, i, len(param_order), theta_slots, theta_width, dry_samples, device, args.seed)

    print("악기 판별 부분공간 투영 비율(부트스트랩 CI) 계산 중...")
    results = {}
    for key, vec in directions.items():
        mean, lo, hi, n_used = bootstrap_projection_ratio(vec, dry_embeddings_all, dry_families, dry_src_ids, args.seed, n_boot=args.n_boot)
        results[key] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n_boot_used": n_used}

    print("그림 저장 중...")
    plot_subspace_projection(results, out_dir / "subspace_projection.png")

    results_path = out_dir / "results.json"
    results_json = {}
    if results_path.exists():
        with open(results_path) as f:
            results_json = json.load(f)
    results_json["subspace"] = {"projection_ratio_by_param": results}
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"완료: {results_path}, {out_dir}/subspace_projection.png")


if __name__ == "__main__":
    main()
