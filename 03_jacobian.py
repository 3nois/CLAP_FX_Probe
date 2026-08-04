"""CLAP FX Probe — 03_jacobian.py (3차 핵심)

02_surrogate.py가 학습한 residual MLP 대리모델의 야코비안 J = ∂e'/∂θ를 autograd로
계산해 분석한다.

  (a) 게이트 구조 — reverb의 wet_level이 실제로 다른 파라미터의 곱셈 게이트로
      작동하는지: ‖∂f/∂damping‖, ‖∂f/∂room_size‖, ‖∂f/∂width‖가 wet_level에
      단조 증가해야 한다 (Spearman ρ). 확인되면 야코비안 접근 자체가 검증된 것이고,
      안 되면 대리모델이 구조를 학습하지 못한 것이다.

  (b) 악기 패밀리별 손잡이 차이 — 이 프로젝트의 원래 질문. 파라미터별로 패밀리 간
      J 코사인의 평균/분산을 낸다. 높으면(>0.8) 공통 손잡이로 충분, 낮으면(<0.5)
      악기별 손잡이가 필요하다는 뜻.

  θ 간 코사인(같은 소스, 다른 θ에서 방향이 유지되는가)도 함께 산출한다.

★ J는 실제 CLAP의 미분이 아니라 학습된 근사의 미분이다. 02_surrogate.py의
surrogate_quality.png(held-out 코사인)를 반드시 함께 볼 것 — 낮으면 이 분석 전체가
무의미하다.

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
from scipy.stats import spearmanr
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
GATE_TARGET_COLORS = {"room_size": "#2a78d6", "damping": "#5598e7", "width": "#e34948"}  # width는 음성 통제라 경고색


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


class SurrogateMLP(nn.Module):
    """02_surrogate.py와 동일 구조 — state_dict 로딩을 위해 클래스 정의를 공유한다."""

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


def load_surrogate(out_dir: Path, device):
    ckpt = torch.load(out_dir / "surrogate_model.pt", map_location=device, weights_only=False)
    model = SurrogateMLP(in_dim=ckpt["in_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt["theta_slots"]


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


def batched_jacobian(f, theta_batch: torch.Tensor, dry_batch: torch.Tensor) -> torch.Tensor:
    """(B, 512, theta_dim) — vmap(jacrev)으로 배치 전체를 한 번에."""
    return vmap(jacrev(f, argnums=0), in_dims=(0, 0))(theta_batch, dry_batch)


def extract_oat_curve(model, effect_name, theta_slots, theta_width, device, param_idx, fixed_theta, dry_embedding, n_points=20):
    """다른 파라미터를 fixed_theta에 고정하고 param_idx 하나만 스윕한 사후 OAT 곡선.

    별도 실험 없이 대리모델 평가만으로 1·2차 OAT 결과와 비교 가능한 곡선을 얻는다.
    반환: (values, e_prime 시퀀스 [n_points, 512])
    """
    f = build_forward_fn(model, effect_name, theta_slots, theta_width, device)
    values = np.linspace(0.0, 1.0, n_points)
    dry_t = torch.tensor(dry_embedding.astype(np.float32), device=device)
    outputs = []
    with torch.no_grad():
        for v in values:
            theta = fixed_theta.copy()
            theta[param_idx] = v
            theta_t = torch.tensor(theta.astype(np.float32), device=device)
            outputs.append(f(theta_t, dry_t).cpu().numpy())
    return values, np.stack(outputs)


# ---------------------------------------------------------------------------
# (a) 게이트 구조 — wet_level이 room_size/damping/width의 곱셈 게이트인가
# ---------------------------------------------------------------------------


def gate_structure_analysis(model, theta_slots, theta_width, dry_by_src, device, seed, n_wet_bins=25, n_param_draws=12, n_dry_samples=20):
    effect_name = "reverb"
    param_order = ["wet_level", "room_size", "damping", "width", "freeze_mode"]
    f = build_forward_fn(model, effect_name, theta_slots, theta_width, device)

    rng = np.random.RandomState(seed)
    unique_srcs = np.array(sorted(dry_by_src.keys()))
    chosen_srcs = rng.choice(unique_srcs, size=min(n_dry_samples, len(unique_srcs)), replace=False)
    dry_samples = np.stack([dry_by_src[s] for s in chosen_srcs]).astype(np.float32)

    wet_levels = np.linspace(0.0, 1.0, n_wet_bins)
    targets = {"room_size": 1, "damping": 2, "width": 3}
    norm_by_target = {name: [] for name in targets}

    for wl in wet_levels:
        other = rng.uniform(0, 1, size=(n_param_draws, 3))  # room_size, damping, width
        freeze = rng.randint(0, 2, size=(n_param_draws,)).astype(np.float32)

        theta_rows, dry_rows = [], []
        for pi in range(n_param_draws):
            for dry_vec in dry_samples:
                theta_rows.append([wl, other[pi, 0], other[pi, 1], other[pi, 2], freeze[pi]])
                dry_rows.append(dry_vec)
        theta_t = torch.tensor(np.array(theta_rows, dtype=np.float32), device=device)
        dry_t = torch.tensor(np.array(dry_rows, dtype=np.float32), device=device)
        J = batched_jacobian(f, theta_t, dry_t)  # (N, 512, 5)

        for name, idx in targets.items():
            col_norms = J[:, :, idx].norm(dim=1)
            norm_by_target[name].append(float(col_norms.mean().item()))

    result = {}
    for name in targets:
        rho, pvalue = spearmanr(wet_levels, norm_by_target[name])
        result[name] = {
            "norm_by_wet_level": norm_by_target[name],
            "spearman_vs_wet_level": float(rho),
            "spearman_pvalue": float(pvalue),
        }
    result["wet_levels"] = wet_levels.tolist()
    return result


# ---------------------------------------------------------------------------
# (b) 악기 패밀리별 J 비교 — 이 프로젝트의 원래 질문
# ---------------------------------------------------------------------------


def family_jacobian_analysis(model, theta_slots, theta_width, param_order_by_effect, d, dry_by_src, device, seed, n_theta_draws=10, n_dry_per_family=15):
    dry_mask = d["effect"] == "dry"
    src_to_family = dict(zip(d["src_id"][dry_mask].tolist(), d["instrument_family"][dry_mask].tolist()))
    families = sorted(set(src_to_family.values()))

    rng = np.random.RandomState(seed)
    result = {}

    for effect_name in EFFECTS:
        param_order = param_order_by_effect[effect_name]
        theta_dim = len(param_order)
        f = build_forward_fn(model, effect_name, theta_slots, theta_width, device)
        theta_draws = rng.uniform(0, 1, size=(n_theta_draws, theta_dim)).astype(np.float32)

        family_mean_J = {}
        for family in families:
            srcs = [s for s, fam in src_to_family.items() if fam == family and s in dry_by_src]
            if len(srcs) < 2:
                continue
            chosen = rng.choice(srcs, size=min(n_dry_per_family, len(srcs)), replace=False)
            dry_arr = np.stack([dry_by_src[s] for s in chosen]).astype(np.float32)

            theta_batch = np.repeat(theta_draws, len(chosen), axis=0)
            dry_batch = np.tile(dry_arr, (n_theta_draws, 1))
            theta_t = torch.tensor(theta_batch, device=device)
            dry_t = torch.tensor(dry_batch, device=device)
            J = batched_jacobian(f, theta_t, dry_t)  # (N, 512, theta_dim)
            family_mean_J[family] = J.mean(dim=0).detach().cpu().numpy()  # (512, theta_dim)

        for pi, pname in enumerate(param_order):
            key = f"{effect_name}.{pname}"
            fam_list = list(family_mean_J.keys())
            cos_vals = []
            for i in range(len(fam_list)):
                for j in range(i + 1, len(fam_list)):
                    a = family_mean_J[fam_list[i]][:, pi]
                    b = family_mean_J[fam_list[j]][:, pi]
                    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                    cos_vals.append(cos)
            result[key] = {
                "cosine_mean": float(np.mean(cos_vals)) if cos_vals else None,
                "cosine_std": float(np.std(cos_vals)) if cos_vals else None,
                "n_family_pairs": len(cos_vals),
                "n_families": len(fam_list),
            }
    return result


# ---------------------------------------------------------------------------
# θ 간 코사인(같은 소스, 다른 θ에서 방향이 유지되는가) + jacobian_norm_mean
# ---------------------------------------------------------------------------


def theta_dependence_analysis(model, theta_slots, theta_width, param_order_by_effect, dry_by_src, device, seed, n_theta_points=15, n_dry_samples=12):
    rng = np.random.RandomState(seed)
    unique_srcs = np.array(sorted(dry_by_src.keys()))
    chosen_srcs = rng.choice(unique_srcs, size=min(n_dry_samples, len(unique_srcs)), replace=False)
    dry_samples = np.stack([dry_by_src[s] for s in chosen_srcs]).astype(np.float32)

    cosine_result, norm_result = {}, {}
    for effect_name in EFFECTS:
        param_order = param_order_by_effect[effect_name]
        theta_dim = len(param_order)
        f = build_forward_fn(model, effect_name, theta_slots, theta_width, device)
        theta_points = rng.uniform(0, 1, size=(n_theta_points, theta_dim)).astype(np.float32)
        theta_t = torch.tensor(theta_points, device=device)

        for pi, pname in enumerate(param_order):
            key = f"{effect_name}.{pname}"
            per_dry_cos, per_dry_norm = [], []
            for dry_vec in dry_samples:
                dry_batch = np.tile(dry_vec[None, :], (n_theta_points, 1))
                dry_t = torch.tensor(dry_batch, device=device)
                J = batched_jacobian(f, theta_t, dry_t)  # (n_theta_points, 512, theta_dim)
                cols = J[:, :, pi].detach().cpu().numpy()
                norms = np.linalg.norm(cols, axis=1)
                per_dry_norm.append(float(norms.mean()))
                unit = cols / np.clip(norms[:, None], 1e-12, None)
                sim = unit @ unit.T
                iu = np.triu_indices_from(sim, k=1)
                per_dry_cos.append(float(sim[iu].mean()))
            cosine_result[key] = {"cosine_mean": float(np.mean(per_dry_cos)), "cosine_std": float(np.std(per_dry_cos))}
            norm_result[key] = float(np.mean(per_dry_norm))
    return cosine_result, norm_result


def plot_jacobian_gate(gate_result, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
    wet_levels = gate_result["wet_levels"]
    for ax, name in zip(axes, ["room_size", "damping", "width"]):
        r = gate_result[name]
        ax.plot(wet_levels, r["norm_by_wet_level"], color=GATE_TARGET_COLORS[name], linewidth=2, marker="o", markersize=3)
        tag = " (음성 통제)" if name == "width" else ""
        ax.set_title(f"‖∂f/∂{name}‖{tag}\nSpearman ρ={r['spearman_vs_wet_level']:.2f}")
        ax.set_xlabel("wet_level (정규화)")
        style_axis(ax)
    axes[0].set_ylabel("야코비안 열 노름 (다른 파라미터/소스 평균)")
    fig.suptitle("게이트 구조 검증 — wet_level이 커질수록 다른 파라미터의 영향력도 커져야 한다")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_jacobian_by_family(family_result, param_order_by_effect, negative_control_params, out_path):
    keys = [f"{e}.{p}" for e in EFFECTS for p in param_order_by_effect[e]]
    means = [family_result[k]["cosine_mean"] or 0.0 for k in keys]
    stds = [family_result[k]["cosine_std"] or 0.0 for k in keys]
    colors = []
    for k in keys:
        effect = k.split(".")[0]
        colors.append("#e34948" if k in negative_control_params else COLORS[effect])

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, zorder=3)
    ax.axhline(0.8, color=INK_SECONDARY, linestyle="--", linewidth=1, label="0.8 (공통 손잡이 기준)")
    ax.axhline(0.5, color=INK_SECONDARY, linestyle=":", linewidth=1, label="0.5 (악기별 손잡이 기준)")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("악기 패밀리 간 야코비안 열 코사인 (평균 ± 표준편차)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("파라미터별 악기 패밀리 간 손잡이 방향 일치도")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="대리모델 야코비안 분석 (3차 핵심)")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device == "mps":
        import os

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    out_dir = Path(args.out)
    emb_path = Path(args.embeddings)
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path}가 없습니다. 먼저 01_embed.py를 실행하세요.")
    if not (out_dir / "surrogate_model.pt").exists():
        raise FileNotFoundError(f"{out_dir / 'surrogate_model.pt'}가 없습니다. 먼저 02_surrogate.py를 실행하세요.")

    device = torch.device(args.device)
    model, theta_slots = load_surrogate(out_dir, device)

    d = load_embeddings(emb_path)
    config = load_config(emb_path.parent if emb_path.parent != Path("") else out_dir)
    theta_slots = {e: tuple(v) for e, v in config["theta_slots"].items()}
    theta_width = config["theta_width"]
    param_order_by_effect = config["param_order"]
    negative_control_params = config["negative_control_params"]

    dry_mask = d["effect"] == "dry"
    dry_by_src = dict(zip(d["src_id"][dry_mask].tolist(), d["embeddings"][dry_mask]))

    print("게이트 구조 분석 중 (reverb wet_level)...")
    gate_result = gate_structure_analysis(model, theta_slots, theta_width, dry_by_src, device, args.seed)

    print("악기 패밀리별 야코비안 비교 중...")
    family_result = family_jacobian_analysis(model, theta_slots, theta_width, param_order_by_effect, d, dry_by_src, device, args.seed)

    print("θ 간 코사인(방향 안정성) + 야코비안 노름 계산 중...")
    theta_dep_result, norm_result = theta_dependence_analysis(model, theta_slots, theta_width, param_order_by_effect, dry_by_src, device, args.seed)

    print("그림 저장 중...")
    plot_jacobian_gate(gate_result, out_dir / "jacobian_gate.png")
    plot_jacobian_by_family(family_result, param_order_by_effect, negative_control_params, out_dir / "jacobian_by_family.png")

    results_path = out_dir / "results.json"
    results_json = {}
    if results_path.exists():
        with open(results_path) as f:
            results_json = json.load(f)

    params = results_json.setdefault("params", {})
    for effect_name in EFFECTS:
        for pname in param_order_by_effect[effect_name]:
            key = f"{effect_name}.{pname}"
            entry = params.setdefault(key, {})
            entry["effect"] = effect_name
            entry["name"] = pname
            entry["range"] = config["param_space"][effect_name][pname]
            entry["is_negative_control"] = key in negative_control_params
            entry["jacobian_norm_mean"] = norm_result[key]
            entry["jacobian_theta_dependence"] = theta_dep_result[key]
            entry["jacobian_family_cosine"] = family_result[key]
            entry["gate_spearman_vs_wet_level"] = gate_result[pname]["spearman_vs_wet_level"] if pname in gate_result else None

    results_json["jacobian_gate_analysis"] = {
        k: {"spearman_vs_wet_level": v["spearman_vs_wet_level"], "spearman_pvalue": v["spearman_pvalue"]}
        for k, v in gate_result.items()
        if k != "wet_levels"
    }

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"완료: {results_path}, {out_dir}/jacobian_gate.png, {out_dir}/jacobian_by_family.png")


if __name__ == "__main__":
    main()
