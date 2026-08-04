"""CLAP FX Probe — 02_surrogate.py (4차 개정: H1~H5 위계 폐기 → 입력 마스킹 ablation)

01_embed.py가 만든 (e_dry, θ, e_wet) 데이터로 residual MLP 대리모델을 학습한다.

  e' = e_dry + MLP(e_dry, effect_onehot, θ)

3차의 H1~H5 "위계"는 폐기한다. H3(θ만, 비선형)가 H2(방향고정+선형 스케일)보다 낮게
나왔는데(0.650 vs 0.974) H2 ⊂ H3라 이건 포함 관계 위반 — 구현 오류다. 근본적으로도
H3(J=J(θ), e_dry 무관)와 H4(J=J(e_dry), θ 무관)는 서로를 포함하지 않는다 — 사다리가
아니라 다이아몬드였다.

대신 **입력 마스킹 ablation**을 쓴다. 동일 아키텍처·학습설정·시드로, 입력 슬롯만
0으로 가려서 네 가지를 학습한다:

  M0    MLP(effect_onehot)               θ, e_dry 둘 다 차단
  M_th  MLP(effect_onehot, θ)            e_dry 차단
  M_e   MLP(effect_onehot, e_dry)        θ 차단
  M_the MLP(effect_onehot, e_dry, θ)     전부 사용 (기존 대리모델과 동일)

residual 파라미터화(e' = e_dry + Δ)는 네 모델 모두 유지한다 — Δ 계산에 쓰이는
"입력"만 가리고, 최종 덧셈은 항상 진짜 e_dry를 쓴다.

분산 분해:
  d_total = M_the − M0
  d_th    = M_th  − M0     (파라미터 의존이 버는 몫)
  d_e     = M_e   − M0     (소스 의존이 버는 몫)
  d_int   = d_total − d_th − d_e   (상호작용에서만 나오는 몫 — 03_jacobian.py의
                                     악기 패밀리 코사인 0.62~0.77과 교차 검증됨)

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
ABLATION_COLORS = {"M0": "#c3c2b7", "M_th": "#5598e7", "M_e": "#eb6834", "M_the": "#104281"}


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


class SurrogateMLP(nn.Module):
    """524(e_dry 512 + 이펙트 원핫 3 + θ 9) → 512(Δ). GELU + LayerNorm.

    마스킹 ablation의 네 변형(M0/M_th/M_e/M_the) 모두 이 클래스를 그대로 쓴다 —
    아키텍처를 바꾸면 비교가 성립하지 않는다. 차단은 입력 텐서의 해당 슬롯을 0으로
    채우는 방식으로만 구현한다 (build_masked_input 참고).
    """

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


def build_masked_input(dry, onehot, theta, use_dry: bool, use_theta: bool):
    d = dry if use_dry else np.zeros_like(dry)
    t = theta if use_theta else np.zeros_like(theta)
    return np.concatenate([d, onehot, t], axis=1).astype(np.float32)


ABLATION_VARIANTS = {
    "M0": {"use_dry": False, "use_theta": False},
    "M_th": {"use_dry": False, "use_theta": True},
    "M_e": {"use_dry": True, "use_theta": False},
    "M_the": {"use_dry": True, "use_theta": True},
}


def train_ablation_variant(x_full, dry_true, wet_all, train_mask, test_mask, device, epochs, lr, seed, shuffle_labels=False):
    """x_full은 이미 마스킹이 적용된 입력. residual 덧셈은 항상 진짜 dry_true를 쓴다."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    x_tr = torch.tensor(x_full[train_mask], device=device)
    dry_tr = torch.tensor(dry_true[train_mask], device=device)
    wet_tr = torch.tensor(wet_all[train_mask], device=device)
    if shuffle_labels:
        perm = rng.permutation(len(wet_tr))
        wet_tr = wet_tr[perm]

    x_te = torch.tensor(x_full[test_mask], device=device)
    dry_te = torch.tensor(dry_true[test_mask], device=device)
    wet_te = torch.tensor(wet_all[test_mask], device=device)

    model = SurrogateMLP(in_dim=x_full.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = x_tr.shape[0]
    batch_size = min(64, n_train)

    model.train()
    for _epoch in range(epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            delta = model(x_tr[idx])
            pred = dry_tr[idx] + delta
            mse = F.mse_loss(pred, wet_tr[idx])
            cos_loss = (1 - F.cosine_similarity(pred, wet_tr[idx], dim=1)).mean()
            (mse + cos_loss).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_te = dry_te + model(x_te)
        test_cos_overall = float(F.cosine_similarity(pred_te, wet_te, dim=1).mean().item())
        per_row_cos = F.cosine_similarity(pred_te, wet_te, dim=1).cpu().numpy()

    return test_cos_overall, per_row_cos, model


def plot_ablation(ablation_by_effect, decomposition_by_effect, out_path):
    variants = ["M0", "M_th", "M_e", "M_the"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150, sharey=True)
    for ax, effect_name in zip(axes, EFFECTS):
        scores = [ablation_by_effect[effect_name][v] for v in variants]
        x = np.arange(len(variants))
        ax.bar(x, scores, color=[ABLATION_COLORS[v] for v in variants], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        dec = decomposition_by_effect[effect_name]
        ax.set_title(f"{effect_name}\nd_th={dec['d_th']:.3f} d_e={dec['d_e']:.3f} d_int={dec['d_int']:.3f}")
        style_axis(ax)
    axes[0].set_ylabel("Held-out cos(e', e_wet)")
    fig.suptitle("입력 마스킹 ablation — θ/e_dry 각각의 기여 + 상호작용(d_int)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_variance_decomposition(decomposition_by_effect, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x = np.arange(len(EFFECTS))
    width = 0.25
    d_th = [decomposition_by_effect[e]["d_th"] for e in EFFECTS]
    d_e = [decomposition_by_effect[e]["d_e"] for e in EFFECTS]
    d_int = [decomposition_by_effect[e]["d_int"] for e in EFFECTS]

    ax.bar(x - width, d_th, width, label="d_th (θ 의존)", color="#5598e7", zorder=3)
    ax.bar(x, d_e, width, label="d_e (소스 의존)", color="#eb6834", zorder=3)
    ax.bar(x + width, d_int, width, label="d_int (상호작용) ★", color="#e34948", zorder=3)
    ax.axhline(0, color=GRID_COLOR, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(EFFECTS)
    ax.set_ylabel("held-out cos 기여분")
    ax.set_title("분산 분해 — d_int가 크면 파라미터·소스가 얽혀 있다는 직접 증거\n(03_jacobian.py 악기 패밀리 코사인과 교차 검증)")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_surrogate_quality(real_cos, shuffled_cos, identity_cos, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    labels = ["identity\n(e'=e_dry)", "셔플\n(동일 용량)", "대리모델\n(M_the, 실제 레이블)"]
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
    parser = argparse.ArgumentParser(description="residual MLP 대리모델 학습 + 입력 마스킹 ablation (4차)")
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
        raise FileNotFoundError(f"{emb_path}가 없습니다. 3차 embeddings.npz를 그대로 재사용하세요 (재추출 불필요).")

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

    non_dry = d["effect"] != "dry"
    src_ids = d["src_id"][non_dry]
    dry_all = np.stack([dry_by_src[s] for s in src_ids]).astype(np.float32)
    wet_all = d["embeddings"][non_dry].astype(np.float32)
    theta_all = d["theta_norm"][non_dry].astype(np.float32)
    effect_names_all = d["effect"][non_dry]
    onehot_all = np.zeros((len(effect_names_all), len(EFFECTS)), dtype=np.float32)
    for i, name in enumerate(EFFECTS):
        onehot_all[:, i] = (effect_names_all == name).astype(np.float32)

    train_mask_all = np.array([s in train_srcs for s in src_ids])
    test_mask_all = np.array([s in test_srcs for s in src_ids])

    print("입력 마스킹 ablation 4종 학습 중 (M0, M_th, M_e, M_the — 동일 아키텍처/설정/시드)...")
    ablation_models = {}
    ablation_overall = {}
    ablation_per_row = {}
    for variant, flags in ABLATION_VARIANTS.items():
        print(f"  {variant} (use_dry={flags['use_dry']}, use_theta={flags['use_theta']}) 학습 중...")
        x_full = build_masked_input(dry_all, onehot_all, theta_all, flags["use_dry"], flags["use_theta"])
        cos_overall, per_row_cos, model = train_ablation_variant(
            x_full, dry_all, wet_all, train_mask_all, test_mask_all, device, args.epochs, args.lr, args.seed
        )
        ablation_models[variant] = model
        ablation_overall[variant] = cos_overall
        ablation_per_row[variant] = per_row_cos
        if variant == "M_the":
            x_full_the = x_full  # surrogate_model.pt 저장용 in_dim 기록

    print("M_the 셔플 통제 학습 중 (동일 용량)...")
    x_full_the_masked = build_masked_input(dry_all, onehot_all, theta_all, True, True)
    shuffled_cos, _shuffled_per_row, _ = train_ablation_variant(
        x_full_the_masked, dry_all, wet_all, train_mask_all, test_mask_all, device, args.epochs, args.lr, args.seed, shuffle_labels=True
    )
    identity_cos_pooled = float(cosine_rows(dry_all[test_mask_all], wet_all[test_mask_all]).mean())
    real_cos = ablation_overall["M_the"]

    # 이펙트별 held-out 코사인 슬라이스 + 분산 분해
    print("이펙트별 분산 분해 계산 중...")
    ablation_by_effect = {e: {} for e in EFFECTS}
    decomposition_by_effect = {}
    test_effect_names = effect_names_all[test_mask_all]
    for variant, model in ablation_models.items():
        for e in EFFECTS:
            e_local_mask = test_effect_names == e
            ablation_by_effect[e][variant] = float(ablation_per_row[variant][e_local_mask].mean())

    for e in EFFECTS:
        m0 = ablation_by_effect[e]["M0"]
        m_th = ablation_by_effect[e]["M_th"]
        m_e = ablation_by_effect[e]["M_e"]
        m_the = ablation_by_effect[e]["M_the"]
        d_total = m_the - m0
        d_th = m_th - m0
        d_e = m_e - m0
        d_int = d_total - d_th - d_e
        decomposition_by_effect[e] = {
            "d_total": d_total, "d_th": d_th, "d_e": d_e, "d_int": d_int,
        }

    print("그림 저장 중...")
    plot_ablation(ablation_by_effect, decomposition_by_effect, out_dir / "ablation.png")
    plot_variance_decomposition(decomposition_by_effect, out_dir / "variance_decomposition.png")
    plot_surrogate_quality(real_cos, shuffled_cos, identity_cos_pooled, out_dir / "surrogate_quality.png")

    # M_the를 surrogate_model.pt로 저장 — 03/05/06/07이 재사용 (인터페이스 동일 유지)
    torch.save(
        {
            "state_dict": ablation_models["M_the"].state_dict(),
            "in_dim": x_full_the.shape[1],
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

    # 3차의 hierarchy_H1_to_H5는 폐기 — 포함 관계가 성립하지 않는 잘못된 설계였다 (README 참고).
    results_json.pop("hierarchy", None)
    results_json["surrogate"] = {
        "held_out_cos_real": real_cos,
        "held_out_cos_shuffled": shuffled_cos,
        "held_out_cos_identity": identity_cos_pooled,
        "held_out_cos_by_effect": {e: ablation_by_effect[e]["M_the"] for e in EFFECTS},
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "test_size": args.test_size,
        "n_train_sources": len(train_srcs),
        "n_test_sources": len(test_srcs),
    }

    results_json["ablation"] = {
        "note": "3차의 H1~H5 위계를 대체. M0/M_th/M_e/M_the는 동일 아키텍처·학습설정·시드,"
        " 입력 슬롯만 0으로 마스킹. d_int = d_total - d_th - d_e가 상호작용(파라미터-소스 얽힘)의 직접 증거.",
        "by_effect": ablation_by_effect,
        "decomposition": decomposition_by_effect,
    }

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"완료: {out_dir / 'surrogate_model.pt'}, {results_path}, {out_dir}/ablation.png, "
          f"{out_dir}/variance_decomposition.png, {out_dir}/surrogate_quality.png")


if __name__ == "__main__":
    main()
