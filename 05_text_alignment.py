"""CLAP FX Probe — 05_text_alignment.py (4차 개정: 부트스트랩 CI 추가)

캡션 가설(CLAP은 캡션에 흔한 개념만 인코딩한다)을 CLAP 자신의 cross-modal 성질로
검증한다. 텍스트 방향 u = e_text("distorted guitar") − e_text("guitar")와, 대리모델
야코비안에서 뽑은 오디오 방향 v(그 파라미터의 평균 방향)를 비교해 cos(v, u)를 잰다.

  distortion : ("distorted <악기>", "<악기>") 등, drive_db 방향과 비교
  reverb     : ("<악기> in a large hall", "<악기>") 등, wet_level 방향과 비교
  highshelf  : ("bright <악기>", "<악기>") 등, gain_db(+) 방향과 비교
               ※ highshelf에 대응하는 정확한 관용 표현이 없어 "밝다/선명하다" 계열로
                 근사한다 — 이 근사가 불완전하다는 점을 README에 명시했다.

통제 2종:
  (a) 무작위 텍스트 방향 — 오디오 이펙트와 무관한 수식어 쌍. 우연 수준 기준선.
  (b) 교차 정렬 — cos(v_distortion, u_reverb) 등. 자기 정렬도가 교차보다 높아야 한다.

4차: 3차는 정렬도에 CI가 없어 유의성을 판정할 수 없었다. 텍스트 쌍을 복원추출로
부트스트랩해 정렬도·통제 각각의 CI, 그리고 "격차"(정렬도 − 통제)의 CI를 낸다 — 격차의
CI가 0을 포함하지 않아야 유의하다고 볼 수 있다. 3차에서 highshelf의 "bright" 방향이
자기 오디오 방향(0.031)보다 distortion 오디오 방향(0.138)과 더 잘 맞는 역전이 있었는데,
이게 CI 기준으로도 유의한지 자동으로 확인한다(reversal_check).

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

# 텍스트 캡션 그룹과, 그에 대응하는 대표 파라미터(신호 방향을 뽑을 대상).
# highshelf는 gain_db > 0 (부스트) 방향만 쓴다 — "bright/crisp"는 부스트 쪽 근사이지 컷 쪽이 아니다.
REPRESENTATIVE_PARAM = {"reverb": "wet_level", "distortion": "drive_db", "highshelf": "gain_db"}
HIGHSHELF_SIGN = "positive"


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


def batched_jacobian(f, theta_batch, dry_batch):
    return vmap(jacrev(f, argnums=0), in_dims=(0, 0))(theta_batch, dry_batch)


def audio_direction_for_param(model, effect_name, param_name, param_order, theta_slots, theta_width, dry_by_src, device, seed, n_theta=15, n_dry=30, sign=None):
    """J[:, param]의 평균 방향(정규화 전) — 여러 θ/소스에서 평균."""
    param_idx = param_order.index(param_name)
    theta_dim = len(param_order)
    f = build_forward_fn(model, effect_name, theta_slots, theta_width, device)

    rng = np.random.RandomState(seed)
    unique_srcs = np.array(sorted(dry_by_src.keys()))
    chosen_srcs = rng.choice(unique_srcs, size=min(n_dry, len(unique_srcs)), replace=False)
    dry_samples = np.stack([dry_by_src[s] for s in chosen_srcs]).astype(np.float32)

    theta_draws = rng.uniform(0, 1, size=(n_theta, theta_dim)).astype(np.float32)
    if sign == "positive":
        # gain_db 정규화 [0,1]에서 0.5보다 큰 쪽이 부스트(+) — signed 파라미터는 정규화가
        # (v-lo)/(hi-lo)이므로 0.5가 0dB에 대응한다.
        theta_draws[:, param_idx] = rng.uniform(0.5, 1.0, size=n_theta).astype(np.float32)
    elif sign == "negative":
        theta_draws[:, param_idx] = rng.uniform(0.0, 0.5, size=n_theta).astype(np.float32)

    theta_batch = np.repeat(theta_draws, len(dry_samples), axis=0)
    dry_batch = np.tile(dry_samples, (n_theta, 1))
    theta_t = torch.tensor(theta_batch, device=device)
    dry_t = torch.tensor(dry_batch, device=device)
    J = batched_jacobian(f, theta_t, dry_t)  # (N, 512, theta_dim)
    mean_direction = J[:, :, param_idx].mean(dim=0).detach().cpu().numpy()
    return mean_direction


def load_text_directions(text_npz_path: Path):
    """그룹별 평균 텍스트 방향 u = mean(e_pos - e_neg), 그리고 부트스트랩용 원 쌍별 diff 벡터."""
    data = np.load(text_npz_path, allow_pickle=False)
    embeddings = data["embeddings"]
    groups = data["pair_group"]
    pos_idx = data["pair_pos_idx"]
    neg_idx = data["pair_neg_idx"]

    directions = {}
    diffs_by_group = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        diffs = embeddings[pos_idx[mask]] - embeddings[neg_idx[mask]]
        diffs_by_group[group] = diffs
        directions[group] = diffs.mean(axis=0)
    return directions, diffs_by_group


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


N_BOOTSTRAP = 2000
CI_PERCENTILES = [2.5, 97.5]


def bootstrap_cosine_dist(audio_vec, diffs, rng, n_boot=N_BOOTSTRAP):
    """텍스트 쌍을 복원추출로 재표집해 매번 평균 방향을 다시 내고, audio_vec과의 코사인 분포를 반환.

    diffs: (n_pairs, 512) 이 그룹에 속한 개별 쌍의 e_pos - e_neg. n_pairs가 작으므로(20~30)
    전체 재표집 인덱스를 한 번에 만들어 벡터화한다.
    """
    n = diffs.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_dirs = diffs[idx].mean(axis=1)  # (n_boot, 512)
    num = boot_dirs @ audio_vec
    denom = np.linalg.norm(boot_dirs, axis=1) * np.linalg.norm(audio_vec) + 1e-12
    return num / denom


def ci_summary(point_estimate, dist):
    lo, hi = np.percentile(dist, CI_PERCENTILES)
    return {
        "point": float(point_estimate),
        "boot_mean": float(np.mean(dist)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def gap_ci_summary(self_dist, control_dist):
    """자기 정렬 − 통제. 두 분포는 서로 다른(독립적인) 재표집이므로 그대로 뺀다.
    CI가 0을 포함하지 않아야(특히 하한이 0보다 커야) 격차가 유의하다고 볼 수 있다."""
    gap_dist = self_dist - control_dist
    lo, hi = np.percentile(gap_dist, CI_PERCENTILES)
    return {
        "point": float(np.mean(self_dist) - np.mean(control_dist)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "significant_positive": bool(lo > 0),
    }


def plot_text_alignment(self_align_ci, random_control_ci_by_effect, cross_align_ci, out_path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    labels = [f"{e}\n(자기 정렬)" for e in EFFECTS] + [f"{e}\n(통제)" for e in EFFECTS]
    values = [self_align_ci[e]["point"] for e in EFFECTS] + [random_control_ci_by_effect[e]["point"] for e in EFFECTS]
    err_lo = [self_align_ci[e]["point"] - self_align_ci[e]["ci_lo"] for e in EFFECTS] + [
        random_control_ci_by_effect[e]["point"] - random_control_ci_by_effect[e]["ci_lo"] for e in EFFECTS
    ]
    err_hi = [self_align_ci[e]["ci_hi"] - self_align_ci[e]["point"] for e in EFFECTS] + [
        random_control_ci_by_effect[e]["ci_hi"] - random_control_ci_by_effect[e]["point"] for e in EFFECTS
    ]
    colors = [COLORS[e] for e in EFFECTS] + [COLORS["baseline"]] * len(EFFECTS)

    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, zorder=3, label="자기 정렬 / 통제 (95% CI)")
    ax.errorbar(x, values, yerr=[err_lo, err_hi], fmt="none", ecolor=INK_SECONDARY, elinewidth=1.2, capsize=4, zorder=4)

    # 교차 정렬은 옅은 색으로 겹쳐 그린다 (자기 정렬 옆에 비교되도록)
    cross_x, cross_y = [], []
    for i, e in enumerate(EFFECTS):
        for other in EFFECTS:
            if other == e:
                continue
            cross_x.append(i)
            cross_y.append(cross_align_ci[f"{e}_vs_{other}"]["point"])
    ax.scatter(cross_x, cross_y, color="#111111", marker="x", s=40, zorder=5, label="교차 정렬 (다른 이펙트 텍스트)")

    ax.axhline(0, color=GRID_COLOR, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("cos(오디오 방향 v, 텍스트 방향 u)")
    ax.set_title("텍스트-오디오 방향 정렬 검증 (오차막대 = 부트스트랩 95% CI)")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="텍스트-오디오 방향 정렬 검증 (3차)")
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
    text_path = out_dir / "text_embeddings.npz"
    if not text_path.exists():
        raise FileNotFoundError(f"{text_path}가 없습니다. 01_embed.py가 텍스트 임베딩을 생성했는지 확인하세요 (--skip-text 옵션을 안 썼는지).")
    if not (out_dir / "surrogate_model.pt").exists():
        raise FileNotFoundError(f"{out_dir / 'surrogate_model.pt'}가 없습니다. 먼저 02_surrogate.py를 실행하세요.")

    device = torch.device(args.device)
    model, theta_slots_raw = load_surrogate(out_dir, device)

    d = load_embeddings(emb_path)
    config = load_config(emb_path.parent)
    theta_slots = {e: tuple(v) for e, v in config["theta_slots"].items()}
    theta_width = config["theta_width"]
    param_order_by_effect = config["param_order"]

    dry_mask = d["effect"] == "dry"
    dry_by_src = dict(zip(d["src_id"][dry_mask].tolist(), d["embeddings"][dry_mask]))

    print("텍스트 방향 로딩 중...")
    text_directions, text_diffs = load_text_directions(text_path)
    random_direction = text_directions.get("random_control")
    random_diffs = text_diffs.get("random_control")
    if random_direction is None:
        raise RuntimeError("text_embeddings.npz에 random_control 그룹이 없습니다.")

    print("이펙트별 오디오 방향(야코비안 평균) 계산 중...")
    audio_directions = {}
    for e in EFFECTS:
        param = REPRESENTATIVE_PARAM[e]
        sign = HIGHSHELF_SIGN if e == "highshelf" else None
        audio_directions[e] = audio_direction_for_param(
            model, e, param, param_order_by_effect[e], theta_slots, theta_width, dry_by_src, device, args.seed, sign=sign
        )

    self_align = {e: cosine(audio_directions[e], text_directions[e]) for e in EFFECTS}
    random_control_align = {e: cosine(audio_directions[e], random_direction) for e in EFFECTS}
    cross_align = {}
    for e in EFFECTS:
        for other in EFFECTS:
            if other == e:
                continue
            cross_align[f"{e}_vs_{other}"] = cosine(audio_directions[e], text_directions[other])

    print(f"부트스트랩 CI 계산 중 (n_boot={N_BOOTSTRAP}, 텍스트 쌍 복원추출)...")
    boot_rng = np.random.default_rng(args.seed)

    self_boot_dist = {e: bootstrap_cosine_dist(audio_directions[e], text_diffs[e], boot_rng) for e in EFFECTS}
    control_boot_dist = {e: bootstrap_cosine_dist(audio_directions[e], random_diffs, boot_rng) for e in EFFECTS}
    cross_boot_dist = {}
    for e in EFFECTS:
        for other in EFFECTS:
            if other == e:
                continue
            cross_boot_dist[f"{e}_vs_{other}"] = bootstrap_cosine_dist(audio_directions[e], text_diffs[other], boot_rng)

    self_align_ci = {e: ci_summary(self_align[e], self_boot_dist[e]) for e in EFFECTS}
    random_control_ci = {e: ci_summary(random_control_align[e], control_boot_dist[e]) for e in EFFECTS}
    cross_align_ci = {k: ci_summary(cross_align[k], cross_boot_dist[k]) for k in cross_align}

    # 격차(자기 정렬 − 통제) CI — 이게 유의성 판정 기준이다.
    gap_ci = {e: gap_ci_summary(self_boot_dist[e], control_boot_dist[e]) for e in EFFECTS}

    # 역전 검사: 이펙트 e의 오디오 방향이 자기 텍스트 방향보다 "다른" 이펙트의 텍스트 방향과
    # 더 잘 맞는 경우가 있는지 모든 (e, other) 쌍에 대해 자동 확인한다. 3차의 highshelf
    # (자기 정렬 0.031 vs distortion 텍스트 방향 0.138)가 여기 해당한다.
    reversal_check = {}
    for e in EFFECTS:
        best_other, best_cross_val = None, -np.inf
        for other in EFFECTS:
            if other == e:
                continue
            v = cross_align[f"{e}_vs_{other}"]
            if v > best_cross_val:
                best_other, best_cross_val = other, v
        if best_cross_val > self_align[e]:
            diff_dist = cross_boot_dist[f"{e}_vs_{best_other}"] - self_boot_dist[e]
            lo, hi = np.percentile(diff_dist, CI_PERCENTILES)
            reversal_check[e] = {
                "self_align": self_align[e],
                "best_cross_effect": best_other,
                "best_cross_align": best_cross_val,
                "cross_minus_self_ci_lo": float(lo),
                "cross_minus_self_ci_hi": float(hi),
                "reversal_significant": bool(lo > 0),
            }
        else:
            reversal_check[e] = {"self_align": self_align[e], "reversal_detected": False}

    print("그림 저장 중...")
    plot_text_alignment(self_align_ci, random_control_ci, cross_align_ci, out_dir / "text_alignment.png")

    results_path = out_dir / "results.json"
    results_json = {}
    if results_path.exists():
        with open(results_path) as f:
            results_json = json.load(f)

    random_control_mean = float(np.mean(list(random_control_align.values())))
    results_json["text_alignment"] = {
        "representative_param": REPRESENTATIVE_PARAM,
        "highshelf_note": "highshelf에 대응하는 정확한 캡션 관용구가 없어 'bright/crisp' 계열로 근사함 — 불완전한 근사임에 유의",
        "n_bootstrap": N_BOOTSTRAP,
        "ci_method": "텍스트 쌍(pair)을 그룹 내에서 복원추출로 재표집 → 평균 방향 재계산 → 코사인 재계산, 95% 백분위수 CI",
        "cos_by_effect": self_align,
        "cos_by_effect_ci": self_align_ci,
        "cos_random_control_by_effect": random_control_align,
        "cos_random_control_by_effect_ci": random_control_ci,
        "cos_random_control_mean": random_control_mean,
        "cos_random_control": random_control_mean,
        "gap_self_minus_control_ci": gap_ci,
        "cos_cross_effect": cross_align,
        "cos_cross_effect_ci": cross_align_ci,
        "reversal_check": reversal_check,
    }
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"완료: {results_path}, {out_dir}/text_alignment.png")


if __name__ == "__main__":
    main()
