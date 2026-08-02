"""CLAP FX Probe — 03_mapping.py

(d) 사상 모델(residual MLP) 학습과 H1~H5 위계 사다리 비교.
02_analyze.py가 만든 out/results.json에 위계 결과를 이어 붙인다 (덮어쓰지 않음).

구조의 위계 (작업지시서 참고):
  H0  정보 없음                아무 방법으로도 못 읽음
  H1  e' = e + v               상수 벡터
  H2  e' = e + f(p)·v          방향 고정, 크기만 파라미터 의존
  H3  e' = e + Δ(p)            파라미터별 방향. 소스와는 무관
  H4  e' = W·e + b             선형 변환. 소스에 따라 다르게 움직임
  H5  e' = e + g(e, p)         비선형. MLP로 학습 (residual MLP)
  H6  정보는 있으나 학습 불가   사실상 H0

이 스크립트는 H1~H5를 같은 held-out 소스 분할, 같은 지표(cos(e', e_wet))로 비교해
어느 칸에서 구조가 잡히는지 관찰한다. 결과 해석은 README에서 한다 — 코드가 단정하지 않는다.
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
from sklearn.linear_model import LinearRegression, Ridge

# 한글 라벨이 깨지지 않도록 시스템에 있는 한글 지원 폰트를 우선 사용한다.
# (02_analyze.py와 동일 설정 — 스크립트를 독립적으로 실행할 수 있도록 중복해 둔다.)
_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

EFFECTS = ["reverb", "distortion", "highshelf"]

COLORS = {
    "reverb": "#2a78d6",
    "distortion": "#eb6834",
    "highshelf": "#1baf7a",
    "baseline": "#898781",
}
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
# H1~H5 사다리는 순서가 있는 값이므로 dataviz 참조 팔레트의 순차(blue) 램프를 쓴다
# (밝음→어두움 = H1→H5). identity/셔플 통제는 무채색으로 구분.
H_LADDER_COLORS = {
    "H1": "#86b6ef",
    "H2": "#5598e7",
    "H3": "#2a78d6",
    "H4": "#1c5cab",
    "H5": "#104281",
}

MIN_TRAIN_ROWS_PER_LEVEL = 5  # 이보다 적으면 레벨별 Ridge 대신 풀링 모델로 대체


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
    return {
        "embeddings": data["embeddings"],
        "src_id": data["src_id"],
        "instrument": data["instrument"],
        "pitch": data["pitch"],
        "effect": data["effect"],
        "level_idx": data["level_idx"],
        "param_value": data["param_value"],
    }


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """행 단위 코사인 유사도."""
    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return (a_n * b_n).sum(axis=1)


def split_sources(unique_src_ids: np.ndarray, seed: int, test_size: float = 0.3):
    """소스 단위로 train/test를 나눈다 (전체 파이프라인에서 공통으로 쓰는 단일 분할).

    H1~H5 전부 같은 분할을 써야 공정하게 비교할 수 있다.
    """
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_src_ids)
    n_test = max(1, int(round(len(shuffled) * test_size)))
    test_srcs = set(shuffled[:n_test].tolist())
    train_srcs = set(shuffled[n_test:].tolist())
    return train_srcs, test_srcs


# ---------------------------------------------------------------------------
# H1~H4 — (d) MLP 이전 단계의 참고 베이스라인
# ---------------------------------------------------------------------------


def predict_H1(train_diffs: np.ndarray, test_dry: np.ndarray) -> np.ndarray:
    """H1: e' = e + v (소스·파라미터 무관 상수 벡터)."""
    v = train_diffs.mean(axis=0)
    return test_dry + v


def predict_H2(train_diffs, train_params, test_dry, test_params) -> np.ndarray:
    """H2: e' = e + f(p)·v (방향 고정, 크기만 파라미터에 선형 의존)."""
    v = train_diffs.mean(axis=0)
    norm = np.linalg.norm(v)
    v_hat = v / norm if norm > 1e-12 else v
    proj_train = train_diffs @ v_hat
    lr = LinearRegression().fit(train_params.reshape(-1, 1), proj_train)
    f_p = lr.predict(test_params.reshape(-1, 1))
    return test_dry + f_p[:, None] * v_hat


def predict_H3(train_diffs, train_levels, test_dry, test_levels) -> np.ndarray:
    """H3: e' = e + Δ(p) (파라미터[레벨]별 평균 방향, 소스와는 무관)."""
    lookup = {}
    for lvl in np.unique(train_levels):
        lookup[int(lvl)] = train_diffs[train_levels == lvl].mean(axis=0)
    fallback = train_diffs.mean(axis=0)
    deltas = np.stack([lookup.get(int(lvl), fallback) for lvl in test_levels])
    return test_dry + deltas


def predict_H4(train_dry, train_wet, train_levels, test_dry, test_levels) -> np.ndarray:
    """H4: e' = W·e + b (레벨별 선형 변환, 소스 임베딩에 따라 다르게 이동)."""
    preds = np.zeros_like(test_dry)
    fallback_model = Ridge(alpha=1.0).fit(train_dry, train_wet)
    for lvl in np.unique(test_levels):
        train_mask = train_levels == lvl
        test_mask = test_levels == lvl
        if train_mask.sum() < MIN_TRAIN_ROWS_PER_LEVEL:
            preds[test_mask] = fallback_model.predict(test_dry[test_mask])
            continue
        model = Ridge(alpha=1.0).fit(train_dry[train_mask], train_wet[train_mask])
        preds[test_mask] = model.predict(test_dry[test_mask])
    return preds


def run_ladder_for_effect(effect_name, d, dry_by_src, train_srcs, test_srcs):
    mask = d["effect"] == effect_name
    src_ids = d["src_id"][mask]
    dry = np.stack([dry_by_src[s] for s in src_ids])
    wet = d["embeddings"][mask]
    diffs = wet - dry
    params = d["param_value"][mask]
    levels = d["level_idx"][mask]

    train_mask = np.array([s in train_srcs for s in src_ids])
    test_mask = np.array([s in test_srcs for s in src_ids])

    dry_tr, dry_te = dry[train_mask], dry[test_mask]
    wet_te = wet[test_mask]
    diffs_tr = diffs[train_mask]
    params_tr, params_te = params[train_mask], params[test_mask]
    levels_tr, levels_te = levels[train_mask], levels[test_mask]

    pred_identity = dry_te
    pred_H1 = predict_H1(diffs_tr, dry_te)
    pred_H2 = predict_H2(diffs_tr, params_tr, dry_te, params_te)
    pred_H3 = predict_H3(diffs_tr, levels_tr, dry_te, levels_te)
    pred_H4 = predict_H4(dry_tr, wet[train_mask], levels_tr, dry_te, levels_te)

    return {
        "identity": float(cosine_rows(pred_identity, wet_te).mean()),
        "H1": float(cosine_rows(pred_H1, wet_te).mean()),
        "H2": float(cosine_rows(pred_H2, wet_te).mean()),
        "H3": float(cosine_rows(pred_H3, wet_te).mean()),
        "H4": float(cosine_rows(pred_H4, wet_te).mean()),
        "n_test_rows": int(test_mask.sum()),
        "n_train_rows": int(train_mask.sum()),
    }


# ---------------------------------------------------------------------------
# H5 — (d) residual MLP 사상 모델
# ---------------------------------------------------------------------------


class MappingMLP(nn.Module):
    """516(e_dry 512 + 이펙트 원핫 3 + 파라미터 1) → 512(Δ). GELU + LayerNorm."""

    def __init__(self, in_dim=516, hidden=1024, out_dim=512):
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


def build_mapping_dataset(d, dry_by_src):
    """dry/reverb/distortion/highshelf 행에서 (X_full, dry, wet, src_id)를 만든다."""
    non_dry = d["effect"] != "dry"
    src_ids = d["src_id"][non_dry]
    dry = np.stack([dry_by_src[s] for s in src_ids]).astype(np.float32)
    wet = d["embeddings"][non_dry].astype(np.float32)
    effect_names = d["effect"][non_dry]
    params = d["param_value"][non_dry].astype(np.float32)

    effect_onehot = np.zeros((len(effect_names), len(EFFECTS)), dtype=np.float32)
    for i, name in enumerate(EFFECTS):
        effect_onehot[:, i] = (effect_names == name).astype(np.float32)

    return {
        "dry": dry,
        "wet": wet,
        "effect_onehot": effect_onehot,
        "param": params,
        "src_id": src_ids,
        "effect_name": effect_names,
    }


def normalize_params_per_effect(params, effect_names, train_mask):
    """이펙트별로 파라미터 값을 z-score 정규화 (train 통계만 사용, 스케일 차이가 큰 세 이펙트를 MLP 입력에서 맞춰줌)."""
    normalized = np.zeros_like(params)
    stats = {}
    for name in EFFECTS:
        e_mask = effect_names == name
        train_vals = params[e_mask & train_mask]
        mean, std = float(train_vals.mean()), float(train_vals.std())
        std = std if std > 1e-8 else 1.0
        normalized[e_mask] = (params[e_mask] - mean) / std
        stats[name] = {"mean": mean, "std": std}
    return normalized, stats


def train_mapping_model(X_full, dry, wet, train_mask, test_mask, device, epochs, lr, seed, shuffle_labels=False):
    """MLP를 학습하고 held-out cos(e', e_wet)를 반환. shuffle_labels=True면 (동일 용량) 셔플 통제."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    wet_train_target = wet[train_mask].copy()
    if shuffle_labels:
        # train 내에서 e_wet 레이블만 무작위로 섞는다 (입력-정답 대응을 깨뜨림).
        # 같은 아키텍처/학습 절차를 그대로 써서 "용량이 있으면 뭐든 외운다"는 착시를 통제한다.
        perm = rng.permutation(len(wet_train_target))
        wet_train_target = wet_train_target[perm]

    X_tr = torch.tensor(X_full[train_mask], dtype=torch.float32, device=device)
    dry_tr = torch.tensor(dry[train_mask], dtype=torch.float32, device=device)
    wet_tr = torch.tensor(wet_train_target, dtype=torch.float32, device=device)

    X_te = torch.tensor(X_full[test_mask], dtype=torch.float32, device=device)
    dry_te = torch.tensor(dry[test_mask], dtype=torch.float32, device=device)
    wet_te = torch.tensor(wet[test_mask], dtype=torch.float32, device=device)

    model = MappingMLP(in_dim=X_full.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = X_tr.shape[0]
    batch_size = min(64, n_train)

    model.train()
    for _epoch in range(epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            delta = model(X_tr[idx])
            pred = dry_tr[idx] + delta
            mse = F.mse_loss(pred, wet_tr[idx])
            cos_loss = (1 - F.cosine_similarity(pred, wet_tr[idx], dim=1)).mean()
            loss = mse + cos_loss
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        delta_te = model(X_te)
        pred_te = dry_te + delta_te
        test_cos = F.cosine_similarity(pred_te, wet_te, dim=1).mean().item()

    return float(test_cos), model


def plot_mapping_cos(mapping_results, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    x = np.arange(len(EFFECTS))
    width = 0.25

    identity_vals = [mapping_results[e]["identity"] for e in EFFECTS]
    shuffle_vals = [mapping_results[e]["shuffle"] for e in EFFECTS]
    mapping_vals = [mapping_results[e]["H5"] for e in EFFECTS]

    ax.bar(x - width, identity_vals, width, label="identity (e'=e_dry)", color=COLORS["baseline"], zorder=3)
    ax.bar(x, shuffle_vals, width, label="셔플 (동일 용량 MLP)", color="#c3c2b7", zorder=3)
    ax.bar(x + width, mapping_vals, width, label="사상 모델 (H5)", color=[COLORS[e] for e in EFFECTS], zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(EFFECTS)
    ax.set_ylabel("Held-out cos(e', e_wet)")
    ax.set_ylim(0, 1.05)
    ax.set_title("사상 모델 성능 vs identity vs 셔플 기준선")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_hierarchy(ladder_results, mapping_results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150, sharey=True)
    h_levels = ["H1", "H2", "H3", "H4", "H5"]

    for ax, effect_name in zip(axes, EFFECTS):
        ladder = ladder_results[effect_name]
        h5_score = mapping_results[effect_name]["H5"]
        scores = [ladder["H1"], ladder["H2"], ladder["H3"], ladder["H4"], h5_score]

        x = np.arange(len(h_levels))
        ax.bar(x, scores, color=[H_LADDER_COLORS[h] for h in h_levels], zorder=3)
        ax.axhline(
            ladder["identity"], color=INK_SECONDARY, linestyle="--", linewidth=1.2, zorder=2, label="identity"
        )
        ax.axhline(
            mapping_results[effect_name]["shuffle"],
            color=COLORS["baseline"],
            linestyle=":",
            linewidth=1.2,
            zorder=2,
            label="셔플",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(h_levels)
        ax.set_title(effect_name)
        style_axis(ax)

    axes[0].set_ylabel("Held-out cos(e', e_wet)")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("H1~H5 위계 사다리 — 어느 칸에서 구조가 잡히는가")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="(d) 사상 모델 학습 + H1~H5 위계 사다리 비교")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--results", type=str, default="out/results.json", help="02_analyze.py가 만든 결과 파일 (이어 붙임)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    if args.device == "mps":
        import os

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    emb_path = Path(args.embeddings)
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path}가 없습니다. 먼저 01_embed.py를 실행하세요.")

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path}가 없습니다. 먼저 02_analyze.py를 실행해 (a)(b)(c) 결과와 악기 분류 "
            f"상한을 만들어야 03_mapping.py가 이어 붙일 수 있습니다."
        )
    with open(results_path) as f:
        results_json = json.load(f)

    d = load_embeddings(emb_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dry_mask = d["effect"] == "dry"
    dry_by_src = dict(zip(d["src_id"][dry_mask], d["embeddings"][dry_mask]))

    unique_srcs = np.unique(d["src_id"])
    train_srcs, test_srcs = split_sources(unique_srcs, args.seed, args.test_size)
    print(f"소스 분할: train {len(train_srcs)}개 / test {len(test_srcs)}개 (src_id 기준)")

    print("H1~H4 사다리 계산 중...")
    ladder_results = {e: run_ladder_for_effect(e, d, dry_by_src, train_srcs, test_srcs) for e in EFFECTS}

    print("(d) 사상 모델(H5) 학습 중 — 실제 레이블...")
    device = torch.device(args.device)
    dataset = build_mapping_dataset(d, dry_by_src)
    train_mask = np.array([s in train_srcs for s in dataset["src_id"]])
    test_mask = np.array([s in test_srcs for s in dataset["src_id"]])

    param_norm, param_stats = normalize_params_per_effect(dataset["param"], dataset["effect_name"], train_mask)
    X_full = np.concatenate([dataset["dry"], dataset["effect_onehot"], param_norm[:, None]], axis=1).astype(
        np.float32
    )

    real_cos, _model = train_mapping_model(
        X_full, dataset["dry"], dataset["wet"], train_mask, test_mask, device, args.epochs, args.lr, args.seed,
        shuffle_labels=False,
    )

    print("(d) 사상 모델 셔플 통제 학습 중 — 동일 용량, 레이블 셔플...")
    shuffled_cos, _ = train_mapping_model(
        X_full, dataset["dry"], dataset["wet"], train_mask, test_mask, device, args.epochs, args.lr, args.seed,
        shuffle_labels=True,
    )

    # 이펙트별 held-out 코사인 (전체 풀링 학습된 모델을 이펙트별로 슬라이스해 평가)
    mapping_results = {}
    for e in EFFECTS:
        e_test_mask = test_mask & (dataset["effect_name"] == e)
        e_identity = float(cosine_rows(dataset["dry"][e_test_mask], dataset["wet"][e_test_mask]).mean())
        # H5/셔플 점수는 전체 풀링 평가에서 이펙트별로 다시 계산 (같은 모델, 다른 슬라이스)
        with torch.no_grad():
            X_te_e = torch.tensor(X_full[e_test_mask], dtype=torch.float32, device=device)
            dry_te_e = torch.tensor(dataset["dry"][e_test_mask], dtype=torch.float32, device=device)
            wet_te_e = torch.tensor(dataset["wet"][e_test_mask], dtype=torch.float32, device=device)
            delta_e = _model(X_te_e)
            pred_e = dry_te_e + delta_e
            h5_e = F.cosine_similarity(pred_e, wet_te_e, dim=1).mean().item()
        mapping_results[e] = {
            "identity": e_identity,
            "H5": h5_e,
            "shuffle": shuffled_cos,  # 셔플 통제는 이펙트 풀링 전체로 한 번만 계산 (동일 용량 비교 목적)
            "n_test_rows": int(e_test_mask.sum()),
        }

    print("그림 저장 중...")
    plot_mapping_cos(mapping_results, out_dir / "mapping_cos.png")
    plot_hierarchy(ladder_results, mapping_results, out_dir / "hierarchy.png")

    instrument_ctrl = results_json.get("controls", {}).get("instrument_classification", {})

    results_json["mapping_model"] = {
        "architecture": "MLP 516 -> 1024 -> 1024 -> 512, GELU + LayerNorm, residual (e' = e_dry + Delta)",
        "loss": "MSE + (1 - cosine_similarity), equal weight",
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "test_size": args.test_size,
        "n_train_sources": len(train_srcs),
        "n_test_sources": len(test_srcs),
        "param_normalization_stats": param_stats,
        "held_out_cos_real_labels": real_cos,
        "held_out_cos_shuffled_labels": shuffled_cos,
    }
    results_json["hierarchy"] = {
        e: {
            "identity": ladder_results[e]["identity"],
            "H1": ladder_results[e]["H1"],
            "H2": ladder_results[e]["H2"],
            "H3": ladder_results[e]["H3"],
            "H4": ladder_results[e]["H4"],
            "H5": mapping_results[e]["H5"],
            "shuffle_control": mapping_results[e]["shuffle"],
            "n_test_rows": ladder_results[e]["n_test_rows"],
        }
        for e in EFFECTS
    }
    results_json["hierarchy"]["_note"] = (
        "H0/H6 여부는 이 표의 각 칸이 identity/shuffle_control을 이기는지로 README의 해석 "
        "기준표에 따라 판단할 것 — 이 스크립트는 수치만 산출하고 판정하지 않는다. "
        f"참고용 악기 분류 상한(공유 기준선): {instrument_ctrl.get('accuracy')}"
    )

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"완료: {results_path} (hierarchy/mapping_model 필드 추가), {out_dir}/mapping_cos.png, {out_dir}/hierarchy.png")


if __name__ == "__main__":
    main()
