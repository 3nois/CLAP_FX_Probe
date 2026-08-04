"""CLAP FX Probe — 08_quality_stratified.py (4차, ★최우선)

우리가 "dry"라고 불러온 NSynth 소스가 진짜 dry가 아닐 수 있다 — NSynth는 상용 샘플
라이브러리 산물이라 룸 리버브·EQ·컴프레션이 이미 배어 있을 수 있고, NSynth는 이를
`qualities_str` 태그로 직접 제공한다(reverb, distortion, bright, dark 등 10종).
1~3차 모두 이 태그를 읽지 않았다.

이 스크립트는 examples.json의 태그로 소스를 층화해, 이미 이펙트가 걸려 있는 소스에서
프로브 R²가 더 낮은지("포화 가설")를 직접 검정한다. 사실이면 3차까지의
"distortion ≫ reverb" 순서가 캡션이나 강도가 아니라 "리버브가 이미 걸려 있어서"일 수
있다는 뜻이다.

재추출 없이 기존 embeddings.npz(3차 산출물)만 재사용한다.

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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"reverb": "#2a78d6", "distortion": "#eb6834", "highshelf": "#1baf7a", "baseline": "#898781"}
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
MIN_GROUP_SIZE = 20  # 이보다 적으면 "결론 없음"으로 보고

# (태그, 이펙트, 대표 파라미터) — 05_text_alignment.py의 REPRESENTATIVE_PARAM과 동일 관례
STRATIFY_SPECS = [
    ("reverb", "reverb", "wet_level"),
    ("distortion", "distortion", "drive_db"),
    ("bright", "highshelf", "gain_db"),
]


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


def load_quality_tags(examples_json_path: Path):
    with open(examples_json_path) as f:
        examples = json.load(f)
    return {k: set(v["qualities_str"]) for k, v in examples.items()}


def held_out_r2_single(X, y, groups, seed, n_splits=5, test_size=0.3):
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    scores = []
    for train_idx, test_idx in gss.split(X, y, groups):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(r2_score(y[test_idx], pred))
    return float(np.mean(scores)), float(np.std(scores))


def bootstrap_r2_ci_single(X, y, groups, seed, n_boot=500, ci=0.95):
    unique_srcs = np.unique(groups)
    n = len(unique_srcs)
    src_to_rows = {s: np.where(groups == s)[0] for s in unique_srcs}
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_boot):
        boot_srcs = rng.choice(unique_srcs, size=n, replace=True)
        oob_srcs = np.setdiff1d(unique_srcs, boot_srcs)
        if len(oob_srcs) < 3:
            continue
        train_idx = np.concatenate([src_to_rows[s] for s in boot_srcs])
        test_idx = np.concatenate([src_to_rows[s] for s in oob_srcs])
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(r2_score(y[test_idx], pred))
    scores = np.array(scores)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return float(scores.mean()), float(np.percentile(scores, lo)), float(np.percentile(scores, hi))


def probe_for_source_group(effect_name, param_name, theta_slots, param_order, d, src_ids_in_group, seed, n_boot):
    mask = (d["effect"] == effect_name) & np.isin(d["src_id"], list(src_ids_in_group))
    n_sources = len(set(d["src_id"][mask].tolist()))
    if n_sources < MIN_GROUP_SIZE:
        return {
            "n_sources": n_sources,
            "probe_r2": None,
            "probe_r2_ci_low": None,
            "probe_r2_ci_high": None,
            "note": f"표본 {n_sources}개 < {MIN_GROUP_SIZE} — 결론 없음",
        }

    X = d["embeddings"][mask]
    start, end = theta_slots[effect_name]
    param_idx = param_order[effect_name].index(param_name)
    y = d["theta_norm"][mask][:, start + param_idx]
    groups = d["src_id"][mask]

    r2_mean, _ = held_out_r2_single(X, y, groups, seed)
    boot_mean, ci_lo, ci_hi = bootstrap_r2_ci_single(X, y, groups, seed, n_boot=n_boot)
    return {
        "n_sources": n_sources,
        "n_rows": int(mask.sum()),
        "probe_r2": r2_mean,
        "probe_r2_ci_low": ci_lo,
        "probe_r2_ci_high": ci_hi,
        "note": None,
    }


def plot_quality_stratified(results, out_path):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    labels, has_vals, has_err, no_vals, no_err, colors, n_labels = [], [], [], [], [], [], []

    for tag, effect_name, param_name in STRATIFY_SPECS:
        key = f"{tag}.{effect_name}.{param_name}"
        r = results[key]
        labels.append(f"{param_name}\n({tag} 태그)")
        colors.append(COLORS[effect_name])

        has_r2 = r["has_tag"]["probe_r2"]
        has_vals.append(max(has_r2, 0.0) if has_r2 is not None else 0.0)
        if has_r2 is not None:
            has_err.append([has_r2 - r["has_tag"]["probe_r2_ci_low"], r["has_tag"]["probe_r2_ci_high"] - has_r2])
        else:
            has_err.append([0, 0])

        no_r2 = r["no_tag"]["probe_r2"]
        no_vals.append(max(no_r2, 0.0) if no_r2 is not None else 0.0)
        if no_r2 is not None:
            no_err.append([no_r2 - r["no_tag"]["probe_r2_ci_low"], r["no_tag"]["probe_r2_ci_high"] - no_r2])
        else:
            no_err.append([0, 0])

        n_labels.append(f"n={r['has_tag']['n_sources']}/{r['no_tag']['n_sources']}")

    x = np.arange(len(labels))
    width = 0.35
    has_err_arr = np.clip(np.array(has_err).T, 0, None)
    no_err_arr = np.clip(np.array(no_err).T, 0, None)

    ax.bar(x - width / 2, has_vals, width, yerr=has_err_arr, capsize=3, color=colors, alpha=0.9, label="태그 있음 (이미 걸려 있음)", zorder=3)
    ax.bar(x + width / 2, no_vals, width, yerr=no_err_arr, capsize=3, color=colors, alpha=0.45, label="태그 없음 (더 깨끗함)", zorder=3)

    for xi, nlabel in zip(x, n_labels):
        ax.text(xi, -0.03, nlabel, ha="center", va="top", fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Held-out R² (부트스트랩 95% CI)")
    ax.set_title("NSynth 품질 태그 층화 — 포화 가설 검정\n(태그 없음 쪽이 유의하게 높으면 포화 가설 지지)")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(0, color=GRID_COLOR, linewidth=1)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="NSynth quality 태그 층화 프로브 (4차, 최우선)")
    parser.add_argument("--embeddings", type=str, default="out/embeddings.npz")
    parser.add_argument("--examples-json", type=str, default="nsynth-test/examples.json")
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_embeddings(emb_path)
    config = load_config(emb_path.parent)
    theta_slots = {e: tuple(v) for e, v in config["theta_slots"].items()}
    param_order = config["param_order"]

    print("NSynth examples.json에서 quality 태그 로딩 중...")
    tags_by_filename = load_quality_tags(Path(args.examples_json))

    dry_mask = d["effect"] == "dry"
    src_ids = d["src_id"][dry_mask]
    filenames = d["filename"][dry_mask]

    tags_by_src = {}
    n_missing = 0
    for src_id, fname in zip(src_ids.tolist(), filenames.tolist()):
        note_str = fname[:-4] if fname.endswith(".wav") else fname
        if note_str in tags_by_filename:
            tags_by_src[src_id] = tags_by_filename[note_str]
        else:
            tags_by_src[src_id] = set()
            n_missing += 1

    n_sources = len(tags_by_src)
    print(f"소스 {n_sources}개 중 examples.json 매칭 실패 {n_missing}개")

    print("소스 800개의 태그 분포 산출 중...")
    all_tags = ["bright", "dark", "distortion", "fast_decay", "long_release",
                "multiphonic", "nonlinear_env", "percussive", "reverb", "tempo-synced"]
    tag_distribution = {}
    for tag in all_tags:
        count = sum(1 for tags in tags_by_src.values() if tag in tags)
        tag_distribution[tag] = {"count": count, "fraction": count / n_sources}
        print(f"  {tag}: {count}/{n_sources} ({count/n_sources*100:.1f}%)")

    print("태그별 층화 프로브 계산 중...")
    results = {}
    for tag, effect_name, param_name in STRATIFY_SPECS:
        has_srcs = {s for s, tags in tags_by_src.items() if tag in tags}
        no_srcs = {s for s, tags in tags_by_src.items() if tag not in tags}

        has_result = probe_for_source_group(effect_name, param_name, theta_slots, param_order, d, has_srcs, args.seed, args.n_boot)
        no_result = probe_for_source_group(effect_name, param_name, theta_slots, param_order, d, no_srcs, args.seed, args.n_boot)

        key = f"{tag}.{effect_name}.{param_name}"
        results[key] = {
            "tag": tag,
            "effect": effect_name,
            "param": param_name,
            "has_tag": has_result,
            "no_tag": no_result,
        }
        if has_result["probe_r2"] is not None and no_result["probe_r2"] is not None:
            results[key]["gap_no_minus_has"] = no_result["probe_r2"] - has_result["probe_r2"]
        print(f"  {key}: 있음(n={has_result['n_sources']}) R²={has_result['probe_r2']}, "
              f"없음(n={no_result['n_sources']}) R²={no_result['probe_r2']}")

    print("그림 저장 중...")
    plot_quality_stratified(results, out_dir / "quality_stratified.png")

    results_path = out_dir / "results.json"
    results_json = {}
    if results_path.exists():
        with open(results_path) as f:
            results_json = json.load(f)

    results_json["quality_stratification"] = {
        "n_sources": n_sources,
        "n_examples_json_match_missing": n_missing,
        "tag_distribution": tag_distribution,
        "stratified_probes": results,
    }
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"완료: {results_path}, {out_dir}/quality_stratified.png")


if __name__ == "__main__":
    main()
