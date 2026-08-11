"""9차 Phase 3-3R-5 — 블라인드 청취 응답 채점.

out/audio/blind/abx_test.html에서 저장한 blind_responses.json과 answer_key.json을
대조해 정답률·이항검정·층별 정답률·지표 상관·자연스러움 분포·청취횟수-정답률
관계를 낸다. 사용자가 실제로 청취를 마치고 blind_responses.json을 내려받은 뒤에만
실행할 수 있다(청취 없이 채점 불가 — 데이터가 없다).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, pointbiserialr


def main():
    parser = argparse.ArgumentParser(description="9차 Phase 3-3R-5 — 블라인드 응답 채점")
    parser.add_argument("--responses", type=str, default="out/audio/blind/blind_responses.json")
    parser.add_argument("--answer-key", type=str, default="out/audio/blind/answer_key.json")
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    resp_path = Path(args.responses)
    if not resp_path.exists():
        raise FileNotFoundError(
            f"{resp_path}가 없습니다. out/audio/blind/abx_test.html을 로컬 서버로 열어 "
            f"24쌍을 청취/응답하고 '응답 저장' 버튼으로 내려받은 뒤 이 경로에 둘 것."
        )

    with open(resp_path) as f:
        responses = json.load(f)["responses"]
    with open(args.answer_key) as f:
        answer_key = {a["id"]: a for a in json.load(f)["pairs"]}

    rows = []
    for r in responses:
        pid = r["pair_id"]
        if pid not in answer_key:
            continue
        ans = answer_key[pid]
        if r["choice"] is None:
            continue
        correct = (r["choice"] == ans["correct"])
        rows.append({
            "pair_id": pid, "correct": correct, "choice": r["choice"], "answer": ans["correct"],
            "naturalness": r["naturalness"], "listen_count": r["listen_count"],
            "family": ans["family"], "effect": ans["effect"], "stratum": ans["stratum"],
            "directional_agreement": ans["directional_agreement"],
        })

    n = len(rows)
    n_correct = sum(r["correct"] for r in rows)
    acc = n_correct / n if n else None
    bt = binomtest(n_correct, n, 0.5, alternative="greater")

    by_stratum = {}
    for stratum in ["high", "mid", "low"]:
        sub = [r for r in rows if r["stratum"] == stratum]
        if sub:
            by_stratum[stratum] = {"n": len(sub), "n_correct": sum(r["correct"] for r in sub),
                                    "accuracy": sum(r["correct"] for r in sub) / len(sub)}

    da = np.array([r["directional_agreement"] for r in rows])
    correct_arr = np.array([1.0 if r["correct"] else 0.0 for r in rows])
    corr, corr_p = (pointbiserialr(correct_arr, da) if len(set(correct_arr.tolist())) > 1 else (None, None))

    nat_all = {}
    for n_level in ["상", "중", "하"]:
        nat_all[n_level] = sum(1 for r in rows if r["naturalness"] == n_level)

    nat_by_family = {}
    for r in rows:
        nat_by_family.setdefault(r["family"], {"상": 0, "중": 0, "하": 0})
        nat_by_family[r["family"]][r["naturalness"]] += 1

    listen_counts = np.array([r["listen_count"] for r in rows])
    median_lc = float(np.median(listen_counts))
    low_lc = [r["correct"] for r in rows if r["listen_count"] <= median_lc]
    high_lc = [r["correct"] for r in rows if r["listen_count"] > median_lc]
    listen_vs_acc = {
        "median_listen_count": median_lc,
        "accuracy_low_listen_count": (sum(low_lc) / len(low_lc)) if low_lc else None,
        "accuracy_high_listen_count": (sum(high_lc) / len(high_lc)) if high_lc else None,
        "note": "listen_count가 높은(많이 들어야 했던) 쌍에서 정답률이 낮으면 지표가 '듣기 어려운 정도'까지 예측한다는 뜻",
    }

    verdict_significance = "유의함 (p<0.05)" if bt.pvalue < 0.05 else "유의하지 않음"
    stratum_monotonic = (
        by_stratum.get("high", {}).get("accuracy", 0) >= by_stratum.get("mid", {}).get("accuracy", 0) >= by_stratum.get("low", {}).get("accuracy", 0)
        if all(s in by_stratum for s in ["high", "mid", "low"]) else None
    )
    verdict_metric_validated = (
        "층별 정답률이 지표(high>mid>low)와 같은 방향으로 움직임 — 지표가 검증됨. 향후 청취 없이 지표만으로 대규모 평가 가능"
        if stratum_monotonic else
        "층별 정답률이 지표 순서와 어긋남 — 지표가 '들리는 정도'를 완전히 반영하지 못함"
    )

    results = {
        "meta": {"n_pairs_answered": n, "responses_path": str(resp_path)},
        "depends_on_surrogate": "none",
        "overall": {"n": n, "n_correct": n_correct, "accuracy": acc,
                    "binomial_test_p": bt.pvalue, "significance": verdict_significance,
                    "threshold_for_p05_at_n24": "17/24 이상"},
        "by_stratum": by_stratum,
        "stratum_accuracy_monotonic_with_metric": stratum_monotonic,
        "metric_validation_verdict": verdict_metric_validated,
        "directional_agreement_vs_correctness_pointbiserial": {"r": corr, "p": corr_p},
        "naturalness_distribution_overall": nat_all,
        "naturalness_distribution_by_family": nat_by_family,
        "listen_count_vs_accuracy": listen_vs_acc,
        "rows": rows,
    }

    out_dir = Path(args.out)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results_9_blind.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=== 블라인드 청취 채점 결과 ===")
    print(f"전체 정답률: {n_correct}/{n} = {acc:.3f}  이항검정 p={bt.pvalue:.4f} ({verdict_significance})")
    print("\n층별 정답률:")
    for s, v in by_stratum.items():
        print(f"  {s:<6} {v['n_correct']}/{v['n']} = {v['accuracy']:.3f}")
    print(f"\n지표-정답 상관(point-biserial): r={corr}  p={corr_p}")
    print(f"판정: {verdict_metric_validated}")
    print(f"\n자연스러움 분포(전체): {nat_all}")
    print(f"\n청취횟수-정답률: {listen_vs_acc}")
    print(f"\n저장: {results_dir/'results_9_blind.json'}")


if __name__ == "__main__":
    main()
