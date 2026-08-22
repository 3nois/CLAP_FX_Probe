# -*- coding: utf-8 -*-
"""결함 21 재검증 — 고정방향 베이스라인(raw 평균 후 정규화 vs 단위벡터 평균)을 나란히 비교.

사용자 판정: "고정 방향 하나로 전 소스를 처리할 때 평균 코사인을 최대화하는 방향은
단위벡터의 평균이지 raw 벡터의 평균이 아니다. raw 평균은 열등한 추정량이고, 그
편향은 B2가 베이스라인을 이기는 폭을 실제보다 커 보이게 만드는 방향이다."

대상(사용자 지시):
  21_handle_predict_phase1.py  predict_global_mean / predict_family_mean_oracle
  20_family_cosine_oat.py      split_half_correction (과제 C)

이 스크립트는 위 두 파일에 이미 추가된 단위벡터-평균 버전
(predict_global_mean_unitavg / predict_family_mean_oracle_unitavg /
split_half_correction_unitavg)과 원래 버전을, MLP 재학습 없이(베이스라인은
닫힌형이라 필요 없음) 실제 사용처와 동일한 데이터 구성으로 나란히 비교한다.

  1. 21의 OAT 기반 A/B1/B2 (reverb/distortion/highshelf, oat_emb.npz)
  2. q3q4의 25레벨 5축 x 7구간 forward/B1/B2 (11_phase5_q3q4.py와 동일 구성)
  3. 20의 과제 C(family 평균 코사인, split-half 보정)

차이가 0.01 미만이면 "실질 영향 없음"으로 기록하고 단위평균 버전을 채택,
0.01 이상이면 개별 표에 표시하고 Q4의 between 기준선(20_family_cosine_oat의
bootstrap_within_between 경로 — 이미 unit() 사용, 이 결함과 무관 확인됨)도
같이 재검토 대상으로 표시한다.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

dr = import_module("11_phase2_doseresponse")
fam_mod = import_module("20_family_cosine_oat")
pred_mod = import_module("21_handle_predict_phase1")
q3q4 = import_module("11_phase5_q3q4")

RESULTS_DIR = dr.RESULTS_DIR
SEED = 0
DIFF_THRESHOLD = 0.01


def baseline_pair_compare(Y_train, X_eval_n, family_train, family_eval, Yt_dir_test, src_test):
    d_raw, _ = pred_mod.predict_global_mean(Y_train, X_eval_n)
    d_ua, _ = pred_mod.predict_global_mean_unitavg(Y_train, X_eval_n)
    g_raw = pred_mod.bootstrap_cos_ci(np.sum(d_raw * Yt_dir_test, axis=-1), src_test, SEED)[0]
    g_ua = pred_mod.bootstrap_cos_ci(np.sum(d_ua * Yt_dir_test, axis=-1), src_test, SEED)[0]

    d_raw2, _ = pred_mod.predict_family_mean_oracle(Y_train, family_train, family_eval)
    d_ua2, _ = pred_mod.predict_family_mean_oracle_unitavg(Y_train, family_train, family_eval)
    f_raw = pred_mod.bootstrap_cos_ci(np.sum(d_raw2 * Yt_dir_test, axis=-1), src_test, SEED)[0]
    f_ua = pred_mod.bootstrap_cos_ci(np.sum(d_ua2 * Yt_dir_test, axis=-1), src_test, SEED)[0]
    return g_raw, g_ua, f_raw, f_ua


def main():
    lines = ["# 결함 21 재검증 — 고정방향 베이스라인 raw-평균 vs 단위벡터-평균 (2026-08-22)\n"]
    lines.append(f"판정 기준: |raw 평균 cos − 단위평균 cos| < {DIFF_THRESHOLD} 이면 실질 영향 없음(단위평균 채택), "
                 f"아니면 개별 표시 + between 기준선 재검토 대상.\n")

    # ---- 1. 21 OAT 기반 A/B1/B2 ----
    lines.append("## 1. 21_handle_predict_phase1 — OAT 기반 A/B1/B2 (reverb/distortion/highshelf)\n")
    lines.append("| 이펙트 | 조건 | global_mean(raw) | global_mean(단위평균) | 차이 | family_oracle(raw) | family_oracle(단위평균) | 차이 | 실질영향 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    d = np.load("out/caches/oat_emb.npz", allow_pickle=False)
    emb_oat, family_oat = d["emb"], d["instrument_family"]
    n_sources = emb_oat.shape[0]
    train_idx, val_idx, test_idx, _ = pred_mod.stratified_split(family_oat, SEED)
    EFFECT_NAMES = ["reverb", "distortion", "highshelf"]
    max_diff_21 = 0.0
    for ei, effect in enumerate(EFFECT_NAMES):
        e0, e2 = emb_oat[:, ei, 0], emb_oat[:, ei, 2]
        conditions = {"A(forward)": (e0, e2 - e0), "B1(reverse,known)": (e2, e0 - e2)}
        for cond_name, (X, Y) in conditions.items():
            Xtr, Ytr, Ftr = X[train_idx], Y[train_idx], family_oat[train_idx]
            Xte, Yte, Fte = X[test_idx], Y[test_idx], family_oat[test_idx]
            Yt_dir_test = pred_mod.unit_np(Yte)
            g_raw, g_ua, f_raw, f_ua = baseline_pair_compare(Ytr, len(Xte), Ftr, Fte, Yt_dir_test, test_idx)
            dg, df = abs(g_raw - g_ua), abs(f_raw - f_ua)
            max_diff_21 = max(max_diff_21, dg, df)
            ok = "실질영향없음" if max(dg, df) < DIFF_THRESHOLD else "**차이 유의 — 확인要**"
            lines.append(f"| {effect} | {cond_name} | {g_raw:.4f} | {g_ua:.4f} | {dg:+.4f} | "
                         f"{f_raw:.4f} | {f_ua:.4f} | {df:+.4f} | {ok} |")
            print(f"21/{effect}/{cond_name}: global diff={dg:+.4f} family diff={df:+.4f}")

    # ---- 2. q3q4 25레벨 5축 x 7구간 ----
    lines.append("\n## 2. 11_phase5_q3q4 — 25레벨 5축 x 7구간 forward/B1/B2\n")
    lines.append("| 축 | 구간 | 조건 | global(raw) | global(단위평균) | 차이 | family(raw) | family(단위평균) | 차이 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    family_1200 = q3q4.get_family_array_1200()
    max_diff_q3q4 = 0.0
    for axis_name in q3q4.AXES:
        emb, theta_raw, src_id = dr.load_concat(axis_name)
        order = np.argsort(src_id)
        emb, src_id = emb[order], src_id[order]
        train_idx, val_idx, test_idx, _ = pred_mod.stratified_split(family_1200, SEED)
        for label, idx_a, idx_b in q3q4.INTERVALS:
            e_a, e_b = emb[:, idx_a, :], emb[:, idx_b, :]
            conds = {"forward": (e_a, e_b - e_a), "B1": (e_b, e_a - e_b)}
            if (idx_b - idx_a) != 1:
                pool_idx = list(range(idx_a + 1, idx_b + 1))
                X_parts, Y_parts, pos_parts = [], [], []
                for pi in pool_idx:
                    X_parts.append(emb[:, pi, :]); Y_parts.append(e_a - emb[:, pi, :]); pos_parts.append(np.arange(n_sources))
                X_pool = np.concatenate(X_parts, axis=0); Y_pool = np.concatenate(Y_parts, axis=0)
                pos_pool = np.concatenate(pos_parts, axis=0)
                conds["B2"] = None  # 별도 처리(풀링된 인덱스 매핑 필요)
            else:
                pool_idx = None
            for cond_name, xy in conds.items():
                if xy is None:
                    mask_tr = np.isin(pos_pool, train_idx)
                    mask_te = np.isin(pos_pool, test_idx)
                    Xtr, Ytr, Ftr = X_pool[mask_tr], Y_pool[mask_tr], family_1200[pos_pool[mask_tr]]
                    Xte, Yte, Fte = X_pool[mask_te], Y_pool[mask_te], family_1200[pos_pool[mask_te]]
                    src_te = src_id[pos_pool[mask_te]]
                else:
                    X, Y = xy
                    Xtr, Ytr, Ftr = X[train_idx], Y[train_idx], family_1200[train_idx]
                    Xte, Yte, Fte = X[test_idx], Y[test_idx], family_1200[test_idx]
                    src_te = src_id[test_idx]
                Yt_dir_test = pred_mod.unit_np(Yte)
                g_raw, g_ua, f_raw, f_ua = baseline_pair_compare(Ytr, len(Xte), Ftr, Fte, Yt_dir_test, src_te)
                dg, df = abs(g_raw - g_ua), abs(f_raw - f_ua)
                max_diff_q3q4 = max(max_diff_q3q4, dg, df)
                lines.append(f"| {axis_name} | {label} | {cond_name} | {g_raw:.4f} | {g_ua:.4f} | {dg:+.4f} | "
                             f"{f_raw:.4f} | {f_ua:.4f} | {df:+.4f} |")
        print(f"q3q4/{axis_name} 완료 (max diff so far: {max_diff_q3q4:.4f})")

    # ---- 3. 20 과제 C ----
    lines.append("\n## 3. 20_family_cosine_oat — 과제 C (family 평균 코사인, split-half 보정)\n")
    lines.append("| 이펙트 | 패밀리쌍 | cross_cosine(raw) | cross_cosine(단위평균) | 차이 |")
    lines.append("|---|---|---|---|---|")
    ZERO_NORM_EPS = fam_mod.ZERO_NORM_EPS
    max_diff_20 = 0.0
    families_sorted = sorted(set(family_oat.tolist()))
    for ei, effect in enumerate(EFFECT_NAMES):
        e0, e2 = emb_oat[:, ei, 0], emb_oat[:, ei, 2]
        v = e2 - e0
        norms = np.linalg.norm(v, axis=1)
        keep = norms > ZERO_NORM_EPS
        v_kept, fam_kept = v[keep], family_oat[keep]
        v_by_family = {f: v_kept[fam_kept == f] for f in families_sorted if (fam_kept == f).sum() >= 2}
        cross_raw, _, _, fams_used = fam_mod.split_half_correction(v_by_family, 100, SEED)
        cross_ua, _, _, _ = fam_mod.split_half_correction_unitavg(v_by_family, 100, SEED)
        for fi, fj in cross_raw:
            diff = abs(cross_raw[(fi, fj)] - cross_ua[(fi, fj)])
            max_diff_20 = max(max_diff_20, diff)
            lines.append(f"| {effect} | {fi}\\|{fj} | {cross_raw[(fi,fj)]:.4f} | {cross_ua[(fi,fj)]:.4f} | {diff:+.4f} |")
        print(f"20/{effect} 완료 (max diff so far: {max_diff_20:.4f})")

    # ---- 결론 ----
    overall_max = max(max_diff_21, max_diff_q3q4, max_diff_20)
    lines.append(f"\n## 결론\n")
    lines.append(f"세 영역 전체 최대 차이 = {overall_max:.4f} "
                 f"(21/OAT={max_diff_21:.4f}, q3q4={max_diff_q3q4:.4f}, 20/과제C={max_diff_20:.4f}).\n")
    if overall_max < DIFF_THRESHOLD:
        lines.append(f"모든 항목이 임계값({DIFF_THRESHOLD}) 미만 — **실질 영향 없음**. 단위벡터-평균 버전"
                     f"(`predict_global_mean_unitavg`/`predict_family_mean_oracle_unitavg`/"
                     f"`split_half_correction_unitavg`)을 정식 채택한다. 기존 raw-평균 버전은 코드에는 "
                     f"남기되(하위호환·대조용) 향후 보고서는 단위평균 버전을 우선 인용한다.\n")
    else:
        lines.append(f"임계값({DIFF_THRESHOLD}) 이상인 항목이 있다 — 위 표에서 해당 행을 확인하고, "
                     f"영향을 받는 축/구간에 대해서는 Q4의 between 기준선(값 자체는 이미 `unit(v)` 기반이라 "
                     f"이 결함과 무관하지만, B2가 그 기준선을 넘는지 여부의 해석은) 재검토가 필요하다.\n")
    lines.append("**참고**: Q3/Q4 표의 'between 기준선' 열 자체는 `20_family_cosine_oat.bootstrap_within_between`이 "
                 "`unit(v)`로 이미 정규화한 벡터의 소스쌍별 코사인을 평균한 값이다 — raw-평균 문제와 무관한 "
                 "경로이므로 재계산이 필요 없음을 확인했다.\n")

    lines.append("## 결함 21\n")
    lines.append("> 고정방향 베이스라인(전역평균/패밀리평균오라클/family 평균 코사인)을 raw(비정규화) 벡터를 "
                 "먼저 소스 간 평균한 뒤 정규화해서 구했다. 목표 지표(소스별 코사인의 평균)를 최대화하는 "
                 "상수 방향은 단위벡터의 평균이지 raw 평균이 아니므로, 이 방식은 최적보다 열등한(낮은 cos) "
                 "베이스라인을 만든다 — 편향의 방향이 B2(학습 모델)가 베이스라인을 이기는 폭을 과대평가하게 "
                 "만드는 쪽이었다. 단위벡터-평균 버전을 추가해 나란히 비교했다(위 결론 참고).\n")

    out_path = RESULTS_DIR / "11_phase5_defect21_baseline_recheck.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n저장: {out_path}")
    print(f"전체 최대 차이: {overall_max:.4f}")


if __name__ == "__main__":
    main()
