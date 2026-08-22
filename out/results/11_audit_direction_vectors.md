# 레포 전체 감사 — 임베딩 차분 기반 방향·각도 계산 전수 조사 (2026-08-22)

결함 18(신호가 약한 지점에서 `normalize(a)-normalize(b)`가 잡음에 압도됨)과
결함 19(수정 범위를 부호 분기로 좁게 오진)를 계기로, 임베딩 차분으로
방향·각도를 계산하는 모든 코드를 전수 조사했다. 확인 항목 3가지:

1. `normalize(a) − normalize(b)` (끝점 각각 정규화 후 차분, **결함 18 패턴**) vs
   `normalize(a − b)` (차분 후 정규화, 올바른 패턴)
2. 소스별로 정규화하기 **전에** 소스 평균을 내는가 — 그러면 `‖v‖`가 큰(효과가 강한)
   소스가 평균 방향을 지배한다
3. 각도를 낸 뒤 평균인가, 평균을 낸 뒤 각도인가 — 후자는 반대 방향 벡터끼리
   상쇄된다(결함 17과 같은 종류)

"없음을 확인했다"가 아니라 각 함수의 실제 정의를 표에 적어 대조 가능하게 했다.

## 종합 표

| 스크립트 | 함수 | 벡터 정의 | 소스별 정규화 시점 | 평균 시점 | 판정 |
|---|---|---|---|---|---|
| `11_phase3_rotation.py` (v1) | `get_v_A` | `normalize(e_max) − normalize(e_min)` | 차분 **전**(끝점 각각) | 각도 산출 후 평균(bootstrap) | **결함 18 (미수정, 원본)** |
| `11_phase3_rotation_v2.py` | `run_pair_direction` | `normalize(e_max) − normalize(e_min)` | 차분 **전** | 각도 후 평균 | **결함 18 (미수정 — v5가 대체)** |
| `11_phase3_rotation_v3.py` | `run_signed_branch` | `normalize(e_max) − normalize(e_min)` | 차분 **전** | 각도 후 평균 | **결함 18 (미수정 — v4가 대체)** |
| `11_phase3_rotation_v4.py` | `run_signed_branch_v4` | `normalize(e_max − e_min)` | 차분 **후** | 각도 후 평균 | 수정됨(정상) |
| `11_phase3_rotation_v5.py` | `run_pair_direction_v5` | `normalize(e_max − e_min)` | 차분 **후** | 각도 후 평균 | 수정됨(정상) |
| `11_phase2_doseresponse.py` | `displacement_curve`, `jnd_curve`, `curvature_summary` | 방향벡터 차분 없음 — 스칼라 코사인(`cos_rows`)·raw 크기(`‖a−b‖`)만 계산 | 해당 없음 | 스칼라 산출 후 평균 | 해당없음(방향벡터 미사용, 문제없음) |
| `11_phase2_bend_signedjnd.py` | `bend_curve` | `deltas = e[i+1]−e[i]` (raw), `cos_vec(delta_i, delta_{i+1})`(비율식이라 암묵적으로 차분 후 정규화와 동치) | 차분 후(암묵) | 소스별 각도 산출 → `np.mean(ang)` | 정상 |
| `11_phase5_context.py` | `run_pair` | `Y = e_max − e_min` (raw) → `pred_mod.unit_np(Y)` | 차분 후 | 행별 cos → `bootstrap_cos_ci`(소스 단위) | 정상 |
| `11_phase5_curvature_rotation.py` | `main` | 자체 벡터 계산 없음 — `11_phase3_rotation_raw.json`(**v1**, 결함18 미수정 원본)을 그대로 읽음 | — | — | **하류 오염(신규, 결함 20 후보)** — 자체 버그는 없으나 결함18에 오염된 v1 수치를 v2~v5 수정 이후에도 계속 참조 |
| `11_phase5_q3q4.py` | `run_q3` | `v = emb[idx_b] − emb[idx_a]` (raw) → `fam_mod.unit(v)` | 차분 후 | `bootstrap_within_between`이 소스쌍 코사인 산출 후 평균 | 정상 |
| `11_phase5_q3q4.py` | `run_q4` | `Y = e_b − e_a` (raw), `pred_mod.run_all_models`에 위임 | 차분 후(위임처 확인) | 위임처(21번 파일)에서 행별 cos 후 평균 | 정상 |
| `20_family_cosine_oat.py` | 과제A/B (`v_full`, `unit`, `cosine_matrix`, `bootstrap_within_between`) | `v = e2 − e0` (raw) → `unit(v)` | 차분 후 | 소스쌍 코사인 산출 후 평균(부트스트랩) | 정상 |
| `20_family_cosine_oat.py` | `split_half_correction` | `full_mean = v_by_family[f].mean(axis=0)` (**raw, 정규화 전** 패밀리 내 소스 평균) 후 `dot/norm` | **평균이 정규화보다 먼저**(패밀리 내) | 패밀리 평균 벡터끼리 코사인 | **요주의(항목 2 해당, 결함 아님/미확정)** — docstring이 "raw(비정규화)"라 명시해 의도적으로 보이나, 효과가 큰 소스가 family 방향 추정을 지배할 수 있음. 과제 B(★주 검정, 결과 리포트 §Q2 근거)는 이 함수를 쓰지 않으므로 영향 없음 — 과제 C(family 평균 코사인, 3차 비교용 보조 지표)에만 해당 |
| `20_family_cosine_oat.py` | 하프스윙 `cos_pos_neg` | `v_pos=e2−e1`, `v_neg=e1−e0` (raw) → 직접 `dot/norm` | 차분 후(암묵) | 소스별 cos 산출 후 `np.mean` | 정상 |
| `21_handle_predict_phase1.py` | `train_dual_head`/`dual_head_loss` | `Y` raw → `F.normalize(Y, dim=-1)` | 차분 후 | 행별 cos → `.mean()`(배치) | 정상 |
| `21_handle_predict_phase1.py` | `run_all_models`(linear/mlp 본선 모델) | 모델이 직접 단위벡터 출력, `Yt_dir_test = unit_np(Y_test)` | 차분 후 | 행별 cos → `bootstrap_cos_ci`(소스 단위) | 정상 |
| `21_handle_predict_phase1.py` | `predict_global_mean` | `v_const = Y_train.mean(axis=0)` (**raw 평균**) → `unit_np` | **평균이 정규화보다 먼저**(전체 소스) | — | **요주의(항목 2 해당)** — 기준선(①) 한정. Q3/Q4 핵심 결론(B2 vs "between" 기준선)은 `run_q3`(정상 경로)를 쓰므로 미영향, 단 raw json의 `global_mean`/`family_mean_oracle` 열 자체는 이 편향의 영향을 받을 수 있음 |
| `21_handle_predict_phase1.py` | `predict_family_mean_oracle` | `Y_train[mask].mean(axis=0)` (**raw 평균**, 패밀리 내) → `unit_np` | **평균이 정규화보다 먼저**(패밀리 내) | — | **요주의(항목 2 해당)** — 위와 동일 사유(②만 해당, 핵심 결론 미영향) |
| `22_handle_predict_phase2.py` | B1/B2/B3/복원(C)/LOFO(D) 전체 | `Y = e_target − e_source` (raw) → `unit_np` | 차분 후 | 행별 cos → `bootstrap_mean_ci`(소스 단위) | 정상 |
| `tokensynth_bridge/phase_f4_full.py` | `main`(directional_agreement) | `v_generated=e_regen_d−e_regen_c`, `v_original=e_dry_true−e_wet` (raw) → `cos_np`(비율식, 암묵적 차분 후 정규화) | 차분 후(암묵) | (source,effect,variant)별 cos 산출 → MIDI 3변형 평균 → 소스 부트스트랩 평균 | 정상 |

## 요약

- **결함 18 패턴(항목 1) 자체**: `11_phase3_rotation.py`(v1)·`v2.py`·`v3.py`에만 존재하고
  전부 v4/v5로 대체돼 현재 파이프라인에서는 살아있지 않다. 다른 12개 파일 어디에도
  이 패턴은 없었다 — 전부 "raw diff 후 normalize" 또는 비율식(암묵적으로 동치)을 쓴다.
- **새로 발견 1 (결함 20 후보, 하류 오염)**: `11_phase5_curvature_rotation.py`가
  `11_phase3_rotation_raw.json`(v1, 결함18 미수정 원본)을 여전히 참조한다.
  `out/results/11_phase5_curvature_rotation.md`의 "rot_context 최댓값"/"rot_source"
  열은 v4/v5로 고쳐지지 않은 옛 수치다 — 재실행 필요.
- **새로 발견 2 (항목 2, 요주의)**: `20_family_cosine_oat.py`의 `split_half_correction`과
  `21_handle_predict_phase1.py`의 `predict_global_mean`/`predict_family_mean_oracle`이
  raw(비정규화) 벡터를 소스 간 평균한 뒤 정규화한다. 핵심 결론(Q2의 within/between,
  Q3/Q4의 B2 vs between 기준선)은 전부 "정상" 판정을 받은 다른 경로를 쓰므로 이
  결과에 오염되지 않았지만, 과제 C(family 평균 코사인)와 ①②(전역/패밀리 평균)
  베이스라인 자체의 수치는 이 방식에 의존한다 — 의도적 설계인지 확인 필요.
- **항목 3(각도 후 평균 vs 평균 후 각도) 위반은 어디에서도 발견되지 않았다** —
  결함 17과 같은 유형의 상쇄는 이번 감사 대상 파일들에서는 재발하지 않았다.

## 처리하지 않은 항목

- `20_family_cosine_oat.py` 과제 C의 raw-평균 방식과 `21_handle_predict_phase1.py`의
  두 베이스라인 함수는 "결함"으로 단정하지 않고 "요주의"로만 표시했다 — 수정 여부는
  사용자 판단이 필요하다(의도적 크기가중 평균일 가능성).
- `11_phase5_curvature_rotation.py`는 v1 참조를 v5로 갱신하는 재실행을 하지 않았다 —
  이 감사는 조사만 지시받았다.
