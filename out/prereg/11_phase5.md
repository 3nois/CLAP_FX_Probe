# Phase 5 사전 등록 — 손잡이 구조와 예측 (Q3 · Q4)

날짜: 2026-08-18. Phase 3분석 결과 **Branch B**로 확정됨(`out/results/11_phase3_rotation.md`,
12개 검정 중 6개 Branch B) — 5-D(context 추가) 실행 확정.

데이터: Phase 2 밀집 격자(`out/caches/11_phase2_<axis>.npz` + `ext`, concat 1,200소스,
25레벨) — 6~9차의 3레벨 OAT 임베딩(`oat_emb.npz`)이 아니다. 재사용 대상은
`20_family_cosine_oat.py`/`21_handle_predict_phase1.py`의 **함수**(within/between
부트스트랩, split-half 보정, `MLPDualHead`, `stratified_split`, `run_all_models`)이며
이를 25레벨 데이터에 적용한다(재구현 아님 — `importlib`로 직접 import).

## 축 선정 (규모 제한, 사전 확정)

21개 주축 전부를 돌리면 계산량이 지나치게 크다. 대표 5축으로 축소한다:
`distortion_drive_db`, `reverb_room_size`(4차 R8 이후 reverb 대표축 관행 계승),
`highshelf_gain`, `lowshelf_gain`, `peak_gain`(3 EQ 타입 비교가 이번 라운드 핵심
질문이므로 포함).

## 5-A. 구간 정의 (사전 확정)

```
전 범위    [θ_1, θ_25]                index 0 -> 24            1개
3분할      [θ_1,θ_9] [θ_9,θ_17] [θ_17,θ_25]   index (0,8)(8,16)(16,24)   3개
인접(표본)  하/중/상 1/3에서 대표 1쌍씩          index (2,3)(12,13)(21,22)  3개
```

★ **인접 구간 축소 근거**: 원 설계는 24쌍 전부지만, 이 질문("약하게 건 구간에서
방향이 잡히는가")은 5-E의 bend 각도(Phase 2, 이미 계산됨, 24쌍 전부 존재)가 이미
연속적으로 답하고 있다. Q3/Q4는 신경망 학습이 들어가 24쌍×5축×3조건×2모델을 전부
돌리면 계산량 대비 한계효용이 낮다고 판단해, 하/중/상 1/3 대표 1쌍씩만 표본으로
Q3/Q4를 검증하고 나머지는 5-E의 연속 곡선으로 보완한다.

→ 총 축당 7개 구간 × 5축 = **35개 handle-정의**.

## 5-B. Q3 — 예측 (사전 확정)

35개 handle-정의마다 within/between/gap을 낸다. **예측**: 구간이 좁아질수록(3분할 →
인접) within이 낮아질 것이다 — 좁은 구간일수록 절대 변위가 작아 방향의 신호 대비
잡음비가 나빠지고, 소스 고유 구조보다 측정 잡음이 상대적으로 커질 것으로 예상한다.
전 범위에서 within이 가장 높고 인접 구간에서 가장 낮을 것으로 예측.

방법(재사용): `bootstrap_within_between`(source-level bootstrap, n_boot=300),
`split_half_correction`(n_reps=100) — `20_family_cosine_oat.py`에서 import.

## 5-C. Q4 — 예측 (사전 확정)

정방향(A)/B1(known)/B2(unknown) 3조건 × {global_mean, family_mean_oracle, linear, mlp}
4모델. **기준선은 반드시 5-B의 between 값**(같은 handle-정의에서). **예측**: 전 범위·
3분할에서는 B2(mlp)가 between 기준선을 넘을 것이나(8차 전례), 인접 구간(표본 3개)
에서는 신호가 약해 기준선을 넘지 못하거나 넘더라도 격차가 좁아질 것으로 예측한다.

방법(재사용): `stratified_split`(패밀리 층화 80/10/10), `MLPDualHead`,
`LinearDualHead`, `run_all_models`, `dual_head_loss` — `21_handle_predict_phase1.py`에서
import. 하이퍼파라미터는 원 스크립트 기본값 유지(max_epochs=300, patience=30,
lambda_mag=0.3, mlp_hidden=1024/dropout=0.1, seed=0).

## 5-D. Branch B — context 추가 (실행 확정)

Phase 3분석이 Branch B이므로 실행한다. 범위: 계산량 제한을 위해 **전 범위(1개
handle-정의)에 한해**, Phase 3의 2-D 캐시가 있는 4개 축-쌍만 적용한다:
`{highshelf,lowshelf,peak}_gain`(context=해당 축의 cutoff, `{type}_gain_cutoff.npz`
재사용), `reverb_room_size`(context=wet_level, `reverb_wet_room.npz` 재사용, focus를
room으로 고정 — Phase 3에서 이미 렌더링된 조합이라 별도 렌더 불필요). context
벡터는 2-D 캐시에서 focus=전범위 끝점, context=13레벨 각각을 뽑아 소스별로 이어붙인
입력([e(θ_min); context_onehot_or_value])으로 구성한다 — 세부 설계는 실행 스크립트에
기록한다.

**예측**: context를 알려주면 B2(mlp)의 cos가 상승할 것이다 — Phase 3분석에서 이
4축-쌍 전부가 어느 정도 상호작용을 보였으므로(3-3 ANOVA), context 정보가 실제로
예측에 도움이 될 것으로 예측한다.

## 5-E. 곡률-회전 통합표

Phase 2 bend(`out/results/11_phase2_bend_signedjnd.md`)와 Phase 3 rotation
(`out/results/11_phase3_rotation.md`)을 같은 표에 놓는다. 신규 계산 없음 — 기존
산출 재사용·병기만 한다.

## 5-F. 미해결분 — 이미 완료됨

`11_phase2_cascade_relabel.md`(cascade 재라벨링, 결함 15), `11_phase2_bend_signedjnd.md`
+ `11_phase2jnd_final.md`(부호축 JND 방향분리)로 이미 처리됨. 재실행 없음, 링크만 건다.

---

## 실행 순서 및 산출

1. 5-B (35개 handle-정의 × within/between/gap) → `out/results/11_phase5_q3.md`
2. 5-C (35개 × 3조건 × 4모델) → `out/results/11_phase5_q4.md`
3. 5-D (Branch B, 4축-쌍 × 전범위만) → `out/results/11_phase5_q4_context.md`
4. 5-E (병기표) → `out/results/11_phase5_curvature_rotation.md`
5. 5-F (링크만) → `out/results/11_phase5_unresolved.md`

예상 소요: 수 시간(신경망 학습 다수, 그러나 소규모 데이터라 회당 수 초~수십 초).
**본 문서 승인 후 실행한다.**
