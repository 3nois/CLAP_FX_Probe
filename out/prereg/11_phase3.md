# Phase 3분석 사전 등록 — 축은 독립인가

날짜: 2026-08-18. 실행 전 확정. §3-1 판정표는 결과를 본 뒤 수정하지 않는다.

무결성 검증: `out/results/11_phase3_integrity.md` — **PASS** (10개 파일 shape·src_id
0~1199·NaN/Inf 전부 정상, Phase 3분석 착수 가능).

---

## 3-1. 손잡이 회전 지도 — 판정표 (★ 최우선, 지시 원문 그대로 확정)

| 조건 | 판정 | Phase 5 |
|---|---|---|
| `rot_context` 95% 상한 < `rot_source` 95% 하한 | context 부차적 | **Branch A** |
| CI 가 겹침 | 대등 | **Branch B** |
| `rot_context` 95% 하한 > `rot_source` 95% 상한 | context 우세 | **Branch B** + 손잡이 정의 재검토 |

### 계산 방법 확정 (실행 전 고정)

- 대상: 2-D 페어 6개 × (focus, context) 방향 2가지 = **12개 검정**
  - reverb_wet_room: (focus=wet,context=room), (focus=room,context=wet)
  - reverb_wet_damping: (focus=wet,context=damping), (focus=damping,context=wet)
  - reverb_room_damping: (focus=room,context=damping), (focus=damping,context=room)
  - {highshelf,lowshelf,peak}_gain_cutoff: (focus=gain,context=cutoff), (focus=cutoff,context=gain)
- `b₀` = context 축의 중앙 격자점(13레벨 중 index 6)
- `v_A(b) = normalize(e[:,i_max,b,:]) - normalize(e[:,i_min,b,:])` 소스별 L2 정규화 후 차분(1200,512)
- `rot_context(b) = arccos(clip(cos(v_A(b₀)_s, v_A(b)_s), -1, 1))`, 소스별로 계산 후 소스 단위
  부트스트랩(n_boot=2000, seed=0)으로 평균 + 95% CI
- `rot_source`: 같은 b₀에서 `v_A(b₀)`를 소스 간에 비교 — 무작위 소스쌍 5,000개(seed=0,
  자기 자신 제외, 복원추출) 표본의 각도 분포를 같은 부트스트랩 설정(n_boot=2000, seed=0)으로
  집계
- 판정은 각 검정(12개)마다 개별로 낸다. **전체 집계 규칙(사전 확정)**: 12개 중
  하나라도 Branch B(대등 또는 context 우세) 판정이면 프로젝트 전체를 **Branch B**로
  진행한다 — 보수적 규칙이다(어느 축에서든 context가 무시할 수 없는 수준이면 그
  축만이라도 조건부 노출이 필요하므로 5-D를 건너뛸 수 없다). 개별 쌍·방향별 판정은
  표에 전부 남겨 3-4의 파라미터별 분류에도 그대로 쓴다.

### 예측

축마다 이질적일 것으로 예측한다 — 특히 Phase 3(렌더링)의 분산분해에서 이미 강한
상호작용이 확인된 쌍(`lowshelf_gain_cutoff` 89.5%, `highshelf_gain_cutoff` 65.0%,
`reverb_wet_room` 63.2%)은 **context 우세**로, 상호작용이 약했던 쌍(`peak_gain_cutoff`
5.7%, `reverb_wet_damping` 0.6%, `reverb_room_damping` 4.8%)은 **context 부차적**로
나올 것으로 예측한다. 전체 집계 규칙상 하나라도 우세가 나오면 Branch B이므로,
**Branch B로 귀결될 가능성이 높다고 예측**한다(이미 알려진 상호작용 크기 때문).

---

## 3-2(a). 게이트 sanity check — 사전 확정 중단 조건

EQ 2-D 페어(`{highshelf,lowshelf,peak}_gain_cutoff`)에서 `gain=0`(13레벨 중
index 6) 행은 cutoff 전체에 걸쳐 변위가 **널 바닥 수준으로 평평해야 한다**
(Phase 0.5에서 확인한 gain=0 진짜 dry 판정과 일치해야 함). 평평하지 않으면
Phase 0.5의 게이트 판정 자체가 틀렸다는 뜻이므로 **즉시 중단하고 사람에게 보고**한다
— 3-2(b)·3-3·3-4를 진행하지 않는다.

## 3-3. 교차 확인 — 사전 확정 기준

3-2의 2-D 히트맵에서 읽은 2차 상호작용(13×13)과 3-3 ANOVA의 2차 항(5레벨 격자)이
**순서**(어느 쌍이 더 강한 상호작용인지)와 **대략적 크기**(같은 자릿수)에서 일치해야
한다. 격자 해상도가 달라 정확히 같은 수치는 기대하지 않는다. 어긋나면(순서 역전 또는
자릿수 차이) 결함으로 보고한다.

## 3-4. 도구 사양 분류 기준 (사전 확정)

3-2(b)의 R²(A|B) 13개 값의 범위와 3-3의 주효과/2차 분산비율을 아래 기준으로 분류한다
(임의적 절선이므로 명시): 주효과 분산비율 ≥ 50%이고 2차 상호작용 < 30% → **독립
노출**; 주효과 < 30%이고 2차 이상 상호작용 ≥ 50% → **조건부 노출**; 그 외(둘 다
30~50% 애매 구간 포함) → **노출 보류, 추가 판단 필요**로 표기한다.

---

## 실행 계획

- 3-1·3-2·3-3·3-4 순서로 실행, 캐시만 사용(렌더링 없음), 예상 1~2시간
- 3-2(a) 게이트 sanity check 위반 시 즉시 중단
- 산출: `out/results/11_phase3_rotation.md`, `out/figures/11_phase3_rotation_<pair>.pdf`,
  `out/results/11_phase3_2d.md`, `out/figures/11_phase3_2d_<pair>.pdf`,
  `out/results/11_phase3_ui_spec.md`, ANOVA 표(3-3, `out/results/11_phase3_anova.md`)
- 완료 후 보고 → Branch A/B 승인 → Phase 5 착수

**본 문서 승인 후 3-1~3-4를 실행한다.**

---

## 3-1 v3 addendum (2026-08-19, 사람 지시로 발견된 결함 반영)

v2는 signed context 축(EQ의 gain, −15~+15)에서 부호별로 손잡이가 반대 방향을 향할
수 있다는 것을 반영하지 못했다 — 두 부호 분기를 하나의 분포로 섞어 평균 내면
반대 방향 벡터끼리 상쇄돼 ~90°(무작위 널과 구분 안 됨)로 나온다. 이는 1차
실험(round 1)의 결함 2(부호별 방향 처리 오류)와 같은 종류의 실수다.

**v3 방법(실행 전 확정)**:
1. context 축이 signed인 경우(3개 EQ 쌍의 focus=cutoff, context=gain 방향만
   해당) gain>0/gain<0 두 분기로 나눠 각각 b0=argmax‖v‖를 잡고 분기 내부에서만
   rot_context를 낸다. 분기 간 비교는 "부호 대칭"이라는 별도 지표로 보고한다
   (회전으로 해석하지 않는다).
2. unsigned context 축(reverb 3쌍, EQ의 focus=gain·context=cutoff 방향)은 v2
   결과를 그대로 유지 — 재계산하지 않는다.
3. **다중비교 보정(사전 확정)**: v2의 9개 unsigned 검정 + v3의 6개 분기 검정 =
   15개 "vs 널" 판정에 **Bonferroni 보정**을 적용한다(가장 보수적이고 정당화가
   쉬운 방법을 택함 — FDR 대신). 보정 후 유의수준 = 0.05/15 ≈ 0.00333, 즉 널
   대역을 [0.167, 99.833] 백분위로 넓혀 판정한다. 근소하게 널을 벗어나는 결과는
   보정 후 유의하지 않을 수 있음을 명시한다.

**예측(실행 전)**: v2의 "context 무관" 판정 중 signed-context 테스트(3개 EQ 쌍의
cutoff-focus 방향)는 분기별로 재계산하면 대부분 **"context 부차적"**로 바뀔
것으로 예측한다(사용자가 제시한 참고 수치: 분기 내부 최대 회전이 highshelf
+32.8°/−37.0°, lowshelf +30.7°/−25.1°, peak +23.9°/−25.8°로 무작위 널
[85,95]°에서 뚜렷이 벗어남 — 실재하는 정렬이지만 이것이 rot_source 대비 우세할
정도로 크지는 않을 것으로 예측).

결함 17: "signed context 축에서 부호 분기를 섞으면 회전이 90°로 상쇄되어
무관함으로 오판된다 — 1차 결함 2(부호 처리 오류)의 재발."

## 3-1 v4 addendum (2026-08-22, 세션 소실 후 재개 — 결함 18)

v3를 실행했더니 위 예측(부호 분기 내부 최대 회전 highshelf +32.8/−37.0,
lowshelf +30.7/−25.1, peak +23.9/−25.8)과 달리 signed 6건 전부 88~90°로
나와 "context 무관"으로 판정됐다 — 예측이 재현되지 않았다.

코드 감사 결과, v3(및 v2)의 회전 벡터 정의 `v_b = normalize(e_max) -
normalize(e_min)`(끝점을 각각 단위벡터화한 뒤 차분)가 원인이었다. b0(최강
효과 지점, |gain|=15) 대비 효과가 약한 gain 지점에서는 두 끝점의 반경 방향
잡음이 실제 효과 방향보다 커져 v_b가 사실상 무작위가 된다. 벡터 정의를
`v_b = normalize(e_max - e_min)`(먼저 차분한 뒤 정규화)로 바꾸자 위 예측
수치와 소수점 단위까지 일치했다(11_phase3_rotation_v4.py로 검증).

인수인계 문서(11차, 세션 소실 후 작성)는 이 결함을 "gain=0이 부호 양쪽
분기에 재포함된다"고 서술했으나, 코드 감사 결과 v3.py는 이미 `g1[b]>0` /
`g1[b]<0`로 0을 배타적으로 제외하고 있었다 — 서술된 메커니즘은 실제 코드와
맞지 않았다(0 포함 시 90.0°, 제외 시 88.7°로 거의 무변화— 예측한 23.9°와는
무관). 다만 그 문서가 인용한 "실측 정답" 수치 자체는 정확했고, 위 벡터 정의
수정으로 재현됐다. 즉 결함은 실재했으나(사전 예측 수치가 이를 증명), 소실된
세션과 인수인계 문서 사이에서 결함의 *메커니즘 설명*만 어긋난 것으로 보인다.

결함 18: "signed 회전 벡터를 끝점 개별 정규화 후 차분(`normalize(e_max) -
normalize(e_min)`)으로 정의하면, b0 대비 약한 효과 지점에서 반경 잡음이
방향 신호를 압도해 회전이 무작위 널과 구분되지 않는다 — 차분 후 정규화
(`normalize(e_max - e_min)`)로 고쳐야 한다."

**결과(실행 후)**: signed 6건 전부 "context 무관" → **"context 부차적"**로
뒤집혔다(예측과 일치, `out/results/11_phase3_rotation_v4.md`). 부호 대칭
결론(부스트/컷은 별개 손잡이)은 v3와 동일하게 유지된다.

**미확인 사항(당시 결함 19 후보, 아래에서 확정)**: unsigned 9건과 Q3/Q4 등
다른 스크립트도 같은 `normalize(e_max) - normalize(e_min)` 벡터 정의를 쓴다.
당시 수정 범위는 인수인계 지시대로 signed 6건에 한정했고, unsigned/Q3/Q4에
같은 문제가 있는지는 확인하지 않았다.

## 3-1 v5 addendum (2026-08-22, 결함 19)

v4는 "unsigned 9건은 부호 문제 없음"이라며 v3 결과를 그대로 유지했다. 사용자
지시로 이 면제가 무효라는 것이 확인됐다 — 결함 18의 실제 원인(정규화 순서)은
부호 분기와 무관하므로 unsigned 9건도 영향권에 있다.

**v5 방법(실행 전 확정)**: v2의 `run_pair_direction`(퇴화 판정·b0 선정 로직은
그대로) 안에서 회전 벡터만 `normalize(e_max - e_min)`으로 바꿔 unsigned 9건을
재산출한다. rot_source도 같은 v_b0를 쓰므로 함께 재산출된다. signed 6건은
v4를 그대로 재사용(이미 올바른 정의로 계산됨).

**결과(실행 후)**: unsigned 9건 중 5/9건이 판정을 바꿨다 — 전부 "context
무관" → "context 부차적"으로 뒤집혔다(reverb_wet_room room_size-focus,
reverb_wet_damping damping-focus, reverb_room_damping damping-focus,
highshelf_gain_cutoff gain-focus, lowshelf_gain_cutoff gain-focus).
peak_gain_cutoff(gain-focus)만 재산출 후에도 "context 무관"으로 남았다.
나머지 3건(wet_room wet_level-focus, wet_damping wet_level-focus,
room_damping room_size-focus)은 v3에서도 이미 "context 부차적"이었고 v5에서도
유지됐다(각도 값 자체는 바뀌었으나 판정 문턱을 넘지 않음). 상세는
`out/results/11_phase3_rotation_v5.md` 참고.

결함 19: "결함 18을 signed 6건에만 적용하고 unsigned 9건은 원인을 '부호 분기
경계'로 오진해 면제했다 — 실제 원인은 정규화 순서 자체로 부호와 무관했고,
수정 범위가 실제 영향 범위보다 좁게 설정됐다."

**갱신된 3-1 결론**: 15개 검정(unsigned 9 + signed 6) 중 이제 14건이 "context
부차적"(cutoff/focus 방향의 손잡이가 실재하며, context 값에 따라 방향이 다소
흔들리지만 소스 간 변동보다는 작다), 1건("peak_gain_cutoff, focus=gain")만
"context 무관"이다. signed 6건은 전부 부차적로 일관되고, unsigned 9건 중 8건도
부차적이다. Phase 5-D "context 조건 불필요" 판단(11차 인수인계 §5)은 이 갱신된
결과를 기준으로 재확인해야 한다 — 15건 중 14건에서 context가 부차적으로나마
영향을 준다는 것이 확인됐으므로, "완전히 무시 가능"보다는 "focus 방향이
주도적이되 context 미세조정이 남아 있을 수 있다"는 결론에 더 가깝다.
