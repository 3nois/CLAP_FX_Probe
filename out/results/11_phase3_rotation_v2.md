# 3-1 재실행 — 퇴화 context 제외 + 무작위 널 대조 (2026-08-19)

무작위 널(1000쌍, 512차원 단위벡터 코사인각): mean=89.97°, 95%범위=[84.98,94.77]° (이론값 90°±2.5°와 일치 확인)

**집계 규칙 폐기**: 이전 버전의 "하나라도 Branch B면 전체 Branch B" 전역 집계를 폐기한다. 쌍·방향별 개별 판정만 아래에 보고하며, Phase 5-D는 쌍 단위로 결정한다.

## 퇴화 context 제외 내역

| 쌍 | focus | 제외된 context 값 | d_A(제외 지점) |
|---|---|---|---|
| reverb_wet_room | room_size | 1개 | 0(d=-4.04e-08) |
| reverb_wet_damping | damping | 1개 | 0(d=-4.04e-08) |
| highshelf_gain_cutoff | cutoff | 1개 | 0(d=2.53e-07) |
| lowshelf_gain_cutoff | cutoff | 1개 | 0(d=4.19e-05) |
| peak_gain_cutoff | cutoff | 1개 | 0(d=1.40e-06) |

★ 게이트 구조의 정량적 재확인: EQ 쌍에서 context=gain일 때 gain=0 지점(및 인근)이 전부 퇴화로 제외됐다면 이는 Phase 0.5/3-2(a)의 gain=0 게이트 판정과 정확히 일치하는 교차검증이다.

## 판정 결과 (개별, 집계 없음)

| 쌍 | focus | context | b0(재정의) | rot_context 최대(95%CI) | rot_source(95%CI) | context vs 널 | 판정 |
|---|---|---|---|---|---|---|---|
| reverb_wet_room | wet_level | room_size | 0.85 | 79.9° [79.4,80.3] | 81.3° [81.1,81.5] | 구분됨 | **context 부차적** |
| reverb_wet_room | room_size | wet_level | 0.5 | 84.7° [84.4,84.9] | 85.1° [85.0,85.2] | 구분됨 | **context 부차적** |
| reverb_wet_damping | wet_level | damping | 0 | 76.3° [75.7,76.8] | 86.2° [86.1,86.3] | 구분됨 | **context 부차적** |
| reverb_wet_damping | damping | wet_level | 0.5 | 89.8° [89.8,89.8] | 89.9° [89.9,89.9] | **널과 구분 안 됨** | **context 무관(무작위 널과 구분 안 됨)** |
| reverb_room_damping | room_size | damping | 0 | 80.2° [79.7,80.6] | 86.3° [86.2,86.4] | 구분됨 | **context 부차적** |
| reverb_room_damping | damping | room_size | 0.85 | 89.8° [89.8,89.8] | 89.5° [89.5,89.6] | **널과 구분 안 됨** | **context 무관(무작위 널과 구분 안 됨)** |
| highshelf_gain_cutoff | gain | cutoff | 500 | 83.3° [83.2,83.4] | 84.2° [84.1,84.3] | 구분됨 | **context 부차적** |
| highshelf_gain_cutoff | cutoff | gain | -15 | 91.3° [91.2,91.3] | 89.3° [89.3,89.4] | **널과 구분 안 됨** | **context 무관(무작위 널과 구분 안 됨)** |
| lowshelf_gain_cutoff | gain | cutoff | 200 | 88.1° [87.9,88.2] | 88.2° [88.2,88.3] | **널과 구분 안 됨** | **context 무관(무작위 널과 구분 안 됨)** |
| lowshelf_gain_cutoff | cutoff | gain | 15 | 90.9° [90.9,91.0] | 89.6° [89.6,89.7] | **널과 구분 안 됨** | **context 무관(무작위 널과 구분 안 됨)** |
| peak_gain_cutoff | gain | cutoff | 825 | 88.9° [88.7,89.1] | 85.9° [85.8,86.0] | **널과 구분 안 됨** | **context 무관(무작위 널과 구분 안 됨)** |
| peak_gain_cutoff | cutoff | gain | 15 | 95.3° [95.2,95.4] | 88.7° [88.6,88.7] | 구분됨 | **context 우세** |

## rot_source 재확인 (지시 §4)

Q3(`11_phase5_q3.md`)의 within/between을 각도로 환산: within=0.34→acos(0.34)=70.1°,
between=0.24→acos(0.24)=76.1°. 무작위로 뽑은 소스쌍(대부분 다른 패밀리)의
`rot_source`는 between에 가까운 76°대여야 자연스럽다.

수정 전(v1): rot_source가 86~90°대 — between(76°)보다도 훨씬 커 무작위 널(90°)에
가까웠다. **b0가 퇴화 지점(예: EQ 쌍 gain=0)이었던 게 원인**이었다 — 모든 소스의
`v_A(b0)`가 잡음 지배적이라 소스 간 비교도 잡음끼리의 비교가 됐다.

수정 후(v2): b0를 재정의(argmax d_A)하니 `reverb_wet_room`(81~85°),
`reverb_wet_damping`(wet focus, 86°), `reverb_room_damping`(room focus, 86°),
`highshelf_gain_cutoff`(gain focus, 84°) 등 **여전히 76°(Q3의 between)보다는 크지만
90°(순수 잡음)보다는 뚜렷이 작은 중간값**으로 이동했다. 완전히 Q3와 일치하진
않는다 — 이유: Q3는 축 전체(θ_min~θ_max)의 손잡이를, `rot_source`는 특정
context(b0) 고정 시점의 손잡이를 비교하므로 다른 대상이다(같은 소스라도 특정
context 조건에서의 손잡이는 축 전체 평균 손잡이보다 소스 간 변별력이 낮을 수 있다
— 정보가 적은 단면이므로). 완전 불일치는 아니고, b0 퇴화 문제가 지배적 원인이었음을
확인했다.

## 5-D와의 교차 검증 (지시 필수)

5-D는 `{highshelf,lowshelf,peak}_gain`(context=cutoff)과 `reverb_room_size`
(context=wet_level) 4개 축쌍의 **gain/room_size focus** 방향만 테스트했다(cutoff/
wet_level을 focus로 한 반대 방향은 5-D에서 다루지 않음). 그 방향만 골라 대조한다:

| 축쌍 | 5-D 개선폭 | 5-D 해석 | v2 판정(같은 focus 방향) |
|---|---|---|---|
| highshelf_gain(context=cutoff) | +0.009 | 미미 | context 부차적 |
| lowshelf_gain(context=cutoff) | +0.012 | 미미 | context 무관 |
| peak_gain(context=cutoff) | +0.025 | 상대적으로 가장 큼(그래도 CI 겹침) | context 무관 |
| reverb_room_size(context=wet_level) | +0.023 | 상대적으로 큼(CI 겹침) | context 부차적 |

**정합적이다.** v1(수정 전)이 예측한 "context 우세"는 이 4개 축쌍 중 어디에서도
재현되지 않았다 — 전부 "부차적" 또는 "무관"으로, 5-D의 "개선폭은 있으나 미미하고
CI가 겹친다"는 관측과 방향이 일치한다. 유일하게 "context 우세"로 나온
`peak_gain_cutoff`(focus=**cutoff**, context=gain)는 애초 5-D가 테스트하지 않은
반대 방향이라 직접 대조 대상이 아니다 — 5-D를 이 방향으로도 추가 실행하면(같은
쌍을 반대 focus로) 검증 가능하나, 이번 재실행 범위 밖이다.