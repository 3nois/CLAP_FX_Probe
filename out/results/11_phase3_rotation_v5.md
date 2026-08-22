# 3-1 v5 — 결함 19 수정: unsigned 9건도 v4 벡터 정의로 재산출 (2026-08-22)

## 사전 등록 (실행 전 기록)

**결함 19**: v4는 "unsigned 9건은 v3 유지"라고 면제했다. 그 근거는 결함 18의 원인을 "signed 부호 분기 경계"로 좁게 진단한 데 있었다 — 실제 원인(회전 벡터를 `normalize(e_max)-normalize(e_min)`로 정의해 약한 효과 지점에서 반경 잡음이 방향을 압도하는 것)은 부호와 무관하며 unsigned 9건도 같은 정의를 쓴다. 잘못된 원인 진단에 근거해 수정 범위를 실제 영향 범위보다 좁게 잡은 사례다.

**예측**: unsigned 9건 중 이미 v3에서 "context 부차적/우세/대등"으로 (널과 구분되어) 나온 항목은 각도 값은 바뀌어도 판정이 유지될 가능성이 높다(원래도 방향 신호가 잡음보다 컸다는 뜻이므로). 반대로 v3에서 "context 무관"으로 나온 항목 중 일부는 결함 18과 같은 메커니즘으로 signed-6건처럼 "무관" → "부차적"로 뒤집힐 수 있다. 어느 쪽이든 실행 후 표로 대조한다.

## unsigned 9개 — v5 재산출(결함 19 수정)

| 쌍 | focus | context | rot_context 최대(95%CI, context값) | rot_source(95%CI) | 판정(v5) | 판정(v3, 참고) | 뒤집힘 |
|---|---|---|---|---|---|---|---|
| reverb_wet_room | wet_level | room_size | 51.4° [50.4,52.3] (val=0.05) | 70.3° [69.9,70.6] | **context 부차적** | context 부차적 | ✗ |
| reverb_wet_room | room_size | wet_level | 55.9° [54.8,57.0] (val=0.0417) | 70.9° [70.5,71.3] | **context 부차적** | context 무관(보정 후 널과 구분 안 됨) | ✓ |
| reverb_wet_damping | wet_level | damping | 16.5° [15.9,17.0] (val=1) | 74.3° [73.9,74.6] | **context 부차적** | context 부차적 | ✗ |
| reverb_wet_damping | damping | wet_level | 74.0° [72.9,75.1] (val=0.0417) | 83.4° [83.1,83.8] | **context 부차적** | context 무관(보정 후 널과 구분 안 됨) | ✓ |
| reverb_room_damping | room_size | damping | 29.0° [28.0,30.0] (val=1) | 71.2° [70.8,71.6] | **context 부차적** | context 부차적 | ✗ |
| reverb_room_damping | damping | room_size | 75.9° [75.0,76.7] (val=0.05) | 80.4° [80.1,80.8] | **context 부차적** | context 무관(보정 후 널과 구분 안 됨) | ✓ |
| highshelf_gain_cutoff | gain | cutoff | 40.1° [39.4,40.7] (val=4e+03) | 68.7° [68.3,69.1] | **context 부차적** | context 무관(보정 후 널과 구분 안 됨) | ✓ |
| lowshelf_gain_cutoff | gain | cutoff | 37.9° [36.9,38.9] (val=30) | 76.7° [76.3,77.2] | **context 부차적** | context 무관(보정 후 널과 구분 안 됨) | ✓ |
| peak_gain_cutoff | gain | cutoff | 82.6° [81.4,83.8] (val=6e+03) | 71.9° [71.5,72.3] | **context 무관(보정 후 널과 구분 안 됨)** | context 무관(보정 후 널과 구분 안 됨) | ✗ |

**대조 결과**: unsigned 9건 중 5/9건이 v3 대비 판정이 바뀌었다.

## signed 6개 — v4 그대로 재사용(이미 이 벡터 정의로 계산됨, 변경 없음)

| 쌍 | 분기 | rot_context 최대(95%CI) | rot_source | 판정 |
|---|---|---|---|---|
| highshelf_gain_cutoff | gain+ | 32.8° [32.4,33.3] | 72.7° [72.3,73.0] | **context 부차적** |
| highshelf_gain_cutoff | gain- | 37.0° [36.2,37.9] | 75.3° [75.0,75.8] | **context 부차적** |
| lowshelf_gain_cutoff | gain+ | 30.7° [29.5,32.2] | 78.4° [78.0,78.9] | **context 부차적** |
| lowshelf_gain_cutoff | gain- | 25.1° [24.3,26.1] | 79.0° [78.5,79.5] | **context 부차적** |
| peak_gain_cutoff | gain+ | 23.9° [23.5,24.4] | 79.5° [79.1,80.0] | **context 부차적** |
| peak_gain_cutoff | gain- | 25.8° [25.1,26.5] | 80.7° [80.3,81.2] | **context 부차적** |

## 결함 19

> v4는 결함 18(회전 벡터 정의 `normalize(e_max)-normalize(e_min)`가 약한 효과 지점에서 잡음에 압도됨)을 signed 6건에만 적용하고 unsigned 9건은 "부호 문제 없음"이라며 v3 결과를 그대로 유지했다. 이는 결함 18의 원인을 "부호 분기 경계 처리"로 오진한 데서 비롯된 조치였다 — 실제 원인은 정규화 순서 자체이고 부호 유무와 무관하므로, unsigned 9건도 같은 결함의 영향권에 있었다. 잘못된 원인 진단에 근거해 수정 범위를 실제 영향 범위보다 좁게 설정한 사례다. 1차 결함 1(지표 불일치)과 마찬가지로 '부분적으로만 맞는 진단이 불완전한 수정으로 이어진' 유형이다.
