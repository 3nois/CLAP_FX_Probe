# 3-1 v3 — signed context 부호 분기 분리 + Bonferroni 보정 (2026-08-19)

Bonferroni 보정: 15개 검정(unsigned 9 + signed 분기 6) 기준 alpha=0.003333, 널 대역=[82.60°,96.45°]

## unsigned 9개 — v2 재사용, Bonferroni 재판정

| 쌍 | focus | context | rot_context 최대(95%CI) | rot_source | 보정 후 널과 구분 | 판정(보정) |
|---|---|---|---|---|---|---|
| reverb_wet_room | wet_level | room_size | 79.9° [79.4,80.3] | 81.3° [81.1,81.5] | 구분됨 | **context 부차적** |
| reverb_wet_room | room_size | wet_level | 84.7° [84.4,84.9] | 85.1° [85.0,85.2] | **구분 안 됨** | **context 무관(보정 후 널과 구분 안 됨)** |
| reverb_wet_damping | wet_level | damping | 76.3° [75.7,76.8] | 86.2° [86.1,86.3] | 구분됨 | **context 부차적** |
| reverb_wet_damping | damping | wet_level | 89.8° [89.8,89.8] | 89.9° [89.9,89.9] | **구분 안 됨** | **context 무관(보정 후 널과 구분 안 됨)** |
| reverb_room_damping | room_size | damping | 80.2° [79.7,80.6] | 86.3° [86.2,86.4] | 구분됨 | **context 부차적** |
| reverb_room_damping | damping | room_size | 89.8° [89.8,89.8] | 89.5° [89.5,89.6] | **구분 안 됨** | **context 무관(보정 후 널과 구분 안 됨)** |
| highshelf_gain_cutoff | gain | cutoff | 83.3° [83.2,83.4] | 84.2° [84.1,84.3] | **구분 안 됨** | **context 무관(보정 후 널과 구분 안 됨)** |
| lowshelf_gain_cutoff | gain | cutoff | 88.1° [87.9,88.2] | 88.2° [88.2,88.3] | **구분 안 됨** | **context 무관(보정 후 널과 구분 안 됨)** |
| peak_gain_cutoff | gain | cutoff | 88.9° [88.7,89.1] | 85.9° [85.8,86.0] | **구분 안 됨** | **context 무관(보정 후 널과 구분 안 됨)** |

## signed 3쌍 — 부호 분기 분리 (focus=cutoff, context=gain)

| 쌍 | 분기 | b0(gain) | rot_context 최대(95%CI) | rot_source | 보정 후 널과 구분 | 판정(보정) |
|---|---|---|---|---|---|---|
| highshelf_gain_cutoff | gain+ | 15 | 89.6° [89.6,89.6] | 89.3° [89.3,89.3] | **구분 안 됨** | **context 무관(보정 후 널과 구분 안 됨)** |
| highshelf_gain_cutoff | gain- | -15 | 89.6° [89.6,89.6] | 89.3° [89.3,89.4] | **구분 안 됨** | **context 무관(보정 후 널과 구분 안 됨)** |
| lowshelf_gain_cutoff | gain+ | 15 | 89.7° [89.7,89.8] | 89.6° [89.6,89.7] | **구분 안 됨** | **context 무관(보정 후 널과 구분 안 됨)** |
| lowshelf_gain_cutoff | gain- | -15 | 89.8° [89.8,89.8] | 89.8° [89.7,89.8] | **구분 안 됨** | **context 무관(보정 후 널과 구분 안 됨)** |
| peak_gain_cutoff | gain+ | 15 | 88.7° [88.7,88.8] | 88.7° [88.6,88.7] | **구분 안 됨** | **context 무관(보정 후 널과 구분 안 됨)** |
| peak_gain_cutoff | gain- | -15 | 88.8° [88.8,88.9] | 88.9° [88.9,89.0] | **구분 안 됨** | **context 무관(보정 후 널과 구분 안 됨)** |

## 부호 대칭 (별도 지표 — 회전으로 해석하지 않음)

| 쌍 | cos(v_gain+, v_gain-) | 각도 환산 | 해석 |
|---|---|---|---|
| highshelf_gain_cutoff | -0.499 [-0.506,-0.491] | 119.9° | 부호 반전 — 부스트/컷을 별개 손잡이로 다뤄야 함 |
| lowshelf_gain_cutoff | -0.684 [-0.693,-0.675] | 133.2° | 부호 반전 — 부스트/컷을 별개 손잡이로 다뤄야 함 |
| peak_gain_cutoff | -0.758 [-0.763,-0.753] | 139.3° | 부호 반전 — 부스트/컷을 별개 손잡이로 다뤄야 함 |

## 다중비교 명시

15개 검정을 95% 널 대역(보정 전)으로 개별 판정하면 우연 기대 오탐 수 = 15×0.05 = 0.75건. v2에서 근소 초과(peak_gain_cutoff cutoff-focus, 95.3° vs 널 상한 94.77°)는 이 우연 기대값 범위 안에 있어 보정 전 기준으로도 단정하기 어려웠다. Bonferroni 보정(alpha=0.003333) 적용 후 재판정한 결과는 위 표에 반영했다.

## 결함 17

> signed context 축(EQ gain)에서 부호 분기를 섞으면 반대 방향 벡터끼리 상쇄돼 회전이 ~90°(무작위 널과 구분 안 됨)로 나와 '무관'으로 오판된다 — 1차 실험 결함 2(부호별 방향 처리 오류)의 재발이다. v2에서 이 문제로 오판됐던 3개 EQ 쌍의 cutoff-focus 방향을 부호 분기별로 재계산해 바로잡았다.
