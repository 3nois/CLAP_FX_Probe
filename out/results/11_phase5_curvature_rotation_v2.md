# Phase 5-E v2 — 곡률(bend)-회전(rotation) 통합표 (결함 20 수정, 2026-08-22)

**결함 20**: 원본(`11_phase5_curvature_rotation.md`)은 `11_phase3_rotation_raw.json`(v1, 결함18 미수정 — 회전 벡터를 `normalize(e_max)-normalize(e_min)`로 정의해 약한 효과 지점에서 잡음에 압도되는 버전)을 참조하고 있었다. v2~v5로 rotation 수치가 갱신되는 동안 이 파일만 누락됐다 — **하류 오염 사례**. 원본 파일은 재인용 금지로 표시한다. 이 파일은 v5(unsigned 9건)와 v4(signed 6건, 부호 분기별)의 최신 산출을 그대로 재사용한다(신규 rotation 계산 없음, bend만 재사용).

bend: 한 축 안에서 세게 걸수록 방향이 바뀌나(Phase 2). rotation: 다른 파라미터가 바뀌면 방향이 바뀌나(Phase 3, 3-1 v4/v5). cutoff-focus(EQ 3쌍)는 부호 분기가 있어 gain+/gain- 두 행으로 나눠 보고한다.

| bend 축 | rotation 쌍(focus/분기) | bend 중앙값 | bend 최댓값 | rot_context 최댓값 | rot_source | rotation 판정 | 해석 |
|---|---|---|---|---|---|---|---|
| reverb_wet_level | reverb_wet_room(wet_level) | 19.4° | 62.2° | 51.4° | 70.3° | context 부차적 | 축 내부 회전이 축 간보다 큼 — 구간 세분화가 더 중요 |
| reverb_room_size | reverb_wet_room(room_size) | 27.4° | 31.7° | 55.9° | 70.9° | context 부차적 | 축 내부는 안정, 축 간 회전이 지배적 |
| reverb_wet_level | reverb_wet_damping(wet_level) | 19.4° | 62.2° | 16.5° | 74.3° | context 부차적 | 축 내부 회전이 축 간보다 큼 — 구간 세분화가 더 중요 |
| reverb_damping | reverb_wet_damping(damping) | 26.0° | 28.1° | 74.0° | 83.4° | context 부차적 | 축 내부는 안정, 축 간 회전이 지배적 |
| reverb_room_size | reverb_room_damping(room_size) | 27.4° | 31.7° | 29.0° | 71.2° | context 부차적 | 축 내부 회전이 축 간보다 큼 — 구간 세분화가 더 중요 |
| reverb_damping | reverb_room_damping(damping) | 26.0° | 28.1° | 75.9° | 80.4° | context 부차적 | 축 내부는 안정, 축 간 회전이 지배적 |
| highshelf_gain | highshelf_gain_cutoff(gain) | 11.0° | 18.1° | 40.1° | 68.7° | context 부차적 | 축 내부는 안정, 축 간 회전이 지배적 |
| lowshelf_gain | lowshelf_gain_cutoff(gain) | 21.4° | 22.7° | 37.9° | 76.7° | context 부차적 | 축 내부는 안정, 축 간 회전이 지배적 |
| peak_gain | peak_gain_cutoff(gain) | 10.4° | 10.9° | 82.6° | 71.9° | context 무관(보정 후 널과 구분 안 됨) | 축 내부는 안정, 축 간 회전이 지배적 |
| highshelf_cutoff_gp6 | highshelf_gain_cutoff(cutoff, gain+) | 12.6° | 14.5° | 32.8° | 72.7° | context 부차적 | 축 내부는 안정, 축 간 회전이 지배적 |
| highshelf_cutoff_gp6 | highshelf_gain_cutoff(cutoff, gain-) | 12.6° | 14.5° | 37.0° | 75.3° | context 부차적 | 축 내부는 안정, 축 간 회전이 지배적 |
| lowshelf_cutoff_gp6 | lowshelf_gain_cutoff(cutoff, gain+) | 33.5° | 67.3° | 30.7° | 78.4° | context 부차적 | 축 내부 회전이 축 간보다 큼 — 구간 세분화가 더 중요 |
| lowshelf_cutoff_gp6 | lowshelf_gain_cutoff(cutoff, gain-) | 33.5° | 67.3° | 25.1° | 79.0° | context 부차적 | 축 내부 회전이 축 간보다 큼 — 구간 세분화가 더 중요 |
| peak_cutoff_gp6 | peak_gain_cutoff(cutoff, gain+) | 23.8° | 28.2° | 23.9° | 79.5° | context 부차적 | 축 내부 회전이 축 간보다 큼 — 구간 세분화가 더 중요 |
| peak_cutoff_gp6 | peak_gain_cutoff(cutoff, gain-) | 23.8° | 28.2° | 25.8° | 80.7° | context 부차적 | 축 내부 회전이 축 간보다 큼 — 구간 세분화가 더 중요 |