# Phase 5-E — 곡률(bend)-회전(rotation) 통합표

bend: 한 축 안에서 세게 걸수록 방향이 바뀌나(Phase 2). rotation: 다른 파라미터가 바뀌면 방향이 바뀌나(Phase 3, 3-1). 신규 계산 없음, 기존 산출 재사용.

| bend 축 | rotation 쌍(focus) | bend 중앙값 | bend 최댓값 | rot_context 최댓값 | rot_source | 해석 |
|---|---|---|---|---|---|---|
| reverb_wet_level | reverb_wet_room(wet_level) | 19.4° | 62.2° | 79.6° | 86.5° | 축 내부는 안정, 축 간 회전이 지배적 |
| reverb_room_size | reverb_wet_room(room_size) | 27.4° | 31.7° | 90.0° | 86.8° | 축 내부·축 간 모두 회전 큼 — 국소 손잡이 필요 |
| reverb_wet_level | reverb_wet_damping(wet_level) | 19.4° | 62.2° | 76.2° | 86.2° | 축 내부는 안정, 축 간 회전이 지배적 |
| reverb_damping | reverb_wet_damping(damping) | 26.0° | 28.1° | 90.0° | 89.9° | 축 내부는 안정, 축 간 회전이 지배적 |
| reverb_room_size | reverb_room_damping(room_size) | 27.4° | 31.7° | 80.1° | 86.7° | 축 내부는 안정, 축 간 회전이 지배적 |
| reverb_damping | reverb_room_damping(damping) | 26.0° | 28.1° | 89.8° | 89.9° | 축 내부는 안정, 축 간 회전이 지배적 |
| highshelf_gain | highshelf_gain_cutoff(gain) | 11.0° | 18.1° | 83.3° | 86.6° | 축 내부는 안정, 축 간 회전이 지배적 |
| highshelf_cutoff_gp6 | highshelf_gain_cutoff(cutoff) | 12.6° | 14.5° | 90.0° | 90.0° | 축 내부는 안정, 축 간 회전이 지배적 |
| lowshelf_gain | lowshelf_gain_cutoff(gain) | 21.4° | 22.7° | 88.5° | 89.5° | 축 내부는 안정, 축 간 회전이 지배적 |
| lowshelf_cutoff_gp6 | lowshelf_gain_cutoff(cutoff) | 33.5° | 67.3° | 90.0° | 90.0° | 축 내부·축 간 모두 회전 큼 — 국소 손잡이 필요 |
| peak_gain | peak_gain_cutoff(gain) | 10.4° | 10.9° | 87.9° | 86.2° | 축 내부는 안정, 축 간 회전이 지배적 |
| peak_cutoff_gp6 | peak_gain_cutoff(cutoff) | 23.8° | 28.2° | 90.0° | 90.0° | 축 내부는 안정, 축 간 회전이 지배적 |